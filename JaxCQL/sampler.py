import numpy as np

from .replay_buffer import calc_return_to_go


ADROIT_ENVS = {
    "pen-binary-v0",
    "door-binary-v0",
    "relocate-binary-v0",
    "pen-binary",
    "door-binary",
    "relocate-binary",
}


class TrajSampler(object):
    """Collect trajectories from either legacy Gym or Gymnasium environments."""

    def __init__(
        self,
        env,
        use_goal=False,
        use_mc=False,
        gamma=0.99,
        reward_scale=1.0,
        reward_bias=0.0,
    ):
        self._env = env
        self.use_goal = use_goal
        self.use_mc = use_mc
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.max_traj_length = env.spec.max_episode_steps

    def sample(self, policy, n_trajs, deterministic=False, replay_buffer=None):
        trajs = []
        for _ in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []
            if self.use_goal:
                goal_achieved_list = []

            reset_result = self.env.reset()
            observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result

            for _ in range(self.max_traj_length):
                action = policy(
                    observation.reshape(1, -1), deterministic=deterministic
                ).reshape(-1)

                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_observation, reward, terminated, truncated, env_infos = step_result
                    done = terminated or truncated
                else:
                    next_observation, reward, done, env_infos = step_result

                if self.use_goal:
                    goal_achieved = int(bool(env_infos["goal_achieved"]))
                    goal_achieved_list.append(goal_achieved)
                    if goal_achieved:
                        done = True

                observations.append(observation)
                actions.append(action)
                rewards.append(reward * self.reward_scale + self.reward_bias)
                next_observations.append(next_observation)
                dones.append(done)
                observation = next_observation

                if done:
                    break

            if self.use_mc:
                env_spec = getattr(self.env, "spec", None)
                env_id = getattr(env_spec, "id", None) or ""
                env_name = getattr(env_spec, "name", None) or env_id or "unknown"
                is_sparse_reward = "antmaze" in env_name or env_name in ADROIT_ENVS
                mc_returns = calc_return_to_go(
                    env_name,
                    rewards,
                    dones,
                    self.gamma,
                    self.reward_scale,
                    self.reward_bias,
                    is_sparse_reward=is_sparse_reward,
                )

            if replay_buffer is not None:
                for i in range(len(rewards)):
                    sample = (
                        observations[i],
                        actions[i],
                        rewards[i],
                        next_observations[i],
                        dones[i],
                    )
                    if self.use_mc:
                        replay_buffer.add_sample(*sample, mc_returns[i])
                    else:
                        replay_buffer.add_sample(*sample)

            traj = dict(
                observations=np.asarray(observations, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32),
                rewards=np.asarray(rewards, dtype=np.float32),
                next_observations=np.asarray(next_observations, dtype=np.float32),
                dones=np.asarray(dones, dtype=np.float32),
            )
            if self.use_mc:
                traj["mc_returns"] = np.asarray(mc_returns, dtype=np.float32)
            if self.use_goal:
                traj["goal_achieved"] = np.asarray(
                    goal_achieved_list, dtype=np.float32
                )
            trajs.append(traj)

        return trajs

    @property
    def env(self):
        return self._env
