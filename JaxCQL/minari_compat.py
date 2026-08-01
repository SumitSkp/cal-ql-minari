"""Small Minari adapter that keeps Minari details out of the Cal-QL core."""

import numpy as np

from .replay_buffer import calc_return_to_go, concatenate_batches


def is_minari_env(name):
    return bool(name) and name.startswith("mujoco/")


def make_minari_env(minari_dataset):
    from gymnasium.wrappers import RescaleAction

    env = minari_dataset.recover_environment()
    return RescaleAction(env, min_action=-1.0, max_action=1.0)


def load_minari_dataset(
    dataset_id,
    reward_scale,
    reward_bias,
    clip_action,
    gamma,
    download=True,
    is_sparse_reward=False,
):
    import minari

    minari_dataset = minari.load_dataset(dataset_id, download=download)

    reference_env = minari_dataset.recover_environment()
    action_low = np.asarray(reference_env.action_space.low, dtype=np.float32)
    action_high = np.asarray(reference_env.action_space.high, dtype=np.float32)
    reference_env.close()

    action_range = action_high - action_low
    if np.any(action_range <= 0) or not np.all(np.isfinite(action_range)):
        raise ValueError(
            f"Unsupported action space for Minari dataset {dataset_id}: "
            f"low={action_low}, high={action_high}"
        )

    iterate_episodes = getattr(minari_dataset, "iterate_episodes", None)
    episode_iterator = (
        iterate_episodes() if callable(iterate_episodes) else iter(minari_dataset)
    )

    episodes = []
    for episode in episode_iterator:
        observations = np.asarray(episode.observations, dtype=np.float32)
        actions = np.asarray(episode.actions, dtype=np.float32)
        # Convert native environment actions to the policy/CQL range [-1, 1].
        actions = 2.0 * (actions - action_low) / action_range - 1.0
        rewards = (
            np.asarray(episode.rewards, dtype=np.float32) * reward_scale
            + reward_bias
        )
        # Time-limit truncation ends collection but is not a true MDP terminal.
        dones = np.asarray(episode.terminations, dtype=np.float32)

        if clip_action is not None:
            actions = np.clip(actions, -clip_action, clip_action)

        transition_count = actions.shape[0]
        if not (
            observations.shape[0] == transition_count + 1
            and rewards.shape[0] == transition_count
            and dones.shape[0] == transition_count
        ):
            raise ValueError(f"Malformed episode in Minari dataset {dataset_id}")

        episodes.append(
            dict(
                observations=observations[:-1],
                actions=actions,
                next_observations=observations[1:],
                rewards=rewards,
                dones=dones,
                mc_returns=calc_return_to_go(
                    dataset_id,
                    rewards,
                    dones,
                    gamma,
                    reward_scale,
                    reward_bias,
                    is_sparse_reward=is_sparse_reward,
                ),
            )
        )

    if not episodes:
        raise ValueError(f"Minari dataset {dataset_id} contains no episodes")

    return concatenate_batches(episodes), minari_dataset
