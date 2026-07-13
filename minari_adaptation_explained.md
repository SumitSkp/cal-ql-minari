# Minari Adaptation Notes

This file records the sequence of changes used to adapt the Cal-QL codebase from a D4RL-only path to a Minari-compatible path. The goal is to make the same changes reproducible on another machine.

## 1. Add Dataset Source Flags

In `JaxCQL/conservative_sac_main.py`, add flags to choose between the old D4RL path and the new Minari path:

```python
dataset_source="d4rl",
minari_dataset_id="",
minari_download=True,
minari_sparse_reward=False,
```

Short explanation:

- `dataset_source` decides whether to use D4RL or Minari.
- `minari_dataset_id` stores the Minari dataset id, for example `mujoco/hopper/medium-v0`.
- `minari_download` lets Minari download the dataset if it is not local.
- `minari_sparse_reward` tells the return-to-go code whether sparse reward handling is needed.

## 2. Import The Minari Loader

In `JaxCQL/conservative_sac_main.py`, import the Minari dataset conversion function:

```python
from .replay_buffer import (
    subsample_batch,
    concatenate_batches,
    get_d4rl_dataset_with_mc_calculation,
    get_hand_dataset_with_mc_calculation,
    get_minari_dataset_with_mc_calculation,
)
```

Also import Minari:

```python
import minari
```

Short explanation:

The main script needs Minari for `minari.load_dataset(...)`, and it needs the replay-buffer helper to convert Minari episodes into the batch dictionary expected by Cal-QL.

## 3. Branch Dataset Loading By Source

In `main`, branch on `FLAGS.dataset_source`.

The Minari branch should:

```python
minari_dataset = None

if FLAGS.dataset_source == "minari":
    if not FLAGS.minari_dataset_id:
        raise ValueError("Please set --minari_dataset_id when using --dataset_source=minari.")

    minari_dataset = minari.load_dataset(
        FLAGS.minari_dataset_id,
        download=FLAGS.minari_download,
    )

    dataset = get_minari_dataset_with_mc_calculation(
        minari_dataset=minari_dataset,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        clip_action=FLAGS.clip_action,
        gamma=FLAGS.cql.discount,
        is_sparse_reward=FLAGS.minari_sparse_reward,
    )

    eval_env = minari_dataset.recover_environment(eval_env=True)
    train_env = minari_dataset.recover_environment()
    use_goal = False
```

The D4RL branch should still use the old dataset logic, but must also define `eval_env` and `train_env`:

```python
elif FLAGS.dataset_source == "d4rl":
    if FLAGS.env in ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0"]:
        import mj_envs
        dataset = get_hand_dataset_with_mc_calculation(...)
        use_goal = True
    elif FLAGS.env == "hopper-medium-v2":
        dataset = get_d4rl_dataset_with_mc_calculation(...)
        use_goal = False
    else:
        dataset = get_d4rl_dataset_with_mc_calculation(...)
        use_goal = False

    eval_env = gym.make(FLAGS.env).unwrapped
    train_env = gym.make(FLAGS.env).unwrapped
else:
    raise ValueError(f"Unknown dataset_source: {FLAGS.dataset_source}")
```

Short explanation:

After this branch, the following variables must exist no matter which successful path was used:

```python
dataset
use_goal
eval_env
train_env
```

## 4. Use Shared Env Variables In The Samplers

Replace direct `gym.make(...)` calls in the sampler construction with the branch-created env variables:

```python
eval_sampler = TrajSampler(
    eval_env,
    use_goal,
    gamma=FLAGS.cql.discount,
)

train_sampler = TrajSampler(
    train_env,
    use_goal,
    use_mc=True,
    gamma=FLAGS.cql.discount,
    reward_scale=FLAGS.reward_scale,
    reward_bias=FLAGS.reward_bias,
)
```

Short explanation:

For D4RL, `eval_env` and `train_env` come from `gym.make(...)`.

For Minari, they come from `minari_dataset.recover_environment(...)`.

The rest of the training code can then use the same sampler construction for both paths.

## 5. Convert Minari Episodes Without Reloading The Dataset

In `JaxCQL/replay_buffer.py`, change the Minari helper so it accepts an already loaded Minari dataset object:

```python
def get_minari_dataset_with_mc_calculation(
        minari_dataset,
        reward_scale,
        reward_bias,
        clip_action,
        gamma,
        is_sparse_reward=False
        ):

    dataset_id = getattr(minari_dataset, "dataset_id", "")
    episodes = []

    for episode in minari_dataset:
        observations_raw = np.asarray(episode.observations, dtype=np.float32)
        observations = observations_raw[:-1]
        next_observations = observations_raw[1:]
        actions = np.asarray(episode.actions, dtype=np.float32)
        rewards = np.asarray(episode.rewards, dtype=np.float32)
        terminations = np.asarray(episode.terminations, dtype=bool)

        rewards = rewards * reward_scale + reward_bias
        dones = terminations.astype(np.float32)

        if clip_action is not None:
            actions = np.clip(actions, -clip_action, clip_action)

        assert observations.shape[0] == actions.shape[0]
        assert observations.shape[0] == rewards.shape[0]
        assert observations.shape[0] == next_observations.shape[0]
        assert observations.shape[0] == dones.shape[0]

        mc_returns = calc_return_to_go(
            dataset_id,
            rewards,
            dones,
            gamma,
            reward_scale,
            reward_bias,
            is_sparse_reward=is_sparse_reward,
        )

        episodes.append(
            dict(
                observations=observations,
                actions=actions,
                next_observations=next_observations,
                rewards=rewards,
                dones=dones,
                mc_returns=mc_returns,
            )
        )

    if not episodes:
        raise ValueError(f"Minari dataset {dataset_id} did not contain any episodes.")

    return concatenate_batches(episodes)
```

Short explanation:

Minari observations contain one more observation than actions/rewards. Therefore:

```python
observations = observations_raw[:-1]
next_observations = observations_raw[1:]
```

We use:

```python
dones = terminations.astype(np.float32)
```

This avoids treating time-limit truncations as true task terminals.

## 6. Handle Gym And Gymnasium Reset/Step APIs

In `JaxCQL/sampler.py`, update reset handling:

```python
reset_result = self.env.reset()

if isinstance(reset_result, tuple):
    observation = reset_result[0]
else:
    observation = reset_result
```

Update step handling:

```python
step_result = self.env.step(action)

if len(step_result) == 5:
    next_observation, reward, terminated, truncated, env_infos = step_result
    done = terminated or truncated
else:
    next_observation, reward, done, env_infos = step_result
```

Short explanation:

Old Gym returns:

```python
obs = env.reset()
obs, reward, done, info = env.step(action)
```

Gymnasium returns:

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

The sampler should adapt to the environment API, not to `dataset_source`.

## 7. Make Online MC Return Env Names Robust

In `JaxCQL/sampler.py`, avoid directly relying on `self.env.spec.name`:

```python
env_spec = getattr(self.env, "spec", None)
env_id = getattr(env_spec, "id", None) or ""
env_name = getattr(env_spec, "name", None) or env_id
```

Then use `env_name` in `calc_return_to_go(...)`.

Known sparse/dense cases:

```python
if env_id == "hopper-medium-v2" or env_name == "hopper-medium-v2":
    mc_returns = calc_return_to_go(..., is_sparse_reward=False)
elif "antmaze" in env_name or "antmaze" in env_id:
    mc_returns = calc_return_to_go(..., is_sparse_reward=True)
elif env_name in ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0", "pen-binary", "door-binary", "relocate-binary"]:
    mc_returns = calc_return_to_go(..., is_sparse_reward=True)
else:
    mc_returns = calc_return_to_go(..., is_sparse_reward=False)
```

Short explanation:

Minari-recovered environments may have different `spec.id` / `spec.name` values. Unknown dense-reward environments should use normal discounted return-to-go instead of immediately raising `NotImplementedError`.

## 8. Make Normalized Evaluation Score Optional

In `JaxCQL/conservative_sac_main.py`, keep raw return always:

```python
metrics["evaluation/average_return"] = np.mean(
    [np.sum(t["rewards"]) for t in trajs]
)
```

Then make normalized score optional:

```python
returns = np.array([np.sum(t["rewards"]) for t in trajs])

if minari_dataset is not None:
    try:
        metrics["evaluation/average_normalized_return"] = np.mean(
            minari.get_normalized_score(minari_dataset, returns)
        )
    except (AttributeError, ValueError, KeyError):
        pass
elif hasattr(eval_sampler.env, "get_normalized_score"):
    metrics["evaluation/average_normalized_return"] = np.mean(
        [
            eval_sampler.env.get_normalized_score(r)
            for r in returns
        ]
    )
```

Short explanation:

D4RL envs often provide `env.get_normalized_score(...)`.

Minari may provide `minari.get_normalized_score(...)`.

Some datasets may not provide normalized scores, so missing normalized score should not crash training.

## 9. Syntax Check

Run this after the code changes:

```bash
python -m py_compile JaxCQL/conservative_sac_main.py JaxCQL/replay_buffer.py JaxCQL/sampler.py
```

Result from our run:

```text
passed
```

## 10. Find A Valid Minari Dataset ID

The first guessed id failed:

```text
D4RL/hopper/medium-v2
```

The error was:

```text
Couldn't find any version for dataset D4RL/hopper/medium in the remote Farama server.
```

We listed remote Hopper datasets:

```bash
python -c "import minari; print([d for d in minari.list_remote_datasets() if 'hopper' in d.lower()])"
```

Output:

```python
['mujoco/hopper/expert-v0', 'mujoco/hopper/simple-v0', 'mujoco/hopper/medium-v0', 'atari/choppercommand/expert-v0']
```

So we used:

```text
mujoco/hopper/medium-v0
```

## 11. Evaluation Smoke Test

Command:

```bash
python -m JaxCQL.conservative_sac_main \
  --dataset_source=minari \
  --minari_dataset_id=mujoco/hopper/medium-v0 \
  --n_pretrain_epochs=1 \
  --n_train_step_per_epoch_offline=1 \
  --offline_eval_every_n_epoch=1 \
  --eval_n_trajs=1 \
  --max_online_env_steps=0
```

Result:

```text
Starting Evaluation for Epoch 0
evaluation/average_return       123.257
evaluation/average_traj_length  139
Finished Training
```

Short explanation:

This validated dataset loading, environment recovery, and evaluation rollout, but not training.

## 12. Training Smoke Test

Command:

```bash
python -m JaxCQL.conservative_sac_main \
  --dataset_source=minari \
  --minari_dataset_id=mujoco/hopper/medium-v0 \
  --n_pretrain_epochs=1 \
  --n_train_step_per_epoch_offline=1 \
  --offline_eval_every_n_epoch=1 \
  --eval_n_trajs=1 \
  --max_online_env_steps=1
```

Important output:

```text
Starting Evaluation for Epoch 0
grad_steps 0

jit compiling train function: will take a while

Starting Evaluation for Epoch 1
grad_steps 1

collecting online trajs: 1

Starting Evaluation for Epoch 2
grad_steps 18
env_steps 17
exploration/average_return 14.0922
exploration/average_traj_length 17

Finished Training
```

Short explanation:

This validated the full path:

- Minari dataset loading
- Minari offline conversion
- Minari environment recovery
- Gymnasium-compatible reset/step
- evaluation rollout
- offline training step
- online rollout
- mixed offline/online training

## 13. Current Status

The Minari adaptation is functionally validated for:

```text
mujoco/hopper/medium-v0
```

Remaining cleanup items:

- Remove stray `#` markers after imports/flags.
- Remove old commented-out normalized-score block.
- Optionally remove unused `truncations` in the Minari loader, or keep it if planning to handle timeout transitions later.
- Optionally move Minari dataset loading/conversion into a cleaner `dataset_loaders.py` later.

