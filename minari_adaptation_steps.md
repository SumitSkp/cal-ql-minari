# Minari Adaptation Steps

## 1. Add Flags

In `JaxCQL/conservative_sac_main.py`, add:

```python
dataset_source="d4rl",
minari_dataset_id="",
minari_download=True,
minari_sparse_reward=False,
```

## 2. Add Imports

In `JaxCQL/conservative_sac_main.py`, add:

```python
import minari
```

Add to the replay buffer imports:

```python
get_minari_dataset_with_mc_calculation,
```

## 3. Initialize Minari Dataset Variable

Before the dataset-source branch in `main`, add:

```python
minari_dataset = None
```

## 4. Add Minari Dataset Branch

In `main`, add:

```python
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

## 5. Wrap Existing D4RL Logic

Put the old D4RL dataset-loading code under:

```python
elif FLAGS.dataset_source == "d4rl":
```

Inside that branch, define:

```python
eval_env = gym.make(FLAGS.env).unwrapped
train_env = gym.make(FLAGS.env).unwrapped
```

After the branch, add:

```python
else:
    raise ValueError(f"Unknown dataset_source: {FLAGS.dataset_source}")
```

## 6. Use `eval_env` And `train_env`

Replace direct `gym.make(...)` sampler construction with:

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

## 7. Update Minari Loader Signature

In `JaxCQL/replay_buffer.py`, change the Minari loader to:

```python
def get_minari_dataset_with_mc_calculation(
        minari_dataset,
        reward_scale,
        reward_bias,
        clip_action,
        gamma,
        is_sparse_reward=False
        ):
```

Inside it, add:

```python
dataset_id = getattr(minari_dataset, "dataset_id", "")
episodes = []
```

Loop over:

```python
for episode in minari_dataset:
```

Remove any internal call to:

```python
minari.load_dataset(...)
```

## 8. Convert Minari Episodes

Inside the Minari loader:

```python
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
```

## 9. Add Minari MC Returns

Inside the Minari loader:

```python
mc_returns = calc_return_to_go(
    dataset_id,
    rewards,
    dones,
    gamma,
    reward_scale,
    reward_bias,
    is_sparse_reward=is_sparse_reward,
)
```

Append:

```python
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
```

Return:

```python
return concatenate_batches(episodes)
```

## 10. Update Sampler Reset Handling

In `JaxCQL/sampler.py`, replace:

```python
observation = self.env.reset()
```

with:

```python
reset_result = self.env.reset()

if isinstance(reset_result, tuple):
    observation = reset_result[0]
else:
    observation = reset_result
```

## 11. Update Sampler Step Handling

Replace:

```python
next_observation, reward, done, env_infos = self.env.step(action)
```

with:

```python
step_result = self.env.step(action)

if len(step_result) == 5:
    next_observation, reward, terminated, truncated, env_infos = step_result
    done = terminated or truncated
else:
    next_observation, reward, done, env_infos = step_result
```

## 12. Make Env Name Robust In Sampler

Inside `if self.use_mc:`, add:

```python
env_spec = getattr(self.env, "spec", None)
env_id = getattr(env_spec, "id", None) or ""
env_name = getattr(env_spec, "name", None) or env_id
```

Use `env_name` in all `calc_return_to_go(...)` calls.

## 13. Add Dense Fallback In Sampler

Replace final `raise NotImplementedError` with:

```python
mc_returns = calc_return_to_go(
    env_name,
    rewards,
    dones,
    self.gamma,
    self.reward_scale,
    self.reward_bias,
    is_sparse_reward=False,
)
```

## 14. Update Normalized Score Logic

In `JaxCQL/conservative_sac_main.py`, replace unconditional D4RL normalized score with:

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

## 15. Syntax Check

Run:

```bash
python -m py_compile JaxCQL/conservative_sac_main.py JaxCQL/replay_buffer.py JaxCQL/sampler.py
```

## 16. List Hopper Minari Datasets

Run:

```bash
python -c "import minari; print([d for d in minari.list_remote_datasets() if 'hopper' in d.lower()])"
```

Use:

```text
mujoco/hopper/medium-v0
```

## 17. Evaluation Smoke Test

Run:

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

## 18. Training Smoke Test

Run:

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

## 19. Cleanup

Remove:

```python
#
```

markers after imports/flags.

Remove old commented-out normalized-score block.

