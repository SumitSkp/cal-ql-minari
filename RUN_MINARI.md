# Running Cal-QL with Minari

Use the installation steps in [DLL.md](DLL.md). The commands below assume the
`calql-minari` environment is active and `PYTHONNOUSERSITE=1` is exported.

Minari stores downloaded datasets in `~/.minari/datasets/` by default. To use a
different location, set `MINARI_DATASETS_PATH` before downloading or training.

## Small offline CPU run

This checks the actual Hopper dataset, evaluation, offline update, online
rollout, and shutdown path with a very small budget. It logs to W&B offline.

```bash
JAX_PLATFORM_NAME=cpu python -m JaxCQL.conservative_sac_main \
  --env=mujoco/hopper/medium-v0 \
  --n_train_step_per_epoch_offline=1 \
  --n_pretrain_epochs=1 \
  --max_online_env_steps=1 \
  --n_online_traj_per_epoch=1 \
  --online_utd_ratio=0 \
  --mixing_ratio=0.5 \
  --eval_n_trajs=1 \
  --offline_eval_every_n_epoch=1 \
  --online_eval_every_n_env_steps=1 \
  --logging.online=False \
  --logging.project=CalQL-Minari \
  --logging.experiment_id=hopper_cpu_smoke
```

For GPU, install `requirements-gpu.txt`, remove `JAX_PLATFORM_NAME=cpu`, and
use the same command. Full experiments used 1,000 pretraining epochs with 1,000
updates per epoch and 250,000 online environment steps.

## Humanoid ablations

Larger critic and replay mixing:

```bash
python -m JaxCQL.conservative_sac_main \
  --env=mujoco/humanoid/medium-v0 \
  --qf_arch=512-512 \
  --mixing_ratio=0.25 \
  --logging.online=True \
  --logging.project=CalQL-Minari \
  --logging.experiment_id=humanoid_qf512_mix025
```

Fixed fall penalty:

```bash
python -m JaxCQL.conservative_sac_main \
  --env=mujoco/humanoid/medium-v0 \
  --use_fall_penalty=True \
  --fall_penalty=100 \
  --logging.online=True \
  --logging.project=CalQL-Minari \
  --logging.experiment_id=humanoid_fall_penalty_100
```

Learned `Q_fall(s,a)` cost critic:

```bash
python -m JaxCQL.conservative_sac_main \
  --env=mujoco/humanoid/medium-v0 \
  --use_cost_critic=True \
  --cost_lambda=10 \
  --logging.online=True \
  --logging.project=CalQL-Minari \
  --logging.experiment_id=humanoid_qfall_lambda10
```

The safety variants are off by default. They were exploratory negative
ablations and should not be presented as solved safe-RL methods.

## Optional original D4RL tasks

Minari does not require `mujoco-py` or MuJoCo 2.1. To run the original D4RL
tasks, install `requirements-d4rl.txt` and follow the legacy MuJoCo setup noted
in [DLL.md](DLL.md).
