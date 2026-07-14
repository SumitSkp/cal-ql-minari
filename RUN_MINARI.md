# Cal-QL + Minari quick start

Requires Ubuntu 22.04/WSL, Miniconda, an NVIDIA driver, and MuJoCo 2.1 at
`~/.mujoco/mujoco210`.

## 1. Clone and install

```bash
git clone -b codex/shared-minari-calql https://github.com/SumitSkp/cal-ql-minari.git
cd cal-ql-minari

source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate Cal-QL
python -m pip install -r requirements-lock.txt
```

If a broken `Cal-QL` environment already exists, remove it first with
`conda env remove -n Cal-QL -y`.

## 2. Prepare and verify

```bash
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
wandb login
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

Continue only if the last command prints `gpu`.

## 3. Tiny Hopper smoke test

```bash
python -m JaxCQL.conservative_sac_main \
  --env=mujoco/hopper/medium-v0 \
  --n_train_step_per_epoch_offline=1 \
  --n_pretrain_epochs=1 \
  --max_online_env_steps=1 \
  --eval_n_trajs=1 \
  --logging.online=True \
  --logging.project=CalQL-Minari \
  --logging.experiment_id=hopper_smoke
```

For the large seed-0 run, change the three budgets to `1000`, `1000`, and
`250000`, and use experiment ID `NR2_Hopper_Medium_CalQL_seed0`.
