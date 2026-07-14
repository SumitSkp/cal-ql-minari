# Run Cal-QL with Minari

## 1. Create the environment

If an old `Cal-QL` environment is broken, remove it once with
`conda env remove -n Cal-QL -y`.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate Cal-QL
python -m pip install numpy==1.26.4 Cython==0.29.37
python -m pip install -r requirements-lock.txt
```

## 2. Prepare the shell

```bash
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
wandb login
```

Confirm that `python -c "import jax; print(jax.default_backend(), jax.devices())"`
prints `gpu` before a long run.

## 3. Run a tiny Hopper check

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

The older explicit flags also work:
`--dataset_source=minari --minari_dataset_id=mujoco/hopper/medium-v0`.
