# Development and Learning Log

This file records the code and setup used for our Deep Learning Lab project. We
wrote it so that another student can understand what we changed and run a small
check without first reading the complete Cal-QL codebase.

## 1. Project version

- Repository: `https://github.com/SumitSkp/cal-ql-minari.git`
- Submission tag: `final-submission-2026-08-05`
- Starting point for this merge: GitHub `main` at `ae96360`
- Main task: retain the newer Minari adapter and add our two experimental
  safety/stability ablations.

The implementation still follows the public JAX/Flax Cal-QL code. Our changes
are mainly in dataset handling, termination/truncation handling, the optional
cost critic, logging, and reproducibility checks.

## 2. Fresh CPU setup

We used Ubuntu/WSL and Miniforge. A fresh environment can be created with:

```bash
git clone https://github.com/SumitSkp/cal-ql-minari.git
cd cal-ql-minari

source ~/miniforge3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate calql-minari

export PYTHONNOUSERSITE=1
python -m pip install --no-user -r requirements-lock.txt
```

`PYTHONNOUSERSITE=1` and `--no-user` are intentional. They stop packages from
`~/.local` from replacing the versions inside the conda environment.

Run the submission check from the repository root:

```bash
bash scripts/check_submission.sh
```

It checks the environment, compiles the changed Python files, and runs four
small checks:

1. replay-buffer costs and truncations are stored correctly;
2. fall terminations are separated from time-limit truncations;
3. a standard Cal-QL/CQL update still works with the safety code disabled;
4. a `Q_fall(s,a)` cost-critic update runs and returns finite metrics.

The first JAX compilation took about two minutes on our machine. EGL, Gym, and
dependency deprecation warnings can appear, but the final output should contain
`all safety smoke tests passed` and `submission checks passed`.

## 3. Minari datasets and GPU setup

Minari downloads datasets to `~/.minari/datasets/` by default. The location can
be changed with `MINARI_DATASETS_PATH`. Hopper can be downloaded and checked
before training with:

```bash
python -c "import minari; minari.download_dataset('mujoco/hopper/medium-v0'); print(minari.list_local_datasets())"
```

For GPU experiments, install the CUDA-enabled JAX extra:

```bash
python -m pip install --no-user -r requirements-gpu.txt
unset JAX_PLATFORM_NAME
python -c "import jax, jax.numpy as jnp; x=(jnp.ones((8,8))@jnp.ones((8,8))).block_until_ready(); print(jax.default_backend(), jax.devices(), float(x.sum()))"
```

We check a compiled operation because printing `jax.devices()` alone does not
prove that CUDA compilation works. A compatible NVIDIA driver is still needed.
The small CPU and GPU Hopper commands are in [RUN_MINARI.md](RUN_MINARI.md).

W&B is offline by default. For experiment logging, run `wandb login` and pass
`--logging.online=True`. No API key or local `wandb_config.py` should be
committed.

## 4. Minari adaptation

`JaxCQL/minari_compat.py` keeps the Minari-specific work outside the main
training loop. It:

- loads `mujoco/.../medium-v0` datasets;
- converts native environment actions to the policy range `[-1, 1]`;
- reconstructs observations, next observations, rewards, true terminations,
  time-limit truncations, and Monte Carlo returns;
- recovers the Gymnasium environment used for online interaction.

Legacy Gym and D4RL imports are delayed until a D4RL task is selected. This
means the course-required Minari path does not depend on `mujoco-py`.

## 5. Safety/stability ablations

### Fixed fall penalty

The flags are:

```bash
--use_fall_penalty=True --fall_penalty=100
```

The reward used for learning becomes:

```text
reward_for_training = raw_reward - fall_penalty * terminal_cost
```

`terminal_cost` is one only for a true environment termination. A time-limit
truncation ends the trajectory but is not counted as a fall. Raw and shaped
returns are both logged so they are not confused in analysis.

### Learned Q_fall(s,a) critic

The flags are:

```bash
--use_cost_critic=True --cost_lambda=10
```

Two extra cost Q-functions are trained from the terminal-cost signal. The actor
loss is changed to:

```text
alpha * log_pi - Q_reward(s,a) + cost_lambda * Q_fall(s,a)
```

We use the larger of the two cost estimates for the actor penalty. This is a
simple conservative choice for the safety cost. The feature is disabled by
default, so the standard Cal-QL/CQL update does not create or train these
networks.

Extra W&B metrics include raw return, fall rate, average fall cost, timeout
rate, cost-critic losses, and estimated cost values.

## 6. Results and limitations

The fixed penalty and learned cost critic were our own exploratory extensions;
they are not part of the original Cal-QL implementation. Neither produced a
clean improvement in our available Humanoid runs. We think the main difficulty
was balancing reward and cost scales with only one selected penalty or lambda.

A stronger study would sweep the fall penalty, `cost_lambda`, cost discount,
reward/cost normalization, and the cost-critic learning rate over several
seeds. Safe and constrained reinforcement learning is a research topic on its
own, so these negative ablations should be read as first attempts, not as a
general conclusion about safety critics.

We use true termination as a simple fall signal. This is useful for Humanoid
but is not a complete definition of safety. The smoke tests verify the code
paths, not the final performance of a full-budget run.

## 7. Optional legacy D4RL setup

The course experiments use Minari. The original Cal-QL D4RL path remains
available separately:

```bash
python -m pip install --no-user -r requirements-d4rl.txt
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
```

This optional path needs MuJoCo 2.1 and `mujoco-py`. It is kept separate so
that these older packages cannot interfere with the Minari setup.
