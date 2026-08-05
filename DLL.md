# Development and Learning Log

This is the setup and work log for our two-person Deep Learning Lab project at
the University of Freiburg. We worked on the project for around two months. The
goal of this file is to leave enough information for another student to set up
the code and understand what we changed without turning it into a large software
manual.

## 1. Conda setup we used

We ran the project on Ubuntu through WSL with Miniforge. Python 3.10 was used.
First we made Conda available in the terminal:

```bash
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Conda was not found. Install Miniforge or Miniconda first."
    exit 1
fi

conda --version
```

After `conda --version` printed a version, we cloned the project and created a
clean environment:

```bash
git clone https://github.com/SumitSkp/cal-ql-minari.git
cd cal-ql-minari

conda create --name calql-minari \
    python=3.10 numpy=1.26.4 cython=0.29.37 pip -y
conda activate calql-minari

export PYTHONNOUSERSITE=1
python -m pip install --no-user -r requirements.txt
```

If an environment with this name already exists, activate it and install the
requirements again:

```bash
conda activate calql-minari
export PYTHONNOUSERSITE=1
python -m pip install --no-user -r requirements.txt
```

We needed `PYTHONNOUSERSITE=1` and `--no-user` because our WSL installation had
a Python package contamination problem. Some packages from `~/.local` were
being loaded instead of the versions in the Conda environment. This caused
version conflicts even after we made a new environment. The two options above
fixed the problem on our system. We export `PYTHONNOUSERSITE=1` again in every
new terminal before running the project.

We used these commands to confirm that the correct environment was active:

```bash
python -c "import os, sys; print(os.environ.get('CONDA_DEFAULT_ENV')); print(sys.executable)"
python -c "import jax, flax, minari, mujoco; print('imports successful:', jax.default_backend())"
python -m pip check
```

The first command should print `calql-minari`, and the Python path should point
inside that environment. The second command normally prints
`imports successful: cpu` before the GPU version of JAX is installed.

## 2. Small checks we ran

We wrote one small test while adding the fall penalty and cost critic:

```bash
python -m compileall -q JaxCQL scripts/smoke_test_safety.py
JAX_PLATFORM_NAME=cpu python scripts/smoke_test_safety.py
```

It checks that normal rewards are unchanged when the penalty is off, true
terminations and timeouts are separated, the normal Cal-QL/CQL update still
runs, and the `Q_fall(s,a)` critic can make one finite update. On our CPU setup
the final line was:

```text
all safety smoke tests passed
```

The first JAX compilation can take around two minutes. This is only a quick
code check, not a replacement for a complete training run.

## 3. Minari datasets

Minari saves datasets in `~/.minari/datasets/` by default. We downloaded Hopper
with:

```bash
python -c "import minari; minari.download_dataset('mujoco/hopper/medium-v0'); print(minari.list_local_datasets())"
```

The Hugging Face download package is already in `requirements.txt`. If Hopper
is already downloaded, Minari prints `Skipping Download`; this is normal. A
different storage folder can be selected by setting `MINARI_DATASETS_PATH`
before running the command.

The same form works for our other environments:

```text
mujoco/halfcheetah/medium-v0
mujoco/walker2d/medium-v0
mujoco/humanoid/medium-v0
```

## 4. GPU setup on our machine

Our GPU was an NVIDIA RTX 4060. We first checked that WSL could see it:

```bash
nvidia-smi
```

We then installed the CUDA 12 JAX build in the same `calql-minari` environment:

```bash
conda activate calql-minari
export PYTHONNOUSERSITE=1
python -m pip install --no-user --upgrade "jax[cuda12]==0.6.2"
unset JAX_PLATFORM_NAME
```

We checked the GPU with a real calculation, not only `jax.devices()`:

```bash
python -c "import jax, jax.numpy as jnp; x=(jnp.ones((8,8))@jnp.ones((8,8))).block_until_ready(); print(jax.default_backend(), jax.devices(), float(x.sum()))"
```

On our RTX 4060, the output was:

```text
gpu [CudaDevice(id=0)] 512.0
```

For another NVIDIA GPU, the main point is that the installed JAX CUDA version
and the NVIDIA driver must be compatible. `nvidia-smi` should see the GPU, and
the command above should print `gpu` instead of `cpu`. If it still prints CPU,
we would first check the driver shown by `nvidia-smi`, then choose the matching
CUDA installation from the JAX installation instructions. The learning code
itself does not need to be changed for a different NVIDIA GPU.

## 5. Small end-to-end run

This command checks the real Hopper dataset, one offline update, one online
rollout, evaluation, and shutdown. It uses W&B offline:

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

Our CPU and RTX 4060 tests both ended with:

```text
Finished Training
```

To use the GPU, remove `JAX_PLATFORM_NAME=cpu` after completing the GPU setup.

## 6. What we changed for Minari

The Minari conversion is mainly in `JaxCQL/minari_compat.py`. We added the
course datasets, scaled actions to the policy range `[-1, 1]`, reconstructed
the transition fields, calculated the Monte Carlo returns used by Cal-QL, and
recovered the Gymnasium environment for online fine-tuning.

We also separated a true termination from a time-limit truncation. Both finish
an episode, but a timeout should not be treated as a fall. Old Gym and D4RL
imports are loaded only when an old D4RL environment is requested, so the
Minari experiments do not require `mujoco-py`.

Our full experiments used 1,000 offline pretraining epochs with 1,000 updates
per epoch, followed by 250,000 online environment steps. W&B was offline for
small tests. For real logged runs we used `wandb login` once on the machine and
passed `--logging.online=True`. The API key should never be added to the code.

## 7. Main Humanoid experiment commands

The positive ablation combining the larger critic and lower replay mixing was
started with:

```bash
python -m JaxCQL.conservative_sac_main \
  --env=mujoco/humanoid/medium-v0 \
  --qf_arch=512-512 \
  --mixing_ratio=0.25 \
  --logging.online=True \
  --logging.project=CalQL-Minari \
  --logging.experiment_id=humanoid_qf512_mix025
```

The fixed fall-penalty run used:

```bash
python -m JaxCQL.conservative_sac_main \
  --env=mujoco/humanoid/medium-v0 \
  --use_fall_penalty=True \
  --fall_penalty=100 \
  --logging.online=True \
  --logging.project=CalQL-Minari \
  --logging.experiment_id=humanoid_fall_penalty_100
```

The learned fall-cost critic used:

```bash
python -m JaxCQL.conservative_sac_main \
  --env=mujoco/humanoid/medium-v0 \
  --use_cost_critic=True \
  --cost_lambda=10 \
  --logging.online=True \
  --logging.project=CalQL-Minari \
  --logging.experiment_id=humanoid_qfall_lambda10
```

The safety options are off by default, so the usual Cal-QL/CQL path does not
create the extra cost networks.

## 8. Safety ideas and what we observed

For the fixed penalty, the reward used for training becomes:

```text
training reward = raw reward - fall_penalty * terminal cost
```

The terminal cost is one only for a true termination. We logged the raw and
shaped returns separately.

For the learned version, two extra Q-networks learn the terminal-cost signal.
The actor loss becomes:

```text
alpha * log_pi - Q_reward(s,a) + cost_lambda * Q_fall(s,a)
```

These were our own exploratory additions, not part of the original Cal-QL
paper. Neither gave a clean improvement in the Humanoid runs we completed. The
fixed penalty learned more slowly. The learned critic improved for some time
but later became unstable. We think the single penalty and lambda values did
not balance the reward and cost scales well enough.

A proper follow-up would need several seeds and sweeps over the penalty,
`cost_lambda`, cost discount, normalization, and cost-critic learning rate.
Safe and constrained RL is a research topic by itself, so we treat these as
first negative ablations rather than a conclusion that the ideas cannot work.
We also used a Humanoid termination as a simple fall signal, which is useful
for this experiment but is not a complete definition of safety.

## 9. Version used for the project

- Repository: `https://github.com/SumitSkp/cal-ql-minari.git`
- Original final merge started from commit `ae96360`.
- The code is based on the public JAX/Flax Cal-QL implementation by Nakamoto et
  al. The original paper and repository are linked from the README.
