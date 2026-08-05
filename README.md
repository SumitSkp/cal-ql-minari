# Cal-QL on Minari MuJoCo

This repository contains our Deep Learning Lab course project at the University
of Freiburg. We adapted the public JAX implementation of Cal-QL to the Minari
MuJoCo datasets and ran offline-to-online experiments on Hopper, Half-Cheetah,
Walker2d, and Humanoid.

The main goal was to study the speed of offline-to-online transfer, not only the
last evaluation return. We also tested replay mixing, a larger critic, and two
exploratory safety/stability ideas: a fixed fall penalty and a learned
`Q_fall(s,a)` cost critic.

The exact implementation notes, commands, and limitations are in
[DLL.md](DLL.md). Short experiment commands are collected in
[RUN_MINARI.md](RUN_MINARI.md). The submission version is tagged
`final-submission-2026-08-05`.

## Quick CPU check

The smoke test does not need a GPU, a W&B account, or a downloaded Minari
dataset.

```bash
git clone https://github.com/SumitSkp/cal-ql-minari.git
cd cal-ql-minari

source ~/miniforge3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate calql-minari

export PYTHONNOUSERSITE=1
python -m pip install --no-user -r requirements-lock.txt
bash scripts/check_submission.sh
```

The first JAX compilation can take around two minutes. Warnings from the old
Gym package are only expected if the optional D4RL dependencies were installed.

## GPU experiments

For the full Minari runs, install the CUDA-enabled JAX extra and verify a real
compiled operation:

```bash
python -m pip install --no-user -r requirements-gpu.txt
unset JAX_PLATFORM_NAME
python -c "import jax, jax.numpy as jnp; x=(jnp.ones((8,8))@jnp.ones((8,8))).block_until_ready(); print(jax.default_backend(), jax.devices(), float(x.sum()))"
```

The host still needs a compatible NVIDIA driver. W&B is optional for local
checks. Run `wandb login` and set `--logging.online=True` only when online
logging is wanted; API keys must not be stored in this repository.

## Relation to the original project

This work builds on the public Cal-QL implementation and JaxCQL. Cal-QL was
introduced in:

> Nakamoto et al., “Cal-QL: Calibrated Offline RL Pre-Training for Efficient
> Online Fine-Tuning,” 2023. <https://arxiv.org/abs/2303.05479>

The original repository and project page are:

- <https://github.com/nakamotoo/Cal-QL>
- <https://nakamotoo.github.io/projects/Cal-QL/>
