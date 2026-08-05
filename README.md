# Cal-QL on Minari MuJoCo

This repository contains our two-person Deep Learning Lab project at the
University of Freiburg. Over roughly two months, we adapted the public JAX
implementation of Cal-QL to the Minari MuJoCo datasets and ran offline-to-online
experiments on Hopper, Half-Cheetah, Walker2d, and Humanoid.

Our main question was whether Cal-QL transfers an offline policy to online
learning faster than the comparison methods. We therefore looked at the full
learning curves as well as the last evaluation return. On Humanoid, we also
tried replay mixing, a larger critic, a fixed fall penalty, and a learned
`Q_fall(s,a)` critic.

The project uses only three setup documents:

- this README for a short overview;
- [DLL.md](DLL.md) for our setup, commands, changes, and observations;
- [requirements.txt](requirements.txt) for the Python packages.

## Short CPU setup

We used Ubuntu through WSL and Miniforge:

```bash
git clone https://github.com/SumitSkp/cal-ql-minari.git
cd cal-ql-minari

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda create --name calql-minari python=3.10 numpy=1.26.4 cython=0.29.37 pip -y
conda activate calql-minari

export PYTHONNOUSERSITE=1
python -m pip install --no-user -r requirements.txt
python scripts/smoke_test_safety.py
```

The complete setup, Minari download, CPU/GPU checks, and experiment commands
are written in [DLL.md](DLL.md). W&B is offline in the small tests. We never
store a W&B API key in the repository.

## Original work

This project builds on the public Cal-QL implementation:

> Nakamoto et al., “Cal-QL: Calibrated Offline RL Pre-Training for Efficient
> Online Fine-Tuning,” 2023. <https://arxiv.org/abs/2303.05479>

- Original repository: <https://github.com/nakamotoo/Cal-QL>
- Project page: <https://nakamotoo.github.io/projects/Cal-QL/>
