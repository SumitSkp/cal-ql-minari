#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONNOUSERSITE=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m pip check
python -m py_compile \
  JaxCQL/conservative_sac.py \
  JaxCQL/conservative_sac_main.py \
  JaxCQL/minari_compat.py \
  JaxCQL/replay_buffer.py \
  JaxCQL/sampler.py \
  scripts/smoke_test_safety.py
python scripts/smoke_test_safety.py

echo "submission checks passed"
