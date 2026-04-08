#!/bin/bash
# ============================================================
# activate.sh — Source this at the start of every session
#   source hpc/activate.sh
#
# Edit the module versions to match what setup_env.sh used.
# ============================================================

PYTHON_MOD="python/3.11.2"
CUDA_MOD="cuda/12.6.0"
R_MOD="r/4.3.1"

module purge
module load "${PYTHON_MOD}"
module load "${CUDA_MOD}"
module load "${R_MOD}"

source /scratch/jfe4/GNN_forecast/.venv/bin/activate
export PYTHONPATH=/scratch/jfe4/GNN_forecast/src:${PYTHONPATH:-}

echo "Environment ready. Python: $(python --version), CUDA: $(nvcc --version 2>/dev/null | tail -1 || echo 'not on GPU node')"
