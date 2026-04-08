#!/bin/bash
# ============================================================
# setup_env.sh — One-time environment setup for GNN Forecast
# on Penn State ROAR Collab (RC).
#
# Uses RC module system + Python venv.
# Run this ONCE on a login node after cloning the repo:
#   cd /scratch/jfe4/GNN_forecast
#   bash hpc/setup_env.sh
#
# Before running, verify module versions with:
#   module spider python
#   module spider cuda
#   module spider r
# Then edit the PYTHON_MOD / CUDA_MOD / R_MOD variables below.
# ============================================================
set -euo pipefail

# ----------------------------------------------------------
# CONFIGURATION — edit these to match your RC module versions
# ----------------------------------------------------------
PYTHON_MOD="python/3.11.2"
CUDA_MOD="cuda/12.6.0"
R_MOD="r/4.3.1"
# CUDA_TAG should match the CUDA major.minor for PyTorch wheels
# e.g. cuda/12.6.0 → cu126, cuda/11.8.0 → cu118
CUDA_TAG="cu126"

PROJECT_DIR="/scratch/jfe4/GNN_forecast"
VENV_DIR="${PROJECT_DIR}/.venv"

echo "=== GNN Forecast: ROAR Collab environment setup ==="
echo "Project directory: ${PROJECT_DIR}"

# ----------------------------------------------------------
# 1. Load modules
# ----------------------------------------------------------
module purge
module load "${PYTHON_MOD}"
module load "${CUDA_MOD}"
module load "${R_MOD}"

echo "Loaded modules:"
module list

# ----------------------------------------------------------
# 2. Create Python venv
# ----------------------------------------------------------
if [ -d "${VENV_DIR}" ]; then
    echo "Virtual environment already exists at ${VENV_DIR}"
    echo "To recreate, delete it first:  rm -rf ${VENV_DIR}"
else
    echo "Creating virtual environment at ${VENV_DIR} ..."
    python -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# ----------------------------------------------------------
# 3. Upgrade pip, install core deps
# ----------------------------------------------------------
pip install --upgrade pip setuptools wheel

# ----------------------------------------------------------
# 4. Install PyTorch with CUDA support
#    Check https://pytorch.org/get-started/locally/ for the
#    correct index URL matching your CUDA module version.
# ----------------------------------------------------------
pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# ----------------------------------------------------------
# 5. Install torch_geometric and its dependencies
#    After PyTorch installs, detect its version for the PyG
#    wheel index automatically.
# ----------------------------------------------------------
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
echo "PyG wheel index: ${PYG_WHEEL_URL}"

pip install torch_geometric
# torch_spline_conv is excluded — it requires source build and is not
# used by GCN/GraphSAGE/GAT encoders in this project.
pip install torch_scatter torch_sparse torch_cluster \
    -f "${PYG_WHEEL_URL}"

# ----------------------------------------------------------
# 6. Install remaining Python dependencies
# ----------------------------------------------------------
pip install numpy pandas scipy scikit-learn matplotlib networkx pytest

# ----------------------------------------------------------
# 7. Install R packages (for peacesciencer export)
# ----------------------------------------------------------
echo "Installing R packages (dplyr, readr, fixest, peacesciencer)..."
Rscript -e '
  lib <- Sys.getenv("R_LIBS_USER")
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib, .libPaths()))
  pkgs <- c("dplyr", "readr", "fixest", "remotes")
  new <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
  if (length(new)) install.packages(new, lib = lib, repos = "https://cloud.r-project.org")
  if (!requireNamespace("peacesciencer", quietly = TRUE))
    remotes::install_github("svmiller/peacesciencer", lib = lib)
  cat("R packages OK\n")
'

# ----------------------------------------------------------
# 8. Install the project in editable mode
# ----------------------------------------------------------
cd "${PROJECT_DIR}"
pip install -e .

# ----------------------------------------------------------
# 9. Verify installation
# ----------------------------------------------------------
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG {torch_geometric.__version__}')"
python -c "import gnn_forecast; print('gnn_forecast package importable')"
Rscript -e 'library(peacesciencer); cat("peacesciencer OK\n")'

echo ""
echo "=== Setup complete ==="
echo "Activate the environment in future sessions with:"
echo "  module load python/3.11.2 cuda/12.6.0 r/4.3.1"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Save these module versions — they are also used in hpc/activate.sh"
