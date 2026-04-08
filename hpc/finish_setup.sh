#!/bin/bash
# ============================================================
# finish_setup.sh — Complete setup after torch_spline_conv failure
# Run from /scratch/jfe4/GNN_forecast:
#   bash hpc/finish_setup.sh
# ============================================================
set -euo pipefail

cd /scratch/jfe4/GNN_forecast

module purge
module load python/3.11.2
module load cuda/12.6.0
module load r/4.3.1

source .venv/bin/activate

echo "=== Installing remaining Python deps ==="
pip install scipy scikit-learn matplotlib pytest

echo "=== Installing project in editable mode ==="
pip install -e .

echo "=== Installing R packages ==="
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

echo ""
echo "=== Verification ==="
python -c "import torch; print('PyTorch ' + torch.__version__ + ', CUDA: ' + str(torch.cuda.is_available()))"
python -c "import torch_geometric; print('PyG ' + torch_geometric.__version__)"
python -c "import gnn_forecast; print('gnn_forecast OK')"
Rscript -e 'library(peacesciencer); cat("peacesciencer OK\n")'

echo ""
echo "=== Setup complete ==="
echo "To use: source hpc/activate.sh"
