#!/bin/bash
#SBATCH --job-name=gnn_diag2
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/jfe4/GNN_forecast/logs/diag2_%j.out
#SBATCH --error=/scratch/jfe4/GNN_forecast/logs/diag2_%j.err

set -euo pipefail

cd /scratch/jfe4/GNN_forecast
mkdir -p logs outputs/diagnostic_v2

source hpc/activate.sh

echo "=== Diagnostic v2: R-GCN + rich features ==="
echo "Start: $(date)"

# Confirm the node features file exists
if [ ! -f data/processed/node_features.csv ]; then
  echo "ERROR: data/processed/node_features.csv missing."
  echo "Run scripts/02_export_country_year_features.R first to build it."
  exit 1
fi

python scripts/run_diagnostic_v2.py \
    --data-dir data/processed \
    --node-features data/processed/node_features.csv \
    --start-year 1945 \
    --end-year 2025 \
    --num-epochs 200 \
    --out-dir outputs/diagnostic_v2

echo "Done: $(date)"
