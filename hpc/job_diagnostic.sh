#!/bin/bash
#SBATCH --job-name=gnn_diag
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/jfe4/GNN_forecast/logs/diag_%j.out
#SBATCH --error=/scratch/jfe4/GNN_forecast/logs/diag_%j.err

set -euo pipefail

cd /scratch/jfe4/GNN_forecast
mkdir -p logs outputs/diagnostic

source hpc/activate.sh

echo "=== Phase 1 Diagnostic Ablation ==="
echo "Start: $(date)"

python scripts/run_diagnostic.py \
    --data-dir data/processed \
    --start-year 1945 \
    --end-year 2025 \
    --out-dir outputs/diagnostic

echo "Done: $(date)"
