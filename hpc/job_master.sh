#!/bin/bash
# ============================================================
# job_master.sh — SLURM job: full master pipeline (GPU)
# Submit: sbatch hpc/job_master.sh
# ============================================================
#SBATCH --job-name=gnn_master
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=sla-prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jfe4/GNN_forecast/logs/master_%j.out
#SBATCH --error=/scratch/jfe4/GNN_forecast/logs/master_%j.err

set -euo pipefail

cd /scratch/jfe4/GNN_forecast
mkdir -p logs outputs

source hpc/activate.sh

echo "=== Master Pipeline ==="
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"

# Use full processed data if available, otherwise fall back to fixtures
if [ -d "data/processed" ] && [ -f "data/processed/nodes.csv" ]; then
    DATA_DIR="data/processed"
else
    DATA_DIR="tests/fixtures/tiny_processed"
    echo "WARNING: Using test fixtures — run job_export_data.sh first for full data"
fi

python scripts/master.py \
    --data-dir "${DATA_DIR}" \
    --summary

echo "Done: $(date)"
