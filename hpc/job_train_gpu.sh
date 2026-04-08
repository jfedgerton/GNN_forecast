#!/bin/bash
# ============================================================
# job_train_gpu.sh — SLURM job: GNN training + forecast + interventions (GPU)
# Submit: sbatch hpc/job_train_gpu.sh
# ============================================================
#SBATCH --job-name=gnn_train
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=sla-prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jfe4/GNN_forecast/logs/train_%j.out
#SBATCH --error=/scratch/jfe4/GNN_forecast/logs/train_%j.err

set -euo pipefail

cd /scratch/jfe4/GNN_forecast
mkdir -p logs outputs

source hpc/activate.sh

echo "=== GNN Training + Forecast + Interventions ==="
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"

python scripts/train_forecast_and_intervene.py \
    --embedding-file data/model_inputs/embedding_history.pt \
    --adj-file data/model_inputs/adjacency_2025.npy \
    --nodes-file data/processed/nodes.csv \
    --out-dir outputs

echo "Done: $(date)"
