#!/bin/bash
# ============================================================
# job_build_layers.sh — SLURM job: build multiplex network layers
# Submit: sbatch hpc/job_build_layers.sh
# ============================================================
#SBATCH --job-name=gnn_build
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/jfe4/GNN_forecast/logs/build_%j.out
#SBATCH --error=/scratch/jfe4/GNN_forecast/logs/build_%j.err

set -euo pipefail

cd /scratch/jfe4/GNN_forecast
mkdir -p logs data/model_inputs

source hpc/activate.sh

echo "=== Building multiplex network layers ==="
echo "Start: $(date)"

python scripts/run_research_pipeline.py \
    --data-dir data/processed \
    --start-year 1945 \
    --end-year 2025 \
    --out-dir data/model_inputs

echo "Done: $(date)"
