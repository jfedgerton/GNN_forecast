#!/bin/bash
# ============================================================
# job_export_data.sh — SLURM job: export peacesciencer layers
# Submit: sbatch hpc/job_export_data.sh
# ============================================================
#SBATCH --job-name=gnn_export
#SBATCH --account=jfe4_cr_default
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/jfe4/GNN_forecast/logs/export_%j.out
#SBATCH --error=/scratch/jfe4/GNN_forecast/logs/export_%j.err

set -euo pipefail

cd /scratch/jfe4/GNN_forecast
mkdir -p logs data/processed

module purge
module load r/4.3.1

echo "=== Exporting peacesciencer layers ==="
echo "Start: $(date)"

Rscript scripts/export_peacesciencer_layers.R

echo "Done: $(date)"
