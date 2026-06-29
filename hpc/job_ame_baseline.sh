#!/bin/bash
#SBATCH --job-name=ame_baseline
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast/logs/ame_baseline_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast/logs/ame_baseline_%j.err
set -uo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/GNN_forecast
mkdir -p logs
module purge
module load r/4.5.0

echo "=== Step 1: fit AME on each (layer, year) cell ==="
Rscript scripts/10_run_ame_baseline.R 1948 2016

echo "=== Step 2: compare AME latents to R-GCN embeddings ==="
module load python/3.11.2
source .venv/bin/activate
PYTHONPATH=src python scripts/11_compare_ame_to_gnn.py \
    --encoder outputs/diagnostic_v3/diagnostic_v3_encoder.pt \
    --ame-dir outputs/ame_baseline \
    --data-dir data/processed
echo "Done: $(date)"
