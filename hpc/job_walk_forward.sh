#!/bin/bash
#SBATCH --job-name=walk_fwd
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast/logs/walk_fwd_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast/logs/walk_fwd_%j.err
set -uo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/GNN_forecast
mkdir -p logs
module purge
module load python/3.11.2 cuda/12.6.0 r/4.3.1
source .venv/bin/activate

PYTHONPATH=src python scripts/run_walk_forward.py \
    --data-dir data/processed \
    --split-years 1985 1990 1995 2000 2005 \
    --horizons 5 10 15 20 \
    --encoder-epochs 200 --gru-epochs 200 \
    --out-dir outputs/walk_forward
echo "Done: $(date)"
