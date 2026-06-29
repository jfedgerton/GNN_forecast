#!/bin/bash
#SBATCH --job-name=ensemble_uq
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast/logs/ensemble_uq_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast/logs/ensemble_uq_%j.err
set -uo pipefail
cd /storage/group/LiberalArts/default/jfe4_collab/GNN_forecast
mkdir -p logs
module purge
module load python/3.11.2 cuda/12.6.0 r/4.3.1
source .venv/bin/activate

PYTHONPATH=src python scripts/14_run_ensemble_uq.py \
    --data-dir data/processed \
    --n-members 10 \
    --num-epochs 200 \
    --symmetric-n-edges 5 \
    --out-dir outputs/uncertainty
echo "Done: $(date)"
