#!/bin/bash
#SBATCH --job-name=full_pipeline
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast/logs/full_pipeline_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast/logs/full_pipeline_%j.err

# Build steps 1-5 of the new R-GCN pipeline:
#   (1) Planted-edge SBM validation (R-GCN edge intervention validation)
#   (2) Joint Polity+edge intervention (interaction-term decomposition)
#   (3) Train GRU temporal head on R-GCN embeddings
#   (4) 2017-2040 forecast rollout for baseline + 8 regime-shock scenarios
#   (5) Generate all paper figures from the resulting CSVs
#
# Prerequisites:
#   - hpc/job_diagnostic_v3.sh has run successfully and saved
#     outputs/diagnostic_v3/diagnostic_v3_encoder.pt
#   - hpc/job_interventions.sh has run successfully (multi_focal_edge sweep
#     and regime_shock empirical CSVs are already in outputs/)

set -uo pipefail

cd /storage/group/LiberalArts/default/jfe4_collab/GNN_forecast
mkdir -p logs

module purge
module load python/3.11.2 cuda/12.6.0 r/4.3.1
source .venv/bin/activate

ENCODER="outputs/diagnostic_v3/diagnostic_v3_encoder.pt"
if [ ! -f "$ENCODER" ]; then
  echo "ERROR: encoder weights not found at $ENCODER"
  exit 1
fi

echo "================================================================"
echo "1) Planted-edge SBM validation (10x{planted, null})"
echo "================================================================"
PYTHONPATH=src python scripts/09_run_planted_edge_simulation.py \
    --n-replicates 10 --n-planted-edges 12 \
    --out-dir outputs/regime_shock_simulation

echo ""
echo "================================================================"
echo "2) Joint Polity+edge intervention (4 focal scenarios)"
echo "================================================================"
PYTHONPATH=src python scripts/13_run_joint_intervention.py \
    --encoder "$ENCODER" \
    --data-dir data/processed \
    --out-dir outputs/joint_intervention

echo ""
echo "================================================================"
echo "3) Train GRU temporal head"
echo "================================================================"
PYTHONPATH=src python scripts/07_train_gru_v3.py \
    --encoder "$ENCODER" \
    --data-dir data/processed \
    --out-dir outputs/forecast \
    --num-epochs 200

echo ""
echo "================================================================"
echo "4) 2017-2040 forecast rollout"
echo "================================================================"
PYTHONPATH=src python scripts/run_forecast_v3.py \
    --encoder "$ENCODER" \
    --gru outputs/forecast/gru_weights.pt \
    --data-dir data/processed \
    --forecast-until 2040 \
    --out-dir outputs/forecast

echo ""
echo "================================================================"
echo "5) Generate paper figures"
echo "================================================================"
PYTHONPATH=src python scripts/16_make_paper_figures.py \
    --outputs-dir outputs --figures-dir outputs/figures

echo ""
echo "Done: $(date)"
ls -la outputs/figures/
