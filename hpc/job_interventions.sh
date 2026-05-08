#!/bin/bash
#SBATCH --job-name=interventions
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast/logs/interventions_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast/logs/interventions_%j.err

# Run the new intervention pipeline:
#   1. Multi-focal edge sweep (USA/CHN/RUS/IND) on the trained R-GCN encoder
#   2. Empirical regime-shock counterfactuals (8 Polity/CINC scenarios)
#   3. Planted-feature-shock SBM validation (10x{planted,null} replicates)
#
# Prerequisites: hpc/job_diagnostic_v3.sh has run successfully and saved
# outputs/diagnostic_v3/diagnostic_v3_encoder.pt. Submit this job AFTER
# that diagnostic completes, or chain with --dependency=afterok.

set -uo pipefail

cd /storage/group/LiberalArts/default/jfe4_collab/GNN_forecast
mkdir -p logs

module purge
module load python/3.11.2 cuda/12.6.0 r/4.3.1
source .venv/bin/activate

ENCODER="outputs/diagnostic_v3/diagnostic_v3_encoder.pt"
if [ ! -f "$ENCODER" ]; then
  echo "ERROR: encoder weights not found at $ENCODER"
  echo "Run hpc/job_diagnostic_v3.sh first."
  exit 1
fi

echo "================================================================"
echo "1) Multi-focal edge sweep (USA/CHN/RUS/IND)"
echo "================================================================"
PYTHONPATH=src python scripts/run_multi_focal_edge_sweep.py \
    --encoder "$ENCODER" \
    --data-dir data/processed \
    --out-dir outputs/multi_focal_edge \
    --symmetric-n-edges 5

echo ""
echo "================================================================"
echo "2) Empirical regime-shock counterfactuals (Polity + CINC, 4 focals)"
echo "================================================================"
PYTHONPATH=src python scripts/run_regime_shock_empirical.py \
    --encoder "$ENCODER" \
    --data-dir data/processed \
    --out-dir outputs/regime_shock

echo ""
echo "================================================================"
echo "3) Planted-feature-shock SBM validation (10x{planted, null})"
echo "================================================================"
PYTHONPATH=src python scripts/run_regime_shock_simulation.py \
    --n-replicates 10 \
    --z-shift 3.0 \
    --out-dir outputs/regime_shock_simulation

echo ""
echo "Done: $(date)"
