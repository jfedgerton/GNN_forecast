#!/bin/bash
#SBATCH --job-name=gnn_diag3
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/jfe4/GNN_forecast/logs/diag3_%j.out
#SBATCH --error=/scratch/jfe4/GNN_forecast/logs/diag3_%j.err

set -euo pipefail

cd /scratch/jfe4/GNN_forecast
mkdir -p logs outputs/diagnostic_v3

source hpc/activate.sh

# Pre-flight checks: required input files
required=(
  "data/processed/cow_state_membership.csv"
  "data/processed/node_features.csv"
  "data/processed/layer_alliances_defensive_offensive_undirected.csv"
  "data/processed/layer_dca_undirected.csv"
)
for f in "${required[@]}"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Required input missing: $f"
    echo "Build prerequisite data files first:"
    echo "  Rscript scripts/01_export_cow_membership.R"
    echo "  Rscript scripts/export_country_year_features_simple.R 1948 2016"
    echo "  python scripts/05_convert_kinne_dca.py"
    echo "  Rscript scripts/04_export_usitc_gravity_layers.R"
    exit 1
  fi
done

echo "=== Diagnostic v3: sparse strategic layers + stratified-negatives AUC ==="
echo "Start: $(date)"

python scripts/06_run_diagnostic_v3.py \
    --data-dir data/processed \
    --cow-membership data/processed/cow_state_membership.csv \
    --node-features data/processed/node_features.csv \
    --start-year 1948 \
    --end-year 2016 \
    --num-epochs 200 \
    --out-dir outputs/diagnostic_v3

echo "Done: $(date)"
