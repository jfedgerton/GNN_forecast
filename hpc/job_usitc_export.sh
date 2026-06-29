#!/bin/bash
#SBATCH --job-name=usitc_exp
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast/logs/usitc_exp_%j.out
#SBATCH --error=/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast/logs/usitc_exp_%j.err

# Pull USITC DGD agreement layers (FTA, PTA-services, CU) and write 3 CSVs.
#
# Uses r/4.5.0 (NOT 4.3.1) because the cli/vctrs/duckdb binaries already
# in ~/R/libs were compiled against R 4.4+ and have ABI mismatches with
# r/4.3.1 (undefined symbol R_getVar / Rf_charIsASCII).
#
# Outputs are CSVs and are R-version-agnostic, so the rest of the pipeline
# (the Python diagnostic, training, etc.) keeps using r/4.3.1 + the same venv.

set -uo pipefail

cd /storage/group/LiberalArts/default/jfe4_collab/GNN_forecast
mkdir -p logs

module purge
module load r/4.5.0

echo "=== R version sanity check ==="
Rscript -e 'cat(R.version.string, "\n")'

echo "=== Verifying usitcgravity loads under R 4.5 ==="
Rscript -e 'suppressPackageStartupMessages(library(usitcgravity)); cat("usitcgravity OK\n")' \
  || { echo "usitcgravity failed to load. Aborting."; exit 1; }

echo "=== Running export script ==="
Rscript scripts/04_export_usitc_gravity_layers.R 1948 2016

echo "=== Done. Layer inventory: ==="
ls -la data/processed/layer_*.csv
