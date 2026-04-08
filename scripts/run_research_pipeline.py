#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gnn_forecast.pipeline import PipelineConfig, export_year_layers, run_research_pipeline

# -----------------------------
# Parse CLI args (no main())
# -----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/processed")
ap.add_argument("--start-year", type=int, default=1945)
ap.add_argument("--end-year", type=int, default=2025)
ap.add_argument("--out-dir", default="data/model_inputs")
args = ap.parse_args()

# -----------------------------
# Hard-coded / explicit objects
# -----------------------------
data_dir = args.data_dir
start_year = int(args.start_year)
end_year = int(args.end_year)
if start_year > end_year:
    raise ValueError(f"start_year ({start_year}) must be <= end_year ({end_year})")
out_dir = Path(args.out_dir)

# Ensure output directory exists
out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Build pipeline config
# -----------------------------
config = PipelineConfig(
    data_dir=data_dir,
    years_start=start_year,
    years_end=end_year,
)

# -----------------------------
# Run pipeline (build layer_map)
# -----------------------------
layer_map = run_research_pipeline(config)

# -----------------------------
# Export year-specific layers
# -----------------------------
export_year_layers(layer_map, out_dir)

print(f"Exported processed layers to {out_dir}")
