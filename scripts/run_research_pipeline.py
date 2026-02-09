#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gnn_forecast.pipeline import PipelineConfig, export_year_layers, run_research_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--start-year", type=int, default=1945)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--out-dir", default="data/model_inputs")
    args = ap.parse_args()

    config = PipelineConfig(
        data_dir=args.data_dir,
        years_start=args.start_year,
        years_end=args.end_year,
    )
    layer_map = run_research_pipeline(config)
    export_year_layers(layer_map, Path(args.out_dir))
    print(f"Exported processed layers to {args.out_dir}")


if __name__ == "__main__":
    main()
