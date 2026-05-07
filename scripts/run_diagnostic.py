#!/usr/bin/env python3
"""Phase 1 diagnostic ablation runner.

Runs the five small experiments in src/gnn_forecast/diagnostic.py to
identify which component of the multiplex GNN setup is preventing
informative embedding learning on real data.

Usage:
    PYTHONPATH=src python scripts/run_diagnostic.py \
        --data-dir data/processed \
        --start-year 1945 --end-year 2025 \
        --out-dir outputs/diagnostic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from gnn_forecast.multiplex_data import (
    discover_layers,
    build_global_node_index,
    build_multiplex_dataset,
)
from gnn_forecast.diagnostic import run_all_diagnostics

SEED = 123


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="tests/fixtures/tiny_processed")
    ap.add_argument("--nodes-csv", default=None)
    ap.add_argument("--start-year", type=int, default=1945)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--out-dir", default="outputs/diagnostic")
    ap.add_argument("--skip-pyg", action="store_true",
                    help="Skip experiment 3 (PyG GAT/SAGE backbone)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Loading dataset from {args.data_dir}")
    layers = discover_layers(Path(args.data_dir))
    nodes_csv = Path(args.nodes_csv) if args.nodes_csv else Path(args.data_dir) / "nodes.csv"
    ccode_to_idx, idx_to_ccode, nodes_df = build_global_node_index(layers, nodes_csv)
    dataset = build_multiplex_dataset(
        layers, ccode_to_idx, idx_to_ccode, nodes_df,
        year_range=(args.start_year, args.end_year),
    )
    print(f"  {len(dataset.years)} years ({dataset.years[0]}-{dataset.years[-1]}), "
          f"{dataset.num_nodes} nodes, {len(dataset.layer_names)} layers")

    summary = run_all_diagnostics(dataset, args.out_dir, skip_pyg=args.skip_pyg)
    print("\n=== DIAGNOSTIC SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
