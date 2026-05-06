#!/usr/bin/env python3
"""Walk-forward backtest at horizons {5, 10, 15, 20} years.

Replaces the older single-25-year-point-prediction framing. Reports MSE
and link AUC at each horizon for each split year, plus a per-horizon
summary across folds.

Usage:
    PYTHONPATH=src python scripts/run_multi_horizon_backtest.py \
        --data-dir data/processed \
        --start-year 1945 --end-year 2025 \
        --split-years 1985 1990 1995 2000 2005 \
        --out-dir outputs/multi_horizon
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
from gnn_forecast.multiplex_model import MultiplexGNNConfig
from gnn_forecast.training import TrainingConfig
from gnn_forecast.validation import multi_horizon_backtest, HORIZONS

SEED = 123


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="tests/fixtures/tiny_processed")
    ap.add_argument("--nodes-csv", default=None)
    ap.add_argument("--start-year", type=int, default=1945)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--split-years", nargs="+", type=int,
                    default=[1985, 1990, 1995, 2000, 2005],
                    help="Training cutoffs to evaluate")
    ap.add_argument("--horizons", nargs="+", type=int,
                    default=list(HORIZONS),
                    help="Forecast horizons to report (default: 5 10 15 20)")
    ap.add_argument("--out-dir", default="outputs/multi_horizon")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    layers = discover_layers(Path(args.data_dir))
    nodes_csv = Path(args.nodes_csv) if args.nodes_csv else Path(args.data_dir) / "nodes.csv"
    ccode_to_idx, idx_to_ccode, nodes_df = build_global_node_index(layers, nodes_csv)
    dataset = build_multiplex_dataset(
        layers, ccode_to_idx, idx_to_ccode, nodes_df,
        year_range=(args.start_year, args.end_year),
    )
    print(f"Loaded {len(dataset.years)} years, {dataset.num_nodes} nodes, {len(dataset.layer_names)} layers")

    model_cfg = MultiplexGNNConfig(
        num_layers=len(dataset.layer_names),
        in_dim=len(dataset.layer_names),
        hidden_dim=64, emb_dim=32, seq_len=5,
    )
    train_cfg = TrainingConfig(num_epochs=100, patience=20, print_every=25, seq_len=5)

    df = multi_horizon_backtest(
        dataset=dataset,
        split_years=args.split_years,
        horizons=tuple(args.horizons),
        model_config=model_cfg,
        train_config=train_cfg,
        save_dir=args.out_dir,
    )
    print(f"\nWrote {len(df)} (fold × horizon) rows to {args.out_dir}")


if __name__ == "__main__":
    main()
