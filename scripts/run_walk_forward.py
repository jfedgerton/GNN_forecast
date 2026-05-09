#!/usr/bin/env python3
"""Walk-forward backtest at 5/10/15/20-year horizons (paper §5.1).

Splits at 1985, 1990, 1995, 2000, 2005. For each split, train R-GCN on
[start, t_split], roll GRU forward, score against held-out years
t_split + {5, 10, 15, 20}.

Output:
  outputs/walk_forward/walk_forward_backtest.csv     (long-form per row)
  outputs/walk_forward/walk_forward_summary.csv      (aggregated by horizon)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn_forecast.diagnostic_v3 import (
    discover_sparse_layers, build_sparse_dataset,
)
from gnn_forecast.node_features import load_node_features
from gnn_forecast.walk_forward import run_walk_forward_study

SEED = 123


def filter_layer_to_cow(df: pd.DataFrame, cow_set: set) -> pd.DataFrame:
    src_in = [(int(c), int(y)) in cow_set for c, y in zip(df["source_ccode"], df["year"])]
    tgt_in = [(int(c), int(y)) in cow_set for c, y in zip(df["target_ccode"], df["year"])]
    keep = pd.Series(src_in) & pd.Series(tgt_in)
    return df[keep.values].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--cow-membership", default="data/processed/cow_state_membership.csv")
    ap.add_argument("--node-features", default="data/processed/node_features.csv")
    ap.add_argument("--start-year", type=int, default=1948)
    ap.add_argument("--end-year", type=int, default=2016)
    ap.add_argument("--split-years", nargs="+", type=int,
                    default=[1985, 1990, 1995, 2000, 2005])
    ap.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 15, 20])
    ap.add_argument("--encoder-epochs", type=int, default=200)
    ap.add_argument("--gru-epochs", type=int, default=200)
    ap.add_argument("--out-dir", default="outputs/walk_forward")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cow = pd.read_csv(args.cow_membership)
    cow["ccode"] = cow["ccode"].astype(int); cow["year"] = cow["year"].astype(int)
    cow_set = set(zip(cow["ccode"], cow["year"]))
    layer_dfs = discover_sparse_layers(Path(args.data_dir))
    cow_filtered = {}
    for name, df in layer_dfs.items():
        df_cow = filter_layer_to_cow(df, cow_set)
        years_in_range = df_cow[
            (df_cow["year"] >= args.start_year)
            & (df_cow["year"] <= args.end_year)
        ].copy()
        cow_filtered[name] = years_in_range

    full_dataset = build_sparse_dataset(
        cow_filtered, year_range=(args.start_year, args.end_year),
    )
    full_feats = load_node_features(
        Path(args.node_features), full_dataset.ccode_to_idx, full_dataset.years,
    )
    print(f"Full dataset: {len(full_dataset.years)} years, "
          f"{full_dataset.num_nodes} nodes, {len(full_dataset.layer_names)} layers")

    df = run_walk_forward_study(
        full_dataset=full_dataset, full_feats=full_feats,
        split_years=args.split_years, horizons=args.horizons,
        encoder_epochs=args.encoder_epochs, gru_epochs=args.gru_epochs,
    )

    backtest_path = out_dir / "walk_forward_backtest.csv"
    df.to_csv(backtest_path, index=False)
    print(f"\nWrote {backtest_path}")

    # Aggregate by horizon
    agg = (
        df.groupby("horizon")
          .agg(
              mean_mse=("embedding_mse", "mean"),
              mean_auc=("link_pred_auc", "mean"),
              mean_cd_drift=("centroid_drift", "mean"),
              n_splits=("split_year", "nunique"),
          )
          .reset_index()
    )
    summary_path = out_dir / "walk_forward_summary.csv"
    agg.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}\n")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
