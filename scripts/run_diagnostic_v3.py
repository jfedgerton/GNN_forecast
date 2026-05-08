#!/usr/bin/env python3
"""Diagnostic v3 runner — sparse strategic layers, COW universe, stratified-negatives AUC.

Pipeline (all merges audited):
  1. Load COW state membership panel.
  2. Discover sparse strategic layers in data/processed/ (alliances, DCA,
     FTA, PTA-goods, CU, EIA — whichever are present).
  3. For each layer, drop rows with ccodes/years not in COW.
  4. Build the multiplex dataset on the COW-only universe.
  5. Load node features (already z-scored), align to COW node index.
  6. Train R-GCN + InfoNCE; evaluate stratified-negatives link AUC.

Cross-layer merge audit prints:
  - per-layer year coverage and density after COW filtering
  - countries present in some layers but missing in others
  - final node count after intersecting all layer ccode sets

Usage:
    PYTHONPATH=src python scripts/run_diagnostic_v3.py \\
        --data-dir data/processed \\
        --node-features data/processed/node_features.csv \\
        --cow-membership data/processed/cow_state_membership.csv \\
        --start-year 1948 --end-year 2016 \\
        --out-dir outputs/diagnostic_v3
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from gnn_forecast.diagnostic_v3 import (
    discover_sparse_layers,
    build_sparse_dataset,
    run_v3_diagnostic,
    STRATEGIC_LAYER_PATTERNS,
)
from gnn_forecast.node_features import load_node_features

SEED = 123


def filter_layer_to_cow(df: pd.DataFrame, cow_set: set) -> pd.DataFrame:
    """Keep only rows where (source_ccode, year) AND (target_ccode, year) are COW."""
    src_in = [(int(c), int(y)) in cow_set for c, y in zip(df["source_ccode"], df["year"])]
    tgt_in = [(int(c), int(y)) in cow_set for c, y in zip(df["target_ccode"], df["year"])]
    keep = pd.Series(src_in) & pd.Series(tgt_in)
    return df[keep.values].reset_index(drop=True)


def merge_audit(
    layer_dfs: Dict[str, pd.DataFrame],
    cow: pd.DataFrame,
) -> None:
    """Print a careful cross-layer merge report."""
    print("\n" + "=" * 60)
    print("MERGE AUDIT")
    print("=" * 60)

    cow_ccodes = set(cow["ccode"].astype(int))
    cow_years = set(cow["year"].astype(int))
    print(f"COW universe: {len(cow_ccodes)} unique ccodes, "
          f"years {min(cow_years)}-{max(cow_years)}")

    per_layer_ccodes: Dict[str, set] = {}
    print("\nPer-layer coverage (after COW filter):")
    for name, df in layer_dfs.items():
        cc = set(df["source_ccode"].astype(int)) | set(df["target_ccode"].astype(int))
        per_layer_ccodes[name] = cc
        years = set(df["year"].astype(int))
        density = (df["tie"].astype(int) == 1).mean() if len(df) else 0.0
        print(f"  {name:>22s}: {len(cc):>4d} ccodes, "
              f"{len(years):>3d} years ({min(years) if years else '-'}-{max(years) if years else '-'}), "
              f"{(df['tie'].astype(int) == 1).sum():>7,} positive ties, "
              f"density {density:.4%}")

    # Pairwise overlap
    if len(per_layer_ccodes) >= 2:
        print("\nPairwise ccode overlap (intersection size):")
        names = list(per_layer_ccodes.keys())
        for i, a in enumerate(names):
            for b in names[i+1:]:
                inter = per_layer_ccodes[a] & per_layer_ccodes[b]
                print(f"  {a} ∩ {b}: {len(inter)}")

    # Universe intersection (countries appearing in ALL active layers)
    if per_layer_ccodes:
        intersection = set.intersection(*per_layer_ccodes.values())
        union = set.union(*per_layer_ccodes.values())
        print(f"\nUnion of ccodes across all layers:        {len(union)}")
        print(f"Intersection of ccodes across all layers: {len(intersection)}")
        only_some = union - intersection
        if only_some:
            print(f"Ccodes appearing in some but not all layers: {len(only_some)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--cow-membership", default="data/processed/cow_state_membership.csv")
    ap.add_argument("--node-features", default="data/processed/node_features.csv")
    ap.add_argument("--start-year", type=int, default=1948)
    ap.add_argument("--end-year", type=int, default=2016)
    ap.add_argument("--num-epochs", type=int, default=200)
    ap.add_argument("--out-dir", default="outputs/diagnostic_v3")
    ap.add_argument("--layers", nargs="*", default=None,
                    help="Subset of strategic layers to use (default: all available). "
                         f"Options: {list(STRATEGIC_LAYER_PATTERNS.keys())}")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load COW universe ----
    cow_path = Path(args.cow_membership)
    if not cow_path.exists():
        raise FileNotFoundError(
            f"COW membership CSV missing: {cow_path}. "
            f"Run scripts/export_cow_membership.R first."
        )
    cow = pd.read_csv(cow_path)
    cow["ccode"] = cow["ccode"].astype(int)
    cow["year"] = cow["year"].astype(int)
    cow_set = set(zip(cow["ccode"], cow["year"]))

    # ---- Discover and load sparse strategic layers ----
    print("Discovering strategic layers")
    layer_dfs = discover_sparse_layers(Path(args.data_dir), requested=args.layers)
    if not layer_dfs:
        raise RuntimeError("No strategic layer files found. Build them first.")

    # ---- Apply COW filter to each layer ----
    print("\nApplying COW dyadic filter")
    cow_filtered = {}
    for name, df in layer_dfs.items():
        before = len(df)
        df_cow = filter_layer_to_cow(df, cow_set)
        years_in_range = df_cow[(df_cow["year"] >= args.start_year)
                                 & (df_cow["year"] <= args.end_year)].copy()
        cow_filtered[name] = years_in_range
        print(f"  {name}: {before:,} -> {len(years_in_range):,} (after COW + year range)")

    # Merge audit
    merge_audit(cow_filtered, cow)

    # ---- Build dataset on the COW universe ----
    print("\nBuilding multiplex dataset")
    dataset = build_sparse_dataset(
        cow_filtered,
        year_range=(args.start_year, args.end_year),
    )
    print(f"  {len(dataset.years)} years, {dataset.num_nodes} nodes, "
          f"{len(dataset.layer_names)} layers")

    # ---- Load node features ----
    print(f"\nLoading node features from {args.node_features}")
    feat_set = load_node_features(
        Path(args.node_features), dataset.ccode_to_idx, dataset.years,
    )
    print(f"  {feat_set.num_features} features per node")

    # ---- Run the diagnostic ----
    print("\n" + "=" * 60)
    print("RUNNING v3 HYPOTHESIS TEST")
    print("=" * 60)
    encoder_state_path = str(out_dir / "diagnostic_v3_encoder.pt")
    result = run_v3_diagnostic(
        dataset, feat_set.by_year, feat_set.num_features,
        num_epochs=args.num_epochs,
        encoder_state_path=encoder_state_path,
    )

    # ---- Save outputs ----
    summary = pd.DataFrame([{
        "name": result.name,
        "layer_set": ",".join(result.layer_set),
        "final_link_auc": result.final_link_auc,
        "best_link_auc": result.best_link_auc,
        "best_epoch": result.best_epoch,
        "best_mean_norm": result.best_mean_norm,
        "final_mean_norm": result.final_mean_norm,
        "n_epochs_run": result.n_epochs_run,
        "notes": result.notes,
    }])
    summary.to_csv(out_dir / "diagnostic_v3_summary.csv", index=False)
    torch.save({
        "result": result,
        "feature_names": feat_set.feature_names,
        "layer_set": result.layer_set,
        "n_nodes": dataset.num_nodes,
        "n_years": len(dataset.years),
    }, out_dir / "diagnostic_v3_results.pt")
    print(f"\nSaved to {out_dir}")

    print(f"\n=== VERDICT ===")
    print(f"  layers used:     {', '.join(result.layer_set)}")
    print(f"  best AUC:        {result.best_link_auc:.4f} (epoch {result.best_epoch}/{result.n_epochs_run})")
    print(f"  final-epoch AUC: {result.final_link_auc:.4f}")
    print(f"  best ||z||:      {result.best_mean_norm:.4f}")
    print(f"  {result.notes}")


if __name__ == "__main__":
    main()
