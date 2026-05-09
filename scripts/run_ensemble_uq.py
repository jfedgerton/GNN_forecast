#!/usr/bin/env python3
"""Ensemble UQ + bootstrap CIs (paper §5.4).

Train M=10 R-GCN encoders with seeds 123-132, run multi_focal_edge_sweep
on each, aggregate per-cell to get mean / 2.5% / 97.5% percentile-
bootstrap CIs on every wedge. Flag cells whose CI excludes zero as
"significant".

Outputs:
  outputs/uncertainty/ensemble_seed{S}_sweep.csv     (per-seed long-form)
  outputs/uncertainty/bootstrap_counterfactual_cis.csv  (per cell mean/lo/hi/sig)
  outputs/uncertainty/top_significant_wedges.csv    (top-K significant wedges)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from gnn_forecast.diagnostic_v3 import (
    discover_sparse_layers, build_sparse_dataset,
    run_v3_diagnostic, STRATEGIC_LAYER_PATTERNS,
)
from gnn_forecast.heterogeneous_model import (
    HeterogeneousEncoder, HeterogeneousEncoderConfig,
)
from gnn_forecast.node_features import load_node_features
from gnn_forecast.edge_intervention import (
    multi_focal_edge_sweep, pairwise_wedges,
)
from gnn_forecast.counterfactual import FOUR_FOCAL_CCODES

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
    ap.add_argument("--n-members", type=int, default=10)
    ap.add_argument("--num-epochs", type=int, default=200)
    ap.add_argument("--symmetric-n-edges", type=int, default=5)
    ap.add_argument("--out-dir", default="outputs/uncertainty")
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

    dataset = build_sparse_dataset(
        cow_filtered, year_range=(args.start_year, args.end_year),
    )
    feat_set = load_node_features(
        Path(args.node_features), dataset.ccode_to_idx, dataset.years,
    )
    print(f"Dataset: {len(dataset.years)} years, {dataset.num_nodes} nodes, "
          f"{len(dataset.layer_names)} layers")

    focal_ccodes = [cc for cc in FOUR_FOCAL_CCODES if cc in dataset.ccode_to_idx]
    print(f"Ensemble: {args.n_members} encoders, focal_ccodes={focal_ccodes}")

    seeds = list(range(SEED, SEED + args.n_members))
    per_seed_sweeps: List[pd.DataFrame] = []

    for i, seed in enumerate(seeds):
        print(f"\n=== Ensemble member {i+1}/{args.n_members} (seed={seed}) ===")
        torch.manual_seed(seed); np.random.seed(seed)
        # Train encoder
        encoder_state_path = str(out_dir / f"encoder_seed{seed}.pt")
        result = run_v3_diagnostic(
            dataset, feat_set.by_year, feat_set.num_features,
            num_epochs=args.num_epochs, seed=seed,
            encoder_state_path=encoder_state_path,
        )
        # Reload encoder weights from saved best-epoch state
        payload = torch.load(encoder_state_path, map_location="cpu", weights_only=False)
        cfg = HeterogeneousEncoderConfig(**payload["encoder_config"])
        encoder = HeterogeneousEncoder(payload["num_nodes"], cfg)
        encoder.load_state_dict(payload["encoder_state_dict"])
        encoder.eval()
        # Run multi-focal sweep
        long_df = multi_focal_edge_sweep(
            encoder=encoder, dataset=dataset, feat_set=feat_set,
            focal_ccodes=focal_ccodes,
            symmetric_n_edges=args.symmetric_n_edges,
            progress_every=10_000,
        )
        long_df["seed"] = seed
        long_df["best_link_auc"] = result.best_link_auc
        per_seed_path = out_dir / f"ensemble_seed{seed}_sweep.csv"
        long_df.to_csv(per_seed_path, index=False)
        print(f"  wrote {per_seed_path} ({len(long_df)} rows, best_auc={result.best_link_auc:.4f})")
        per_seed_sweeps.append(long_df)

    # Combine across seeds, aggregate per (partner, layer, op, focal)
    print(f"\n=== Aggregating ensemble ({len(per_seed_sweeps)} members) ===")
    combined = pd.concat(per_seed_sweeps, ignore_index=True)
    group_cols = ["partner_ccode", "layer_name", "operation", "focal_ccode"]
    agg = (
        combined.groupby(group_cols)
        .agg(
            delta_centroid_mean=("delta_centroid", "mean"),
            delta_centroid_lo=("delta_centroid", lambda x: float(np.percentile(x, 2.5))),
            delta_centroid_hi=("delta_centroid", lambda x: float(np.percentile(x, 97.5))),
            delta_centroid_sd=("delta_centroid", "std"),
            delta_cosine_mean=("delta_cosine", "mean"),
            delta_cosine_lo=("delta_cosine", lambda x: float(np.percentile(x, 2.5))),
            delta_cosine_hi=("delta_cosine", lambda x: float(np.percentile(x, 97.5))),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    agg["wedge_significant_centroid"] = (
        (agg["delta_centroid_lo"] > 0) | (agg["delta_centroid_hi"] < 0)
    ).astype(int)
    agg["wedge_significant_cosine"] = (
        (agg["delta_cosine_lo"] > 0) | (agg["delta_cosine_hi"] < 0)
    ).astype(int)
    cis_path = out_dir / "bootstrap_counterfactual_cis.csv"
    agg.to_csv(cis_path, index=False)
    print(f"Wrote {cis_path}")

    # Top-20 significant wedges (sorted by |mean|)
    sig = agg[agg["wedge_significant_centroid"] == 1].copy()
    sig["abs_mean"] = sig["delta_centroid_mean"].abs()
    top = sig.sort_values("abs_mean", ascending=False).head(20)
    top_path = out_dir / "top_significant_wedges.csv"
    top.to_csv(top_path, index=False)
    print(f"Wrote {top_path} ({len(top)} of {len(agg)} cells significant)")


if __name__ == "__main__":
    main()
