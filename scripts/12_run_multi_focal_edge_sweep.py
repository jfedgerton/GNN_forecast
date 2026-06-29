#!/usr/bin/env python3
"""Multi-focal edge sweep for the four-focal set (USA, China, Russia, India).

For each (partner, layer, operation) edge perturbation in the focal year
(default: latest year of dataset), record per-focal embeddedness deltas
and compute pairwise wedges across all C(4,2)=6 ordered pairs.

Inputs: trained R-GCN encoder weights from outputs/diagnostic_v3/.
Outputs:
  outputs/multi_focal_edge/sweep_long.csv
  outputs/multi_focal_edge/wedges_centroid.csv
  outputs/multi_focal_edge/wedges_cosine.csv
  outputs/multi_focal_edge/top_k_wedges.csv      (top 30 per focal pair)

Usage:
    PYTHONPATH=src python scripts/run_multi_focal_edge_sweep.py \\
        --encoder outputs/diagnostic_v3/diagnostic_v3_encoder.pt \\
        --data-dir data/processed \\
        --out-dir outputs/multi_focal_edge \\
        --symmetric-n-edges 5
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
from gnn_forecast.heterogeneous_model import (
    HeterogeneousEncoder, HeterogeneousEncoderConfig,
)
from gnn_forecast.node_features import load_node_features
from gnn_forecast.edge_intervention import (
    multi_focal_edge_sweep, pairwise_wedges,
)
from gnn_forecast.counterfactual import FOUR_FOCAL_CCODES, MAJOR_POWER_CCODES

SEED = 123


def filter_layer_to_cow(df: pd.DataFrame, cow_set: set) -> pd.DataFrame:
    src_in = [(int(c), int(y)) in cow_set for c, y in zip(df["source_ccode"], df["year"])]
    tgt_in = [(int(c), int(y)) in cow_set for c, y in zip(df["target_ccode"], df["year"])]
    keep = pd.Series(src_in) & pd.Series(tgt_in)
    return df[keep.values].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--cow-membership", default="data/processed/cow_state_membership.csv")
    ap.add_argument("--node-features", default="data/processed/node_features.csv")
    ap.add_argument("--start-year", type=int, default=1948)
    ap.add_argument("--end-year", type=int, default=2016)
    ap.add_argument("--focal-year", type=int, default=None,
                    help="Year at which to perturb (default: end_year).")
    ap.add_argument("--symmetric-n-edges", type=int, default=5)
    ap.add_argument("--out-dir", default="outputs/multi_focal_edge")
    ap.add_argument("--top-k", type=int, default=30,
                    help="Top-K wedges per focal pair to save in top_k_wedges.csv.")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild dataset
    cow = pd.read_csv(args.cow_membership)
    cow["ccode"] = cow["ccode"].astype(int)
    cow["year"] = cow["year"].astype(int)
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
    print(f"Dataset: {len(dataset.years)} years, {dataset.num_nodes} nodes, "
          f"layers={dataset.layer_names}")

    feat_set = load_node_features(
        Path(args.node_features), dataset.ccode_to_idx, dataset.years,
    )

    # Load encoder
    payload = torch.load(args.encoder, map_location="cpu", weights_only=False)
    enc_cfg = HeterogeneousEncoderConfig(**payload["encoder_config"])
    encoder = HeterogeneousEncoder(payload["num_nodes"], enc_cfg)
    encoder.load_state_dict(payload["encoder_state_dict"])
    encoder.eval()
    print(f"Loaded encoder: best_epoch={payload['best_epoch']}, "
          f"best_link_auc={payload['best_link_auc']:.4f}")

    # Filter focals to those present in the dataset
    focal_ccodes = [cc for cc in FOUR_FOCAL_CCODES if cc in dataset.ccode_to_idx]
    missing = [cc for cc in FOUR_FOCAL_CCODES if cc not in dataset.ccode_to_idx]
    if missing:
        print(f"WARNING: focal ccodes missing from dataset (skipped): {missing}")
    print(f"Focal countries: {[MAJOR_POWER_CCODES[cc] for cc in focal_ccodes]}")

    # Run sweep
    print("\n=== Running multi-focal edge sweep ===")
    long_df = multi_focal_edge_sweep(
        encoder=encoder,
        dataset=dataset,
        feat_set=feat_set,
        focal_ccodes=focal_ccodes,
        focal_year=args.focal_year,
        symmetric_n_edges=args.symmetric_n_edges,
    )
    sweep_path = out_dir / "sweep_long.csv"
    long_df.to_csv(sweep_path, index=False)
    print(f"Wrote {sweep_path} ({len(long_df)} rows)")

    # Pairwise wedges (centroid + cosine)
    for metric in ("delta_centroid", "delta_cosine"):
        wedges = pairwise_wedges(long_df, metric=metric)
        suffix = metric.replace("delta_", "")
        wedge_path = out_dir / f"wedges_{suffix}.csv"
        wedges.to_csv(wedge_path, index=False)
        print(f"Wrote {wedge_path} ({len(wedges)} rows)")

    # Top-K wedges per focal pair (centroid metric for headline)
    wedges_centroid = pairwise_wedges(long_df, metric="delta_centroid")
    top_rows = []
    for (a, b), grp in wedges_centroid.groupby(["focal_a_ccode", "focal_b_ccode"]):
        ranked = grp.reindex(grp["wedge"].abs().sort_values(ascending=False).index)
        top_rows.append(ranked.head(args.top_k))
    top_df = pd.concat(top_rows, ignore_index=True)
    top_path = out_dir / "top_k_wedges.csv"
    top_df.to_csv(top_path, index=False)
    print(f"Wrote {top_path} (top {args.top_k} per focal pair)")


if __name__ == "__main__":
    main()
