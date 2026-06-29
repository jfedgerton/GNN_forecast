#!/usr/bin/env python3
"""Compare AME latent positions to R-GCN embeddings.

For each (layer, year) cell where AME has been fit, compute per-year R²
between the GNN's per-node embedding (projected to 2D via the first two
principal components of the system-wide embedding) and AME's 2-D latent
position. Reports a single table summarizing the alignment.

The substantive read: high R² means the GNN encoder recovers the same
structural variation AME does. Low R² means the GNN captures something
AME doesn't — which is the headline contribution of this paper.

Inputs:
  outputs/diagnostic_v3/diagnostic_v3_encoder.pt
  outputs/ame_baseline/<layer>/ame_latent_<year>.csv (one per cell)

Output:
  outputs/ame_baseline/ame_vs_gnn_alignment.csv
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

SEED = 123


def filter_layer_to_cow(df: pd.DataFrame, cow_set: set) -> pd.DataFrame:
    src_in = [(int(c), int(y)) in cow_set for c, y in zip(df["source_ccode"], df["year"])]
    tgt_in = [(int(c), int(y)) in cow_set for c, y in zip(df["target_ccode"], df["year"])]
    keep = pd.Series(src_in) & pd.Series(tgt_in)
    return df[keep.values].reset_index(drop=True)


def pca_2d(emb: np.ndarray) -> np.ndarray:
    """Project [N, D] embeddings to 2D via PCA on the centered matrix."""
    centered = emb - emb.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt.T[:, :2]


def per_year_r2(gnn_2d: np.ndarray, ame_2d: np.ndarray) -> float:
    """Per-dimension R² aggregated as the mean across the 2 dimensions.
    Both inputs are [N, 2]."""
    if gnn_2d.shape[0] == 0 or ame_2d.shape[0] == 0:
        return float("nan")
    rs = []
    for d in range(2):
        y = ame_2d[:, d]
        x = gnn_2d[:, d]
        if y.std() < 1e-9 or x.std() < 1e-9:
            continue
        # Fit y = a + b * x
        beta = np.cov(x, y, ddof=0)[0, 1] / x.var()
        alpha = y.mean() - beta * x.mean()
        y_pred = alpha + beta * x
        ss_res = float(((y - y_pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        if ss_tot < 1e-9:
            continue
        rs.append(1.0 - ss_res / ss_tot)
    return float(np.mean(rs)) if rs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--ame-dir", default="outputs/ame_baseline")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--cow-membership", default="data/processed/cow_state_membership.csv")
    ap.add_argument("--node-features", default="data/processed/node_features.csv")
    ap.add_argument("--start-year", type=int, default=1948)
    ap.add_argument("--end-year", type=int, default=2016)
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    ame_dir = Path(args.ame_dir)

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

    payload = torch.load(args.encoder, map_location="cpu", weights_only=False)
    cfg = HeterogeneousEncoderConfig(**payload["encoder_config"])
    encoder = HeterogeneousEncoder(payload["num_nodes"], cfg)
    encoder.load_state_dict(payload["encoder_state_dict"])
    encoder.eval()

    rows = []
    for layer_dir in sorted(p for p in ame_dir.iterdir() if p.is_dir()):
        layer_name = layer_dir.name
        for ame_csv in sorted(layer_dir.glob("ame_latent_*.csv")):
            year = int(ame_csv.stem.split("_")[-1])
            if year not in dataset.snapshots:
                continue
            ame_df = pd.read_csv(ame_csv)
            ame_df["ccode"] = ame_df["ccode"].astype(int)

            # Encode the GNN at that year
            snap = dataset.snapshots[year]
            nf = feat_set.by_year[year]
            ei = {k: v for k, v in snap.edge_indices.items()}
            ew = {k: v for k, v in snap.edge_weights.items()}
            with torch.no_grad():
                emb = encoder(nf, ei, ew, snap.layer_mask).numpy()
            gnn_2d = pca_2d(emb)

            # Align AME and GNN by ccode
            ame_df["node_idx"] = ame_df["ccode"].map(dataset.ccode_to_idx)
            ame_df = ame_df.dropna(subset=["node_idx"])
            ame_df["node_idx"] = ame_df["node_idx"].astype(int)
            gnn_aligned = gnn_2d[ame_df["node_idx"].values]
            ame_aligned = ame_df[["latent_dim_1", "latent_dim_2"]].values

            r2 = per_year_r2(gnn_aligned, ame_aligned)
            rows.append({
                "layer": layer_name, "year": year,
                "n_aligned_states": len(ame_df),
                "ame_vs_gnn_r2_mean": r2,
            })
            print(f"  {layer_name} year {year}: R²={r2:.3f} ({len(ame_df)} states)")

    df = pd.DataFrame(rows)
    out_path = ame_dir / "ame_vs_gnn_alignment.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    if not df.empty:
        print("\nMean R² by layer:")
        print(df.groupby("layer")["ame_vs_gnn_r2_mean"].mean().to_string())


if __name__ == "__main__":
    main()
