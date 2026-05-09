#!/usr/bin/env python3
"""Train the GRU temporal head on R-GCN encoder embeddings.

Inputs:
  outputs/diagnostic_v3/diagnostic_v3_encoder.pt   (frozen R-GCN weights)
  data/processed/*

Outputs:
  outputs/forecast/gru_weights.pt
  outputs/forecast/gru_training_history.csv

Usage:
    PYTHONPATH=src python scripts/train_gru_v3.py \\
        --encoder outputs/diagnostic_v3/diagnostic_v3_encoder.pt \\
        --out-dir outputs/forecast
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
from gnn_forecast.forecast_v3 import (
    train_gru, _build_embedding_history,
)

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
    ap.add_argument("--seq-len", type=int, default=5)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-epochs", type=int, default=200)
    ap.add_argument("--out-dir", default="outputs/forecast")
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
    print(f"Dataset: {len(dataset.years)} years, {dataset.num_nodes} nodes")
    feat_set = load_node_features(
        Path(args.node_features), dataset.ccode_to_idx, dataset.years,
    )

    # Load encoder
    payload = torch.load(args.encoder, map_location="cpu", weights_only=False)
    enc_cfg = HeterogeneousEncoderConfig(**payload["encoder_config"])
    encoder = HeterogeneousEncoder(payload["num_nodes"], enc_cfg)
    encoder.load_state_dict(payload["encoder_state_dict"])
    encoder.eval()
    print(f"Loaded encoder: best_link_auc={payload['best_link_auc']:.4f}")

    print("\n=== Building embedding history ===")
    emb_by_year = _build_embedding_history(encoder, dataset, feat_set)
    print(f"  {len(emb_by_year)} year embeddings, shape "
          f"{next(iter(emb_by_year.values())).shape}")

    print("\n=== Training GRU ===")
    gru, history = train_gru(
        emb_by_year,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_epochs=args.num_epochs,
    )

    # Save GRU weights
    gru_state_path = out_dir / "gru_weights.pt"
    torch.save({
        "gru_state_dict": gru.state_dict(),
        "emb_dim": gru.emb_dim,
        "hidden_dim": args.hidden_dim,
        "seq_len": args.seq_len,
        "best_val_mse": history["best_val"][0],
        "best_epoch": history["best_epoch"][0],
    }, gru_state_path)
    print(f"\nSaved GRU weights -> {gru_state_path}")

    # Save training history
    hist_df = pd.DataFrame({
        "epoch": list(range(1, len(history["train_mse"]) + 1)),
        "train_mse": history["train_mse"],
        "val_mse": history["val_mse"],
    })
    hist_path = out_dir / "gru_training_history.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"Saved training history -> {hist_path}")


if __name__ == "__main__":
    main()
