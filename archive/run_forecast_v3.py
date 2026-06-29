#!/usr/bin/env python3
"""2017-2040 forecast rollout for the four-focal regime-shock scenarios.

Inputs:
  outputs/diagnostic_v3/diagnostic_v3_encoder.pt   (R-GCN encoder)
  outputs/forecast/gru_weights.pt                  (GRU temporal head)
  data/processed/*

Outputs:
  outputs/forecast/baseline_trajectories.csv      (one row per (focal, year))
  outputs/forecast/scenario_trajectories.csv      (long form: scenario, focal, year, focal_centroid_dist)

Usage:
    PYTHONPATH=src python scripts/run_forecast_v3.py \\
        --encoder outputs/diagnostic_v3/diagnostic_v3_encoder.pt \\
        --gru outputs/forecast/gru_weights.pt \\
        --forecast-until 2040 \\
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
from gnn_forecast.feature_intervention import ALL_REGIME_SHOCKS
from gnn_forecast.forecast_v3 import (
    EmbeddingGRU, _build_embedding_history,
    rollout_forecast, trajectory_centroid_distance,
    run_scenario_forecast,
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
    ap.add_argument("--gru", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--cow-membership", default="data/processed/cow_state_membership.csv")
    ap.add_argument("--node-features", default="data/processed/node_features.csv")
    ap.add_argument("--start-year", type=int, default=1948)
    ap.add_argument("--end-year", type=int, default=2016)
    ap.add_argument("--forecast-until", type=int, default=2040)
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
    feat_set = load_node_features(
        Path(args.node_features), dataset.ccode_to_idx, dataset.years,
    )

    # Load encoder
    enc_payload = torch.load(args.encoder, map_location="cpu", weights_only=False)
    enc_cfg = HeterogeneousEncoderConfig(**enc_payload["encoder_config"])
    encoder = HeterogeneousEncoder(enc_payload["num_nodes"], enc_cfg)
    encoder.load_state_dict(enc_payload["encoder_state_dict"])
    encoder.eval()

    # Load GRU
    gru_payload = torch.load(args.gru, map_location="cpu", weights_only=False)
    gru = EmbeddingGRU(
        emb_dim=gru_payload["emb_dim"],
        hidden_dim=gru_payload["hidden_dim"],
    )
    gru.load_state_dict(gru_payload["gru_state_dict"])
    gru.eval()
    seq_len = gru_payload["seq_len"]
    print(f"Loaded encoder + GRU (best_val_mse={gru_payload['best_val_mse']:.5f})")

    # Build observed-year embeddings (baseline)
    base_emb = _build_embedding_history(encoder, dataset, feat_set)
    base_fcst = rollout_forecast(gru, base_emb, args.forecast_until, seq_len)
    base_combined = {**base_emb, **base_fcst}
    print(f"Forecast horizon: {dataset.years[-1] + 1}-{args.forecast_until} "
          f"({len(base_fcst)} years)")

    # Baseline trajectories for the four focals
    baseline_rows = []
    for focal_ccode in FOUR_FOCAL_CCODES:
        if focal_ccode not in dataset.ccode_to_idx:
            continue
        focal_idx = dataset.ccode_to_idx[focal_ccode]
        traj = trajectory_centroid_distance(base_combined, focal_idx)
        traj["focal_ccode"] = focal_ccode
        traj["focal_name"] = MAJOR_POWER_CCODES.get(focal_ccode, str(focal_ccode))
        baseline_rows.append(traj)
    baseline_df = pd.concat(baseline_rows, ignore_index=True)
    base_path = out_dir / "baseline_trajectories.csv"
    baseline_df.to_csv(base_path, index=False)
    print(f"Wrote {base_path}")

    # Per-scenario counterfactual trajectories
    scenario_rows = []
    for scenario in ALL_REGIME_SHOCKS:
        if scenario.focal_ccode not in dataset.ccode_to_idx:
            print(f"  SKIP {scenario.label}: focal not in dataset")
            continue
        print(f"  forecasting {scenario.label} ...")
        result = run_scenario_forecast(
            encoder, gru, dataset, feat_set, scenario,
            forecast_until_year=args.forecast_until, seq_len=seq_len,
        )
        cf = result.trajectory_counterfactual.copy()
        cf["focal_ccode"] = scenario.focal_ccode
        cf["focal_name"] = MAJOR_POWER_CCODES.get(scenario.focal_ccode,
                                                  str(scenario.focal_ccode))
        scenario_rows.append(cf)

    scenario_df = pd.concat(scenario_rows, ignore_index=True)
    scen_path = out_dir / "scenario_trajectories.csv"
    scenario_df.to_csv(scen_path, index=False)
    print(f"\nWrote {scen_path} ({len(scenario_df)} rows across "
          f"{scenario_df['scenario'].nunique()} scenarios)")


if __name__ == "__main__":
    main()
