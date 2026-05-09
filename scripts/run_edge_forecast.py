#!/usr/bin/env python3
"""Edge-counterfactual forecasts to 2040 for paper #1's §6.

Replaces the regime-shock counterfactual forecasts (which were archived
as paper #2 work) with edge-perturbation forecasts: each scenario
removes (or adds) a key bilateral tie in year 2016 and rolls the GRU
forward to 2040 from the resulting embedding sequence.

Inputs:  outputs/diagnostic_v3/diagnostic_v3_encoder.pt
         outputs/forecast/gru_weights.pt
Output:  outputs/forecast/edge_scenario_trajectories.csv
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
    EmbeddingGRU, EDGE_SCENARIOS, run_edge_scenario_forecast,
)
from gnn_forecast.counterfactual import MAJOR_POWER_CCODES

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

    torch.manual_seed(SEED); np.random.seed(SEED)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cow = pd.read_csv(args.cow_membership)
    cow["ccode"] = cow["ccode"].astype(int); cow["year"] = cow["year"].astype(int)
    cow_set = set(zip(cow["ccode"], cow["year"]))
    layer_dfs = discover_sparse_layers(Path(args.data_dir))
    cow_filtered = {
        name: filter_layer_to_cow(df, cow_set)[
            (filter_layer_to_cow(df, cow_set)["year"] >= args.start_year)
            & (filter_layer_to_cow(df, cow_set)["year"] <= args.end_year)
        ].copy()
        for name, df in layer_dfs.items()
    }
    dataset = build_sparse_dataset(cow_filtered, year_range=(args.start_year, args.end_year))
    feat_set = load_node_features(Path(args.node_features), dataset.ccode_to_idx, dataset.years)

    enc_payload = torch.load(args.encoder, map_location="cpu", weights_only=False)
    enc_cfg = HeterogeneousEncoderConfig(**enc_payload["encoder_config"])
    encoder = HeterogeneousEncoder(enc_payload["num_nodes"], enc_cfg)
    encoder.load_state_dict(enc_payload["encoder_state_dict"]); encoder.eval()

    gru_payload = torch.load(args.gru, map_location="cpu", weights_only=False)
    gru = EmbeddingGRU(emb_dim=gru_payload["emb_dim"], hidden_dim=gru_payload["hidden_dim"])
    gru.load_state_dict(gru_payload["gru_state_dict"]); gru.eval()
    seq_len = gru_payload["seq_len"]
    print(f"Loaded encoder + GRU. Forecast horizon: {dataset.years[-1] + 1}-{args.forecast_until}")

    rows = []
    for scenario in EDGE_SCENARIOS:
        if scenario.focal_ccode not in dataset.ccode_to_idx \
           or scenario.partner_ccode not in dataset.ccode_to_idx \
           or scenario.layer_name not in dataset.layer_names:
            print(f"  SKIP {scenario.label}")
            continue
        print(f"  forecasting {scenario.label} ...")
        result = run_edge_scenario_forecast(
            encoder, gru, dataset, feat_set, scenario,
            forecast_until_year=args.forecast_until, seq_len=seq_len,
        )
        cf = result.trajectory_counterfactual.copy()
        cf["focal_ccode"] = scenario.focal_ccode
        cf["focal_name"] = MAJOR_POWER_CCODES.get(scenario.focal_ccode, str(scenario.focal_ccode))
        rows.append(cf)

    df = pd.concat(rows, ignore_index=True)
    out_path = out_dir / "edge_scenario_trajectories.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows across {df['scenario'].nunique()} scenarios)")


if __name__ == "__main__":
    main()
