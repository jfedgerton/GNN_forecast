#!/usr/bin/env python3
"""Empirical regime-shock counterfactuals for the four-focal set.

Runs the eight pre-baked Polity / CINC shock scenarios from
`feature_intervention.POLITY_SHOCKS + CINC_SHOCKS` against the trained
6-layer R-GCN encoder, writes per-hop cascade summaries plus per-state
long-form embedding-shift tables for downstream plots.

Inputs:
  outputs/diagnostic_v3/diagnostic_v3_encoder.pt   (saved by run_diagnostic_v3.py)
  data/processed/cow_state_membership.csv
  data/processed/node_features.csv
  data/processed/layer_*.csv

Outputs:
  outputs/regime_shock/regime_shock_summary.csv
  outputs/regime_shock/per_state/<scenario_label>.csv

Usage:
    PYTHONPATH=src python scripts/run_regime_shock_empirical.py \\
        --encoder outputs/diagnostic_v3/diagnostic_v3_encoder.pt \\
        --data-dir data/processed \\
        --out-dir outputs/regime_shock
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn_forecast.diagnostic_v3 import (
    discover_sparse_layers, build_sparse_dataset, STRATEGIC_LAYER_PATTERNS,
)
from gnn_forecast.heterogeneous_model import (
    HeterogeneousEncoder, HeterogeneousEncoderConfig,
)
from gnn_forecast.node_features import load_node_features
from gnn_forecast.feature_intervention import (
    ALL_REGIME_SHOCKS, run_feature_intervention,
)

SEED = 123


def filter_layer_to_cow(df: pd.DataFrame, cow_set: set) -> pd.DataFrame:
    """Same dyadic COW filter as run_diagnostic_v3.py."""
    src_in = [(int(c), int(y)) in cow_set for c, y in zip(df["source_ccode"], df["year"])]
    tgt_in = [(int(c), int(y)) in cow_set for c, y in zip(df["target_ccode"], df["year"])]
    keep = pd.Series(src_in) & pd.Series(tgt_in)
    return df[keep.values].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True,
                    help="Path to diagnostic_v3_encoder.pt")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--cow-membership", default="data/processed/cow_state_membership.csv")
    ap.add_argument("--node-features", default="data/processed/node_features.csv")
    ap.add_argument("--start-year", type=int, default=1948)
    ap.add_argument("--end-year", type=int, default=2016)
    ap.add_argument("--out-dir", default="outputs/regime_shock")
    ap.add_argument("--cascade-year", type=int, default=None,
                    help="Year at which to evaluate cascade (default: end of "
                         "perturbation window in each scenario).")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(args.out_dir)
    (out_dir / "per_state").mkdir(parents=True, exist_ok=True)

    # ---- Rebuild the dataset on the same COW universe used for training ----
    cow = pd.read_csv(args.cow_membership)
    cow["ccode"] = cow["ccode"].astype(int)
    cow["year"] = cow["year"].astype(int)
    cow_set = set(zip(cow["ccode"], cow["year"]))

    layer_dfs = discover_sparse_layers(Path(args.data_dir))
    if not layer_dfs:
        raise RuntimeError("No strategic layer files found.")

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
    print(f"Features: {feat_set.num_features} per node "
          f"({feat_set.feature_names})")

    # ---- Load the saved encoder ----
    print(f"\nLoading encoder weights from {args.encoder}")
    payload = torch.load(args.encoder, map_location="cpu", weights_only=False)
    enc_cfg_dict = payload["encoder_config"]
    enc_cfg = HeterogeneousEncoderConfig(**enc_cfg_dict)
    encoder = HeterogeneousEncoder(payload["num_nodes"], enc_cfg)
    encoder.load_state_dict(payload["encoder_state_dict"])
    encoder.eval()
    print(f"  best_epoch={payload['best_epoch']}, "
          f"best_link_auc={payload['best_link_auc']:.4f}, "
          f"layers={payload['layer_names']}")

    # Sanity-check that the saved encoder matches the dataset we just rebuilt
    if payload["num_nodes"] != dataset.num_nodes:
        raise ValueError(
            f"Encoder was trained on {payload['num_nodes']} nodes but "
            f"current dataset has {dataset.num_nodes}. Rebuild the dataset "
            f"with the same COW filter and year range."
        )
    if list(payload["layer_names"]) != list(dataset.layer_names):
        print(f"WARNING: layer order mismatch:\n"
              f"  encoder: {payload['layer_names']}\n"
              f"  dataset: {dataset.layer_names}")

    # ---- Iterate scenarios ----
    summary_rows = []
    for scenario in ALL_REGIME_SHOCKS:
        if scenario.focal_ccode not in dataset.ccode_to_idx:
            print(f"  SKIP {scenario.label}: focal_ccode "
                  f"{scenario.focal_ccode} not in dataset")
            continue
        print(f"\n=== {scenario.label} ===")
        print(f"  focal={scenario.focal_ccode} feature={scenario.feature_name} "
              f"target={scenario.target_value_raw} years={scenario.year_range}")

        result = run_feature_intervention(
            encoder, dataset, feat_set, scenario,
            cascade_year=args.cascade_year,
        )

        print(f"  hop-0 focal displacement:    {result.hop_0_focal_displacement:.4f}")
        print(f"  hop-1 mean ({result.n_hop_1:>3d} states): {result.hop_1_mean_displacement:.4f}")
        print(f"  hop-2 mean ({result.n_hop_2:>3d} states): {result.hop_2_mean_displacement:.4f}")
        print(f"  hop-3+ mean ({result.n_hop_3plus:>3d} states): {result.hop_3plus_mean_displacement:.4f}")
        print(f"  hop-0 centroid delta:        {result.hop_0_centroid_delta:+.4f}")

        summary_rows.append({
            "scenario": scenario.label,
            "focal_ccode": scenario.focal_ccode,
            "feature": scenario.feature_name,
            "target_value_raw": scenario.target_value_raw,
            "year_start": scenario.year_range[0],
            "year_end": scenario.year_range[1],
            "hop_0_displacement": result.hop_0_focal_displacement,
            "hop_1_displacement": result.hop_1_mean_displacement,
            "hop_2_displacement": result.hop_2_mean_displacement,
            "hop_3plus_displacement": result.hop_3plus_mean_displacement,
            "hop_0_centroid_delta": result.hop_0_centroid_delta,
            "hop_1_centroid_delta": result.hop_1_mean_centroid_delta,
            "hop_2_centroid_delta": result.hop_2_mean_centroid_delta,
            "hop_3plus_centroid_delta": result.hop_3plus_mean_centroid_delta,
            "n_hop_1": result.n_hop_1,
            "n_hop_2": result.n_hop_2,
            "n_hop_3plus": result.n_hop_3plus,
        })

        # Per-state long-form table for downstream plots
        per_state_path = out_dir / "per_state" / f"{scenario.label}.csv"
        result.per_state_table.to_csv(per_state_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "regime_shock_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path} ({len(summary_df)} scenarios)")
    print(f"Per-state tables in {out_dir / 'per_state'}/")


if __name__ == "__main__":
    main()
