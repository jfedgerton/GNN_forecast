#!/usr/bin/env python3
"""Joint Polity+edge intervention runner for §6.7 / appendix.

Runs the pre-baked JOINT_SCENARIOS (one per focal) and writes a
summary CSV with the interaction-term decomposition.

Output:
  outputs/joint_intervention/joint_intervention_summary.csv

Usage:
    PYTHONPATH=src python scripts/run_joint_intervention.py \\
        --encoder outputs/diagnostic_v3/diagnostic_v3_encoder.pt \\
        --data-dir data/processed \\
        --out-dir outputs/joint_intervention
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
from gnn_forecast.joint_intervention import (
    JOINT_SCENARIOS, run_joint_intervention,
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
    ap.add_argument("--out-dir", default="outputs/joint_intervention")
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

    payload = torch.load(args.encoder, map_location="cpu", weights_only=False)
    enc_cfg = HeterogeneousEncoderConfig(**payload["encoder_config"])
    encoder = HeterogeneousEncoder(payload["num_nodes"], enc_cfg)
    encoder.load_state_dict(payload["encoder_state_dict"])
    encoder.eval()
    print(f"Loaded encoder: best_link_auc={payload['best_link_auc']:.4f}")

    rows = []
    for scenario in JOINT_SCENARIOS:
        if scenario.focal_ccode not in dataset.ccode_to_idx:
            print(f"  SKIP {scenario.label}: focal_ccode "
                  f"{scenario.focal_ccode} not in dataset")
            continue
        if scenario.partner_ccode not in dataset.ccode_to_idx:
            print(f"  SKIP {scenario.label}: partner_ccode "
                  f"{scenario.partner_ccode} not in dataset")
            continue
        # Resolve layer name (handle case where pta might be 'pta_services' or 'pta')
        layer_name_check = scenario.layer_name
        if layer_name_check not in dataset.layer_names:
            # Try common alternates
            alts = {"pta": "pta_services", "pta_services": "pta",
                    "alliance": "defensive_alliances"}
            for alt_in, alt_out in alts.items():
                if scenario.layer_name == alt_in and alt_out in dataset.layer_names:
                    layer_name_check = alt_out
                    break
            else:
                print(f"  SKIP {scenario.label}: layer '{scenario.layer_name}' "
                      f"not in dataset {dataset.layer_names}")
                continue

        # Use possibly-resolved layer name
        from dataclasses import replace
        scenario = replace(scenario, layer_name=layer_name_check)

        print(f"\n=== {scenario.label} ===")
        result = run_joint_intervention(encoder, dataset, feat_set, scenario)
        print(f"  delta_feature_only: {result.delta_feature_only:+.4f}")
        print(f"  delta_edge_only:    {result.delta_edge_only:+.4f}")
        print(f"  delta_joint:        {result.delta_joint:+.4f}")
        print(f"  delta_additive:     {result.delta_additive:+.4f}")
        print(f"  interaction_term:   {result.interaction_term:+.4f} "
              f"({result.interaction_pct:+.1f}% of additive)")

        rows.append({
            "scenario": scenario.label,
            "focal_ccode": scenario.focal_ccode,
            "partner_ccode": scenario.partner_ccode,
            "feature": scenario.feature_name,
            "feature_target_raw": scenario.feature_target_raw,
            "layer": scenario.layer_name,
            "operation": scenario.operation,
            "cascade_year": result.cascade_year,
            "delta_feature_only": result.delta_feature_only,
            "delta_edge_only": result.delta_edge_only,
            "delta_joint": result.delta_joint,
            "delta_additive": result.delta_additive,
            "interaction_term": result.interaction_term,
            "interaction_pct": result.interaction_pct,
        })

    df = pd.DataFrame(rows)
    out_path = out_dir / "joint_intervention_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} scenarios)")


if __name__ == "__main__":
    main()
