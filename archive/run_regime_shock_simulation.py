#!/usr/bin/env python3
"""Planted-feature-shock SBM validation for the feature-intervention pipeline.

Runs n_replicates each of {planted, null} on a synthetic 3-block, 3-layer
multiplex SBM. The "planted" condition shifts the focal node's feature by
+3 SDs; the "null" condition leaves features unchanged. Recovery
criteria:

  - Planted: monotonic decay in displacement across hop-0 / 1 / 2 / 3+
  - Planted: hop-0 displacement >> null hop-0 displacement (signal>noise)
  - Null: all hops near zero

Output:
  outputs/regime_shock_simulation/recovery_summary.csv
  outputs/regime_shock_simulation/recovery_per_replicate.csv

Usage:
    PYTHONPATH=src python scripts/run_regime_shock_simulation.py \\
        --n-replicates 10 --z-shift 3.0 \\
        --out-dir outputs/regime_shock_simulation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn_forecast.simulation_v3 import run_planted_shock_study

SEED = 123


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-replicates", type=int, default=10)
    ap.add_argument("--z-shift", type=float, default=3.0)
    ap.add_argument("--feature-name", default="f_0")
    ap.add_argument("--num-epochs", type=int, default=50,
                    help="R-GCN training epochs per replicate.")
    ap.add_argument("--out-dir", default="outputs/regime_shock_simulation")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Planted-shock SBM validation: {args.n_replicates} replicates each "
          f"of {{planted, null}}, z_shift = {args.z_shift}")

    df = run_planted_shock_study(
        n_replicates=args.n_replicates,
        z_shift=args.z_shift,
        feature_name=args.feature_name,
        num_epochs=args.num_epochs,
    )

    per_rep_path = out_dir / "recovery_per_replicate.csv"
    df.to_csv(per_rep_path, index=False)
    print(f"\nWrote {per_rep_path}")

    # Aggregate summary by scenario
    summary_rows = []
    for scenario in ("planted", "null"):
        sub = df[df["scenario"] == scenario]
        summary_rows.append({
            "scenario": scenario,
            "n_replicates": len(sub),
            "mean_hop_0_displacement": float(sub["hop_0_displacement"].mean()),
            "mean_hop_1_displacement": float(sub["hop_1_displacement"].mean()),
            "mean_hop_2_displacement": float(sub["hop_2_displacement"].mean()),
            "mean_hop_3plus_displacement": float(sub["hop_3plus_displacement"].mean()),
            "median_hop_0_centroid_delta": float(sub["hop_0_centroid_delta"].median()),
            "frac_monotonic_decay": float(sub["monotonic_decay"].mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "recovery_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print("\n" + summary_df.to_string(index=False))

    # Pass/fail verdict
    planted = summary_df[summary_df["scenario"] == "planted"].iloc[0]
    null = summary_df[summary_df["scenario"] == "null"].iloc[0]
    signal_to_noise = planted["mean_hop_0_displacement"] / max(
        null["mean_hop_0_displacement"], 1e-6,
    )
    print("\n=== VERDICT ===")
    print(f"  planted hop-0 / null hop-0 = {signal_to_noise:.2f}x")
    print(f"  planted monotonic-decay rate: "
          f"{planted['frac_monotonic_decay']*100:.0f}%")
    if signal_to_noise > 3.0 and planted["frac_monotonic_decay"] >= 0.7:
        print("  PASS — feature-intervention machinery propagates as expected.")
    else:
        print("  WEAK — investigate before claiming validation.")


if __name__ == "__main__":
    main()
