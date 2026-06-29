#!/usr/bin/env python3
"""Planted-edge SBM validation for the R-GCN edge-intervention pipeline.

Runs n_replicates each of {planted, null} on a synthetic 3-block, 3-layer
multiplex SBM. The "planted" condition adds k=12 extra alliance ties
between focal A (USA) and a partner from block 2; the "null" condition
adds none. Recovery test: does the planted partner appear at the top of
the USA-vs-China wedge ranking when we sweep all (partner, layer, op)
candidates?

Mirror of the planted-feature-shock SBM but for edge interventions.

Outputs:
  outputs/regime_shock_simulation/edge_recovery_summary.csv
  outputs/regime_shock_simulation/edge_recovery_per_replicate.csv

Usage:
    PYTHONPATH=src python scripts/run_planted_edge_simulation.py \\
        --n-replicates 10 --n-planted-edges 12 \\
        --out-dir outputs/regime_shock_simulation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn_forecast.simulation_v3 import run_planted_edge_study

SEED = 123


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-replicates", type=int, default=10)
    ap.add_argument("--n-planted-edges", type=int, default=12)
    ap.add_argument("--num-epochs", type=int, default=50)
    ap.add_argument("--symmetric-n-edges", type=int, default=5)
    ap.add_argument("--out-dir", default="outputs/regime_shock_simulation")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Planted-edge SBM validation: {args.n_replicates} replicates each "
          f"of {{planted, null}}, n_planted_edges = {args.n_planted_edges}")

    df = run_planted_edge_study(
        n_replicates=args.n_replicates,
        n_planted_edges=args.n_planted_edges,
        num_epochs=args.num_epochs,
        symmetric_n_edges=args.symmetric_n_edges,
    )

    per_rep_path = out_dir / "edge_recovery_per_replicate.csv"
    df.to_csv(per_rep_path, index=False)
    print(f"\nWrote {per_rep_path}")

    summary_rows = []
    for scenario in ("planted", "null"):
        sub = df[df["scenario"] == scenario]
        summary_rows.append({
            "scenario": scenario,
            "n_replicates": len(sub),
            "median_rank_centroid": float(sub["rank_centroid"].median()),
            "median_rank_cosine": float(sub["rank_cosine"].median()),
            "frac_in_top_10_centroid": float(sub["in_top_10_centroid"].mean()),
            "frac_in_top_10_cosine": float(sub["in_top_10_cosine"].mean()),
            "mean_n_candidates": float(sub["n_candidates"].mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "edge_recovery_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}\n")
    print(summary_df.to_string(index=False))

    planted = summary_df[summary_df["scenario"] == "planted"].iloc[0]
    null = summary_df[summary_df["scenario"] == "null"].iloc[0]
    print("\n=== VERDICT ===")
    print(f"  planted top-10 rate: {planted['frac_in_top_10_centroid']*100:.0f}%")
    print(f"  null top-10 rate:    {null['frac_in_top_10_centroid']*100:.0f}%")
    print(f"  planted median rank: {planted['median_rank_centroid']:.0f} of "
          f"{planted['mean_n_candidates']:.0f}")
    print(f"  null median rank:    {null['median_rank_centroid']:.0f}")
    if planted["frac_in_top_10_centroid"] >= 0.7 \
       and null["frac_in_top_10_centroid"] <= 0.2:
        print("  PASS — edge-intervention machinery recovers planted ties.")
    else:
        print("  WEAK — investigate before claiming validation.")


if __name__ == "__main__":
    main()
