#!/usr/bin/env python3
"""Planted-wedge recovery study (simulation validation).

Runs the synthetic multiplex SBM with planted wedge edges, trains the GNN,
runs the dual-focal counterfactual sweep, and reports recovery rates.

Usage:
    PYTHONPATH=src python scripts/run_simulation_study.py
    PYTHONPATH=src python scripts/run_simulation_study.py --replicates 10 --top-k 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from gnn_forecast.simulation import run_recovery_study

SEED = 123


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--num-nodes", type=int, default=60)
    ap.add_argument("--num-years", type=int, default=30)
    ap.add_argument("--usa-extra-edges", type=int, default=12,
                    help="Strength of USA-side planting (Tune 1)")
    ap.add_argument("--no-rotate-partner", action="store_true",
                    help="Disable Tune 2 (vary partner across replicates)")
    ap.add_argument("--no-null", action="store_true",
                    help="Disable Tune 4 (null calibration scenario)")
    ap.add_argument("--out-dir", default="outputs/simulation_study")
    ap.add_argument("--base-seed", type=int, default=SEED)
    args = ap.parse_args()

    torch.manual_seed(args.base_seed)
    np.random.seed(args.base_seed)

    print(f"Running planted-wedge recovery study with seed={args.base_seed}")
    print(f"  replicates={args.replicates}, top-k={args.top_k}")
    print(f"  num_nodes={args.num_nodes}, num_years={args.num_years}")
    print(f"  usa_extra_edges={args.usa_extra_edges}, "
          f"rotate_partner={not args.no_rotate_partner}, "
          f"include_null={not args.no_null}")

    df = run_recovery_study(
        n_replicates=args.replicates,
        base_seed=args.base_seed,
        top_k=args.top_k,
        num_nodes=args.num_nodes,
        num_years=args.num_years,
        usa_extra_edges=args.usa_extra_edges,
        rotate_partner=not args.no_rotate_partner,
        include_null=not args.no_null,
        save_dir=args.out_dir,
    )

    print("\n=== Per-replicate ranks ===")
    cols = ["replicate", "scenario", "planted_partner_ccode",
            "usa_isolation_rank", "chn_isolation_rank", "wedge_rank"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
