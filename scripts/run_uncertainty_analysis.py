#!/usr/bin/env python3
"""Ensemble training + bootstrap CIs on counterfactual edge effects.

Trains an N-member ensemble on the canonical multiplex dataset (seed
123, 124, ...). Computes ensemble-bootstrapped CIs on dual-focal Δ
embeddedness for USA and China. Output is the file PA reviewers will
ask for: a CSV of (partner, layer, op) → (mean Δ, lo, hi, significant).

Usage:
    PYTHONPATH=src python scripts/run_uncertainty_analysis.py \
        --data-dir data/processed \
        --start-year 1945 --end-year 2025 \
        --n-members 10 \
        --out-dir outputs/uncertainty
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn_forecast.multiplex_data import (
    discover_layers,
    build_global_node_index,
    build_multiplex_dataset,
)
from gnn_forecast.multiplex_model import MultiplexGNNConfig
from gnn_forecast.training import TrainingConfig
from gnn_forecast.uncertainty import (
    train_ensemble,
    embeddedness_with_cis,
    bootstrap_counterfactual_cis,
    BASE_SEED,
)
from gnn_forecast.counterfactual import USA_CCODE, CHN_CCODE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="tests/fixtures/tiny_processed")
    ap.add_argument("--nodes-csv", default=None)
    ap.add_argument("--start-year", type=int, default=1945)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--n-members", type=int, default=10)
    ap.add_argument("--n-add-edges", type=int, default=5,
                    help="Symmetric edge count for add/remove perturbations")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="CI level: alpha=0.05 → 95% CI")
    ap.add_argument("--max-partners", type=int, default=50,
                    help="Cap partner count to keep runtime tractable")
    ap.add_argument("--out-dir", default="outputs/uncertainty")
    args = ap.parse_args()

    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading layers from {args.data_dir}")
    layers = discover_layers(Path(args.data_dir))
    nodes_csv = Path(args.nodes_csv) if args.nodes_csv else Path(args.data_dir) / "nodes.csv"
    ccode_to_idx, idx_to_ccode, nodes_df = build_global_node_index(layers, nodes_csv)
    dataset = build_multiplex_dataset(
        layers, ccode_to_idx, idx_to_ccode, nodes_df,
        year_range=(args.start_year, args.end_year),
    )
    print(f"  {len(dataset.years)} years, {dataset.num_nodes} nodes, {len(dataset.layer_names)} layers")

    # Train ensemble
    print(f"\nTraining {args.n_members}-member ensemble (seeds {BASE_SEED}..{BASE_SEED + args.n_members - 1})")
    model_cfg = MultiplexGNNConfig(
        num_layers=len(dataset.layer_names),
        in_dim=len(dataset.layer_names),
        hidden_dim=64, emb_dim=32, seq_len=5,
    )
    train_cfg = TrainingConfig(num_epochs=100, patience=20, print_every=25, seq_len=5)
    ensemble = train_ensemble(
        dataset=dataset, n_members=args.n_members,
        model_config=model_cfg, train_config=train_cfg,
        save_dir=str(out_dir / "ensemble_weights"),
    )

    # Embeddedness CIs at the final year for USA and China
    print("\nComputing embeddedness CIs at final year for USA and China")
    final_year = dataset.years[-1]
    rows = []
    for ccode in (USA_CCODE, CHN_CCODE):
        if ccode in dataset.ccode_to_idx:
            r = embeddedness_with_cis(
                ensemble, dataset, year=final_year, ccode=ccode, alpha=args.alpha,
            )
            rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "embeddedness_ci_final_year.csv", index=False)
    print(f"  Saved {out_dir / 'embeddedness_ci_final_year.csv'}")

    # Bootstrap counterfactual CIs
    print(f"\nBootstrap CIs on dual-focal counterfactuals (n_add_edges={args.n_add_edges})")
    partner_pool = [
        cc for cc in dataset.ccode_to_idx.keys()
        if cc not in (USA_CCODE, CHN_CCODE)
    ][:args.max_partners]
    print(f"  Sweeping {len(partner_pool)} partners")

    cf_df = bootstrap_counterfactual_cis(
        ensemble=ensemble,
        dataset=dataset,
        partner_ccodes=partner_pool,
        n_add_edges=args.n_add_edges,
        alpha=args.alpha,
    )
    cf_df.to_csv(out_dir / "bootstrap_counterfactual_cis.csv", index=False)
    print(f"  Saved {out_dir / 'bootstrap_counterfactual_cis.csv'}")

    # Significance summary
    n_sig = int(cf_df["wedge_significant"].sum())
    print(f"\n{n_sig}/{len(cf_df)} (partner, layer, op) cells have CI excluding zero on the wedge")

    # Top significant wedges
    sig = cf_df[cf_df["wedge_significant"]].copy()
    if not sig.empty:
        sig["abs_wedge"] = sig["wedge_mean"].abs()
        top = sig.sort_values("abs_wedge", ascending=False).head(20)
        top.to_csv(out_dir / "top_significant_wedges.csv", index=False)
        print(f"Saved top significant wedges to {out_dir / 'top_significant_wedges.csv'}")


if __name__ == "__main__":
    main()
