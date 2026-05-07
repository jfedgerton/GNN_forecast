#!/usr/bin/env python3
"""Diagnostic v2: tests whether rich features + R-GCN backbone fix the encoder.

Two head-to-head experiments:

  v2_baseline_old_features: original encoder + degree-only features.
                            (For reference; should reproduce diagnostic v1's
                            chance-level AUC.)

  v2_rgcn_rich_features:    new R-GCN encoder + rich COW node features
                            + InfoNCE pretraining. The hypothesis test.

If v2_rgcn_rich_features hits link AUC > 0.7 while v2_baseline stays at chance,
the architecture overhaul is justified and we proceed to the full two-stage
pipeline (pretrain → freeze → GRU).

Usage:
    Rscript scripts/export_country_year_features.R   # one-time, builds the CSV
    PYTHONPATH=src python scripts/run_diagnostic_v2.py \\
        --data-dir data/processed \\
        --node-features data/processed/node_features.csv \\
        --start-year 1945 --end-year 2025 \\
        --out-dir outputs/diagnostic_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from gnn_forecast.multiplex_data import (
    discover_layers,
    build_global_node_index,
    build_multiplex_dataset,
)
from gnn_forecast.node_features import load_node_features
from gnn_forecast.heterogeneous_model import (
    HeterogeneousEncoder,
    HeterogeneousEncoderConfig,
)
from gnn_forecast.training_v2 import pretrain_encoder, PretrainConfig, _approx_link_auc
from gnn_forecast.training import _merge_edge_indices
from gnn_forecast.diagnostic import exp1_static_gcn_bce  # the v1 baseline

SEED = 123


def run_v2_rich_features_rgcn(
    dataset,
    node_features_by_year: Dict[int, torch.Tensor],
    raw_feat_dim: int,
    num_epochs: int = 200,
    seed: int = SEED,
    device=None,
) -> dict:
    """v2 hypothesis test: R-GCN encoder + rich features + InfoNCE pretraining."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    enc_cfg = HeterogeneousEncoderConfig(
        relation_names=list(dataset.layer_names),
        raw_feat_dim=raw_feat_dim,
        identity_dim=16,
        hidden_dim=64,
        emb_dim=32,
        dropout=0.2,
    )
    encoder = HeterogeneousEncoder(dataset.num_nodes, enc_cfg).to(device)

    pretrain_cfg = PretrainConfig(
        num_epochs=num_epochs,
        learning_rate=1e-3,
        num_neg_per_pos=10,
        temperature=0.5,
        print_every=25,
        multi_year=True,
    )
    result = pretrain_encoder(
        encoder, dataset, node_features_by_year,
        config=pretrain_cfg, seed=seed, device=device,
    )

    # Score AUC on a single test year (most recent with all layers)
    final_year = dataset.years[-1]
    snap = dataset.snapshots[final_year]
    nf = node_features_by_year[final_year].to(device)
    ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
    ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
    encoder.eval()
    with torch.no_grad():
        emb = encoder(nf, ei, ew, snap.layer_mask)
    merged_ei = _merge_edge_indices(ei, snap.layer_mask)
    auc = _approx_link_auc(emb, merged_ei, dataset.num_nodes)
    mean_norm = float(emb.norm(dim=1).mean().cpu())

    return {
        "name": "v2_rgcn_rich_features",
        "year": int(final_year),
        "n_edges": int(merged_ei.size(1)),
        "raw_feat_dim": raw_feat_dim,
        "final_link_auc": auc,
        "final_mean_norm": mean_norm,
        "n_epochs_run": num_epochs,
        "loss_history": result.loss_history,
        "auc_history": result.auc_history,
        "notes": "If link_auc > 0.7 here, the architecture overhaul is justified.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="tests/fixtures/tiny_processed")
    ap.add_argument("--nodes-csv", default=None)
    ap.add_argument("--node-features", required=True,
                    help="Path to country-year covariate CSV (build with export_country_year_features.R)")
    ap.add_argument("--start-year", type=int, default=1945)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--num-epochs", type=int, default=200)
    ap.add_argument("--out-dir", default="outputs/diagnostic_v2")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load layer data
    print(f"Loading layers from {args.data_dir}")
    layers = discover_layers(Path(args.data_dir))
    nodes_csv = Path(args.nodes_csv) if args.nodes_csv else Path(args.data_dir) / "nodes.csv"
    ccode_to_idx, idx_to_ccode, nodes_df = build_global_node_index(layers, nodes_csv)
    dataset = build_multiplex_dataset(
        layers, ccode_to_idx, idx_to_ccode, nodes_df,
        year_range=(args.start_year, args.end_year),
    )
    print(f"  {len(dataset.years)} years, {dataset.num_nodes} nodes, "
          f"{len(dataset.layer_names)} layers")

    # Load node features
    print(f"\nLoading node features from {args.node_features}")
    feat_set = load_node_features(
        Path(args.node_features), ccode_to_idx, dataset.years,
    )
    print(f"  {feat_set.num_features} features per node")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Baseline (v1 reference) ===
    print("\n" + "=" * 60)
    print("BASELINE: original encoder + degree-only features (v1 reference)")
    print("=" * 60)
    baseline = exp1_static_gcn_bce(dataset, num_epochs=args.num_epochs)
    baseline["name"] = "v2_baseline_old_features"

    # === Hypothesis test ===
    print("\n" + "=" * 60)
    print("HYPOTHESIS TEST: R-GCN + rich features + InfoNCE pretraining")
    print("=" * 60)
    rgcn = run_v2_rich_features_rgcn(
        dataset, feat_set.by_year, feat_set.num_features,
        num_epochs=args.num_epochs,
    )

    # Summary
    rows = [
        {
            "name": baseline["name"],
            "final_link_auc": baseline.get("final_link_auc"),
            "final_mean_norm": baseline.get("final_mean_norm"),
            "n_epochs_run": baseline.get("n_epochs_run"),
        },
        {
            "name": rgcn["name"],
            "final_link_auc": rgcn["final_link_auc"],
            "final_mean_norm": rgcn["final_mean_norm"],
            "n_epochs_run": rgcn["n_epochs_run"],
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "diagnostic_v2_summary.csv", index=False)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    torch.save(
        {"baseline": baseline, "rgcn": rgcn,
         "feature_names": feat_set.feature_names},
        out_dir / "diagnostic_v2_results.pt",
    )
    print(f"\nSaved to {out_dir}")

    # Verdict
    bauc = baseline.get("final_link_auc", 0.5) or 0.5
    rauc = rgcn["final_link_auc"]
    print(f"\nVerdict: baseline AUC={bauc:.3f}, R-GCN+rich AUC={rauc:.3f}")
    if rauc > 0.7 and bauc < 0.6:
        print("  ✓ Architecture overhaul justified. Proceed to two-stage pipeline.")
    elif rauc > bauc + 0.1:
        print("  ~ Improvement but not decisive. Worth tuning further.")
    else:
        print("  ✗ Rich features did not break the chance baseline. Consider AME.")


if __name__ == "__main__":
    main()
