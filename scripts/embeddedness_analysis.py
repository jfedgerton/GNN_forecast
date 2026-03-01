#!/usr/bin/env python3
"""
Embeddedness analysis for the USA and China.

Computes two outputs:
  1. Embeddedness over time — per-year, per-layer and aggregate embeddedness
     scores for the USA (ccode 2) and China (ccode 710).
  2. Edge-removal impact — for each year, toggles every existing edge for
     each focal country and ranks partners by the embeddedness change their
     removal causes.

Usage (with fixtures):
    PYTHONPATH=src python scripts/embeddedness_analysis.py

Usage (with real data):
    PYTHONPATH=src python scripts/embeddedness_analysis.py \
        --data-dir data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from gnn_forecast.data_layer import CanonicalDataConfig, load_canonical_multiplex_data
from gnn_forecast.interventions import (
    InterventionResult,
    embeddedness_score,
    interventions_to_frame,
    simulate_edge_toggle,
)

# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
ap = argparse.ArgumentParser(description="USA/China embeddedness analysis")
ap.add_argument("--data-dir", default="tests/fixtures/tiny_processed",
                help="Directory with nodes.csv and edges_*.csv")
ap.add_argument("--out-dir", default="outputs/embeddedness_analysis",
                help="Where to write result CSVs")
args = ap.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

USA_CCODE = 2
CHN_CCODE = 710

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
print("Loading multiplex data ...")
config = CanonicalDataConfig(data_dir=args.data_dir)
data = load_canonical_multiplex_data(config)

ccode_to_idx = data.ccode_to_idx
idx_to_ccode = data.idx_to_ccode
num_nodes = len(ccode_to_idx)
years = sorted(data.yearly_edges.keys())
relations = sorted(next(iter(data.yearly_edges.values())).keys())

print(f"  {num_nodes} countries, {len(years)} years, layers: {relations}")

# Verify focal countries exist in node set
for label, cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
    if cc not in ccode_to_idx:
        raise ValueError(f"{label} (ccode {cc}) not found in nodes.csv")


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def build_adjacency(
    year_edges: Dict[str, pd.DataFrame],
    n: int,
    layer: str | None = None,
) -> np.ndarray:
    """Build a symmetric adjacency matrix from edge DataFrames.

    If *layer* is None, aggregate across all layers (max tie value wins).
    """
    adj = np.zeros((n, n), dtype=float)
    targets = {layer: year_edges[layer]} if layer else year_edges
    for df in targets.values():
        for _, row in df.iterrows():
            i, j = int(row["source_idx"]), int(row["target_idx"])
            w = float(row["tie"])
            adj[i, j] = max(adj[i, j], w)
            adj[j, i] = max(adj[j, i], w)
    np.fill_diagonal(adj, 0.0)
    return adj


def adjacency_to_embeddings(adj: np.ndarray) -> torch.Tensor:
    """One-hop mean-aggregation embedding from adjacency (no GNN needed)."""
    a = torch.tensor(adj, dtype=torch.float32)
    deg = a.sum(1, keepdim=True).clamp(min=1.0)
    emb = a / deg  # row-normalised adjacency = structural embedding
    return emb


# ---------------------------------------------------------------
# 2. Embeddedness over time
# ---------------------------------------------------------------
print("\nComputing embeddedness over time ...")

time_rows: List[dict] = []

for year in years:
    year_edges = data.yearly_edges[year]

    # --- Per-layer embeddedness ---
    for rel in relations:
        adj_layer = build_adjacency(year_edges, num_nodes, layer=rel)
        emb_layer = adjacency_to_embeddings(adj_layer)

        for label, cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
            idx = ccode_to_idx[cc]
            score = embeddedness_score(emb_layer, idx)
            degree = int((adj_layer[idx] > 0).sum())
            time_rows.append({
                "year": year,
                "country": label,
                "ccode": cc,
                "layer": rel,
                "embeddedness": round(score, 6),
                "degree": degree,
            })

    # --- Aggregate (all layers combined) ---
    adj_all = build_adjacency(year_edges, num_nodes)
    emb_all = adjacency_to_embeddings(adj_all)

    for label, cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
        idx = ccode_to_idx[cc]
        score = embeddedness_score(emb_all, idx)
        degree = int((adj_all[idx] > 0).sum())
        time_rows.append({
            "year": year,
            "country": label,
            "ccode": cc,
            "layer": "aggregate",
            "embeddedness": round(score, 6),
            "degree": degree,
        })

time_df = pd.DataFrame(time_rows)
time_path = OUT_DIR / "embeddedness_over_time.csv"
time_df.to_csv(time_path, index=False)

print(f"  Saved: {time_path}")
print()
print(time_df.to_string(index=False))

# ---------------------------------------------------------------
# 3. Edge-removal impact analysis
# ---------------------------------------------------------------
print("\n\nComputing edge-removal impacts ...")

removal_rows: List[dict] = []

for year in years:
    year_edges = data.yearly_edges[year]
    adj_all = build_adjacency(year_edges, num_nodes)
    emb_all = adjacency_to_embeddings(adj_all)

    for label, focal_cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
        focal_idx = ccode_to_idx[focal_cc]
        # Only test partners that currently have an edge (removal analysis)
        connected = [
            idx_to_ccode[j]
            for j in range(num_nodes)
            if adj_all[focal_idx, j] > 0 and idx_to_ccode[j] != focal_cc
        ]

        if not connected:
            continue

        results = simulate_edge_toggle(
            adj=adj_all,
            emb=emb_all,
            node_to_idx=ccode_to_idx,
            focal_ccode=focal_cc,
            partner_ccodes=connected,
            add_if_missing=False,  # removals only
        )

        for r in results:
            partner_name = data.nodes.loc[
                data.nodes["ccode"] == r.partner_ccode, "state_name"
            ]
            partner_label = partner_name.values[0] if len(partner_name) else str(r.partner_ccode)
            removal_rows.append({
                "year": year,
                "focal_country": label,
                "focal_ccode": r.focal_ccode,
                "partner": partner_label,
                "partner_ccode": r.partner_ccode,
                "operation": r.operation,
                "embeddedness_delta": round(r.embeddedness_delta, 6),
            })

removal_df = pd.DataFrame(removal_rows)

if removal_df.empty:
    print("  No existing edges to remove (dataset may be too small).")
    # Fall back: also test additions so the output is non-empty
    print("  Running add-edge analysis as fallback ...")
    for year in years:
        year_edges = data.yearly_edges[year]
        adj_all = build_adjacency(year_edges, num_nodes)
        emb_all = adjacency_to_embeddings(adj_all)

        for label, focal_cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
            partners = [cc for cc in ccode_to_idx if cc != focal_cc]
            results = simulate_edge_toggle(
                adj=adj_all,
                emb=emb_all,
                node_to_idx=ccode_to_idx,
                focal_ccode=focal_cc,
                partner_ccodes=partners,
                add_if_missing=True,
            )
            for r in results:
                partner_name = data.nodes.loc[
                    data.nodes["ccode"] == r.partner_ccode, "state_name"
                ]
                partner_label = partner_name.values[0] if len(partner_name) else str(r.partner_ccode)
                removal_rows.append({
                    "year": year,
                    "focal_country": label,
                    "focal_ccode": r.focal_ccode,
                    "partner": partner_label,
                    "partner_ccode": r.partner_ccode,
                    "operation": r.operation,
                    "embeddedness_delta": round(r.embeddedness_delta, 6),
                })

    removal_df = pd.DataFrame(removal_rows)

# Sort: largest magnitude changes first
removal_df = removal_df.sort_values("embeddedness_delta", key=abs, ascending=False)

removal_path = OUT_DIR / "edge_toggle_impacts.csv"
removal_df.to_csv(removal_path, index=False)
print(f"  Saved: {removal_path}")
print()
print(removal_df.to_string(index=False))

# ---------------------------------------------------------------
# 4. Summary tables: top edge changes per focal country
# ---------------------------------------------------------------
print("\n\n" + "=" * 60)
print("SUMMARY: Top edge toggles by |embeddedness delta|")
print("=" * 60)

for label in ["USA", "China"]:
    subset = removal_df[removal_df["focal_country"] == label].head(10)
    if subset.empty:
        print(f"\n  {label}: no interventions available")
        continue
    print(f"\n  {label}:")
    for _, row in subset.iterrows():
        direction = "+" if row["embeddedness_delta"] >= 0 else ""
        print(
            f"    {row['operation']:6s} edge with {row['partner']:>10s} "
            f"(ccode {row['partner_ccode']:>4d})  =>  "
            f"{direction}{row['embeddedness_delta']:.6f}  [year {row['year']}]"
        )

print(f"\nAll outputs in: {OUT_DIR}/")
