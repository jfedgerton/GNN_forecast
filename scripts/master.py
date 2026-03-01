#!/usr/bin/env python3
"""
Master pipeline script for GNN Forecast.

Runs the full analysis end-to-end: data loading, network construction,
embeddedness analysis, single-edge removal ranking, random multi-edge
deletion simulations, and (optionally) AR forecasting with interventions.

Includes an optional ``--summary`` flag that prints a detailed international
system summary describing network structure across all years and layers.

Usage (fixtures):
    PYTHONPATH=src python scripts/master.py --summary

Usage (full data):
    PYTHONPATH=src python scripts/master.py --data-dir data/processed --summary
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from gnn_forecast.data_layer import CanonicalDataConfig, load_canonical_multiplex_data
from gnn_forecast.forecast import EmbeddingAR, rollout_embeddings
from gnn_forecast.interventions import (
    embeddedness_score,
    interventions_to_frame,
    simulate_edge_toggle,
    simulate_random_multi_edge_deletion,
)

# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
ap = argparse.ArgumentParser(
    description="Master pipeline: full GNN forecast analysis",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
ap.add_argument(
    "--data-dir", default="tests/fixtures/tiny_processed",
    help="Directory with nodes.csv and edges_*.csv (default: test fixtures)",
)
ap.add_argument(
    "--out-dir", default="outputs/master",
    help="Output directory for all results",
)
ap.add_argument(
    "--summary", action="store_true",
    help="Print a detailed international system summary (network stats per year and layer)",
)
ap.add_argument(
    "--skip-forecast", action="store_true",
    help="Skip the AR forecasting stage (useful when only embeddedness analysis is needed)",
)
ap.add_argument(
    "--n-sims", type=int, default=100,
    help="Number of random-deletion simulations per k value (default: 100)",
)
ap.add_argument(
    "--k-min", type=int, default=2,
    help="Minimum number of edges to delete in random simulations (default: 2)",
)
ap.add_argument(
    "--k-max", type=int, default=10,
    help="Maximum number of edges to delete in random simulations (default: 10)",
)
ap.add_argument(
    "--emb-dim", type=int, default=8,
    help="Embedding dimension for AR forecaster (default: 8)",
)
ap.add_argument(
    "--hidden-dim", type=int, default=16,
    help="Hidden dimension for AR GRU forecaster (default: 16)",
)
ap.add_argument(
    "--forecast-steps", type=int, default=25,
    help="Number of forecast steps (default: 25, for 2026-2050)",
)
ap.add_argument(
    "--train-epochs", type=int, default=50,
    help="Training epochs for AR forecaster (default: 50)",
)
ap.add_argument(
    "--seed", type=int, default=42,
    help="Random seed (default: 42)",
)

args = ap.parse_args()
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

USA_CCODE = 2
CHN_CCODE = 710
K_VALUES = range(args.k_min, args.k_max + 1)

t0 = time.time()


# ---------------------------------------------------------------
# Flag-check infrastructure
# ---------------------------------------------------------------
_flag_results: List[dict] = []


def flag_check(code: str, condition: bool, description: str) -> bool:
    """Record a pass/fail flag check. Returns the condition value."""
    status = "PASS" if condition else "FAIL"
    _flag_results.append({"code": code, "status": status, "description": description})
    if not condition:
        print(f"  FLAG {code} FAIL: {description}")
    return condition


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def build_adjacency(
    year_edges: Dict[str, pd.DataFrame],
    n: int,
    layer: str | None = None,
) -> np.ndarray:
    """Build a symmetric adjacency matrix from edge DataFrames."""
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
    """One-hop mean-aggregation embedding from adjacency."""
    a = torch.tensor(adj, dtype=torch.float32)
    deg = a.sum(1, keepdim=True).clamp(min=1.0)
    return a / deg


def ccode_to_name(nodes_df: pd.DataFrame, ccode: int) -> str:
    """Resolve a ccode to its state_name, falling back to str(ccode)."""
    if "state_name" not in nodes_df.columns:
        return str(ccode)
    match = nodes_df.loc[nodes_df["ccode"] == ccode, "state_name"]
    return match.values[0] if len(match) else str(ccode)


def banner(stage: int, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Stage {stage}: {title}")
    print("=" * 60)


# ===============================================================
# Stage 1 — Load data
# ===============================================================
banner(1, "Load multiplex data")

config = CanonicalDataConfig(data_dir=args.data_dir)
data = load_canonical_multiplex_data(config)

ccode_to_idx = data.ccode_to_idx
idx_to_ccode = data.idx_to_ccode
num_nodes = len(ccode_to_idx)
years = sorted(data.yearly_edges.keys())
relations = sorted(next(iter(data.yearly_edges.values())).keys())

print(f"  Countries:  {num_nodes}")
print(f"  Years:      {years[0]}–{years[-1]} ({len(years)} years)")
print(f"  Layers:     {', '.join(relations)}")

for label, cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
    if cc not in ccode_to_idx:
        print(f"  WARNING: {label} (ccode {cc}) not in nodes.csv — skipping interventions for {label}")

# --- Stage 1 flag checks ---
flag_check("FC-01", num_nodes > 0, f"num_nodes={num_nodes} > 0")
flag_check(
    "FC-02",
    len(ccode_to_idx) == len(idx_to_ccode) and set(ccode_to_idx.values()) == set(idx_to_ccode.keys()),
    "ccode_to_idx is bijective (1-to-1 mapping)",
)
for _y in years:
    flag_check(
        "FC-03",
        set(data.yearly_edges[_y].keys()) == set(relations),
        f"Year {_y} has all relation keys: {relations}",
    )
    for _rel, _df in data.yearly_edges[_y].items():
        _required = {"source_ccode", "target_ccode", "tie", "source_idx", "target_idx"}
        flag_check(
            "FC-04",
            _required.issubset(_df.columns),
            f"Year {_y} {_rel} has columns {_required}",
        )


# ===============================================================
# Stage 2 — International system summary (optional)
# ===============================================================
if args.summary:
    banner(2, "International system summary")

    summary_rows: List[dict] = []

    for year in years:
        year_edges = data.yearly_edges[year]

        for rel in relations:
            adj_layer = build_adjacency(year_edges, num_nodes, layer=rel)
            degrees = (adj_layer > 0).sum(axis=1)
            n_edges = int((adj_layer > 0).sum()) // 2  # undirected
            weights = adj_layer[adj_layer > 0]

            summary_rows.append({
                "year": year,
                "layer": rel,
                "n_nodes": num_nodes,
                "n_active_nodes": int((degrees > 0).sum()),
                "n_edges": n_edges,
                "density": round(n_edges / max(num_nodes * (num_nodes - 1) / 2, 1), 6),
                "mean_degree": round(float(degrees.mean()), 4),
                "median_degree": round(float(np.median(degrees)), 4),
                "max_degree": int(degrees.max()),
                "isolates": int((degrees == 0).sum()),
                "mean_weight": round(float(weights.mean()), 6) if len(weights) else 0.0,
                "max_weight": round(float(weights.max()), 6) if len(weights) else 0.0,
            })

        # Aggregate across all layers
        adj_all = build_adjacency(year_edges, num_nodes)
        degrees = (adj_all > 0).sum(axis=1)
        n_edges = int((adj_all > 0).sum()) // 2
        weights = adj_all[adj_all > 0]

        summary_rows.append({
            "year": year,
            "layer": "aggregate",
            "n_nodes": num_nodes,
            "n_active_nodes": int((degrees > 0).sum()),
            "n_edges": n_edges,
            "density": round(n_edges / max(num_nodes * (num_nodes - 1) / 2, 1), 6),
            "mean_degree": round(float(degrees.mean()), 4),
            "median_degree": round(float(np.median(degrees)), 4),
            "max_degree": int(degrees.max()),
            "isolates": int((degrees == 0).sum()),
            "mean_weight": round(float(weights.mean()), 6) if len(weights) else 0.0,
            "max_weight": round(float(weights.max()), 6) if len(weights) else 0.0,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "system_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")
    print()

    # Console overview: one block per layer
    for layer_name in relations + ["aggregate"]:
        layer_data = summary_df[summary_df["layer"] == layer_name]
        print(f"  --- {layer_name.upper()} ---")
        print(f"    Years with data:   {len(layer_data)}")
        print(f"    Edges (min–max):   {layer_data['n_edges'].min()}–{layer_data['n_edges'].max()}")
        print(f"    Density (min–max): {layer_data['density'].min():.4f}–{layer_data['density'].max():.4f}")
        print(f"    Mean degree range: {layer_data['mean_degree'].min():.2f}–{layer_data['mean_degree'].max():.2f}")
        print(f"    Isolates range:    {layer_data['isolates'].min()}–{layer_data['isolates'].max()}")
        print()

    # Focal country profiles
    for label, cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
        if cc not in ccode_to_idx:
            continue
        idx = ccode_to_idx[cc]
        print(f"  --- {label} (ccode {cc}) degree profile ---")
        for year in years:
            year_edges = data.yearly_edges[year]
            adj_all = build_adjacency(year_edges, num_nodes)
            degree = int((adj_all[idx] > 0).sum())
            layer_degrees = []
            for rel in relations:
                adj_l = build_adjacency(year_edges, num_nodes, layer=rel)
                layer_degrees.append(f"{rel}={int((adj_l[idx] > 0).sum())}")
            print(f"    {year}: total={degree}  ({', '.join(layer_degrees)})")
        print()

else:
    print("\n  (Skipping system summary — use --summary to enable)")


# ===============================================================
# Stage 3 — Embeddedness over time
# ===============================================================
banner(3, "Embeddedness over time")

time_rows: List[dict] = []

for year in years:
    year_edges = data.yearly_edges[year]

    for rel in relations:
        adj_layer = build_adjacency(year_edges, num_nodes, layer=rel)
        emb_layer = adjacency_to_embeddings(adj_layer)

        for label, cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
            if cc not in ccode_to_idx:
                continue
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

    adj_all = build_adjacency(year_edges, num_nodes)
    emb_all = adjacency_to_embeddings(adj_all)

    for label, cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
        if cc not in ccode_to_idx:
            continue
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
print(f"  Saved: {time_path}  ({len(time_df)} rows)")

# --- Stage 3 flag checks (run on last year's aggregate) ---
_last_year_edges = data.yearly_edges[years[-1]]
_adj_check = build_adjacency(_last_year_edges, num_nodes)
_emb_check = adjacency_to_embeddings(_adj_check)
flag_check("FC-05", _adj_check.shape == (num_nodes, num_nodes),
           f"Adjacency shape {_adj_check.shape} == ({num_nodes}, {num_nodes})")
flag_check("FC-06", np.allclose(_adj_check, _adj_check.T),
           "Adjacency is symmetric")
flag_check("FC-07", np.allclose(np.diag(_adj_check), 0.0),
           "Adjacency diagonal is zero")
flag_check("FC-08", np.isfinite(_adj_check).all(),
           "Adjacency has no NaN/Inf")
flag_check("FC-09", torch.isfinite(_emb_check).all().item(),
           "Embedding tensor has no NaN/Inf")

for _label, _cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
    if _cc not in ccode_to_idx:
        continue
    _idx = ccode_to_idx[_cc]
    _score = embeddedness_score(_emb_check, _idx)
    _degree_flag = int((_adj_check[_idx] > 0).sum())
    flag_check("FC-10", -1.0 <= _score <= 1.0,
               f"{_label} embeddedness={_score:.6f} in [-1, 1]")
    # Cross-check degree from adjacency vs edges
    _edge_count = sum(
        1 for j in range(num_nodes)
        if _adj_check[_idx, j] > 0 and j != _idx
    )
    flag_check("FC-11", _degree_flag == _edge_count,
               f"{_label} degree from adj row ({_degree_flag}) == edge count ({_edge_count})")


# ===============================================================
# Stage 4 — Single-edge removal impact
# ===============================================================
banner(4, "Single-edge removal impact")

removal_rows: List[dict] = []

for year in years:
    year_edges = data.yearly_edges[year]
    adj_all = build_adjacency(year_edges, num_nodes)
    emb_all = adjacency_to_embeddings(adj_all)

    for label, focal_cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
        if focal_cc not in ccode_to_idx:
            continue
        focal_idx = ccode_to_idx[focal_cc]
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
            add_if_missing=False,
        )

        for r in results:
            removal_rows.append({
                "year": year,
                "focal_country": label,
                "focal_ccode": r.focal_ccode,
                "partner": ccode_to_name(data.nodes, r.partner_ccode),
                "partner_ccode": r.partner_ccode,
                "operation": r.operation,
                "embeddedness_delta": round(r.embeddedness_delta, 6),
            })

removal_df = pd.DataFrame(removal_rows)

if removal_df.empty:
    print("  No existing edges to remove (dataset may be too small).")
    # Fallback: test additions
    for year in years:
        year_edges = data.yearly_edges[year]
        adj_all = build_adjacency(year_edges, num_nodes)
        emb_all = adjacency_to_embeddings(adj_all)
        for label, focal_cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
            if focal_cc not in ccode_to_idx:
                continue
            partners = [cc for cc in ccode_to_idx if cc != focal_cc]
            results = simulate_edge_toggle(
                adj=adj_all, emb=emb_all, node_to_idx=ccode_to_idx,
                focal_ccode=focal_cc, partner_ccodes=partners, add_if_missing=True,
            )
            for r in results:
                removal_rows.append({
                    "year": year, "focal_country": label,
                    "focal_ccode": r.focal_ccode,
                    "partner": ccode_to_name(data.nodes, r.partner_ccode),
                    "partner_ccode": r.partner_ccode, "operation": r.operation,
                    "embeddedness_delta": round(r.embeddedness_delta, 6),
                })
    removal_df = pd.DataFrame(removal_rows)

removal_df = removal_df.sort_values("embeddedness_delta", key=abs, ascending=False)
removal_path = OUT_DIR / "edge_toggle_impacts.csv"
removal_df.to_csv(removal_path, index=False)
print(f"  Saved: {removal_path}  ({len(removal_df)} rows)")

# --- Stage 4 flag checks ---
if not removal_df.empty:
    flag_check("FC-12",
               removal_df["embeddedness_delta"].apply(np.isfinite).all(),
               "All edge toggle deltas are finite")
    _removals = removal_df[removal_df["operation"] == "remove"]
    if not _removals.empty:
        _pct_negative = (_removals["embeddedness_delta"] <= 0).mean()
        flag_check("FC-13", _pct_negative >= 0.5,
                   f"Removal deltas: {_pct_negative*100:.0f}% are <= 0 "
                   f"(expect majority negative — edge removal should hurt embeddedness)")


# ===============================================================
# Stage 5 — Random multi-edge deletion simulations
# ===============================================================
banner(5, f"Random multi-edge deletion (k={args.k_min}..{args.k_max}, {args.n_sims} sims)")

random_deletion_rows: List[dict] = []

for year in years:
    year_edges = data.yearly_edges[year]
    adj_all = build_adjacency(year_edges, num_nodes)
    emb_all = adjacency_to_embeddings(adj_all)

    for label, focal_cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
        if focal_cc not in ccode_to_idx:
            continue
        focal_idx = ccode_to_idx[focal_cc]
        degree = int((adj_all[focal_idx] > 0).sum())

        if degree < args.k_min:
            print(f"  {label} year {year}: degree={degree}, skipping (need >= {args.k_min})")
            continue

        results = simulate_random_multi_edge_deletion(
            adj=adj_all,
            emb=emb_all,
            focal_idx=focal_idx,
            focal_ccode=focal_cc,
            idx_to_ccode=idx_to_ccode,
            k_values=K_VALUES,
            n_sims=args.n_sims,
            rng_seed=args.seed,
        )

        for r in results:
            partner_names = [ccode_to_name(data.nodes, pc) for pc in r.deleted_partner_ccodes]

            random_deletion_rows.append({
                "year": year,
                "focal_country": label,
                "focal_ccode": r.focal_ccode,
                "k": r.k,
                "sim_id": r.sim_id,
                "deleted_partners": "; ".join(partner_names),
                "deleted_ccodes": "; ".join(str(c) for c in r.deleted_partner_ccodes),
                "embeddedness_baseline": round(r.embeddedness_baseline, 6),
                "embeddedness_after": round(r.embeddedness_after, 6),
                "embeddedness_delta": round(r.embeddedness_delta, 6),
            })

random_df = pd.DataFrame(random_deletion_rows)

if random_df.empty:
    print("  No simulations possible (focal nodes have degree < k_min).")
else:
    random_path = OUT_DIR / "random_deletion_impacts.csv"
    random_df.to_csv(random_path, index=False)
    print(f"  Saved: {random_path}  ({len(random_df)} rows)")

    rand_summary = (
        random_df.groupby(["focal_country", "year", "k"])["embeddedness_delta"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    rand_summary.columns = [
        "country", "year", "k", "mean_delta", "std_delta",
        "min_delta", "max_delta", "n_sims",
    ]
    rand_summary_path = OUT_DIR / "random_deletion_summary.csv"
    rand_summary.to_csv(rand_summary_path, index=False)
    print(f"  Saved: {rand_summary_path}  ({len(rand_summary)} rows)")

    # --- Stage 5 flag checks ---
    # Verify baselines are consistent within each (country, year) group
    for (_fc, _yr), _grp in random_df.groupby(["focal_country", "year"]):
        _baselines = _grp["embeddedness_baseline"].unique()
        flag_check("FC-14", len(_baselines) == 1,
                   f"Random deletion baselines for {_fc} year {_yr}: "
                   f"{len(_baselines)} unique value(s) (expect 1)")
    # Verify k matches actual number of deleted partners
    for _row_idx, _row in random_df.iterrows():
        _actual_k = len(str(_row["deleted_ccodes"]).split("; "))
        flag_check("FC-15", _row["k"] == _actual_k,
                   f"Row {_row_idx}: k={_row['k']} matches "
                   f"deleted count={_actual_k}")
        break  # Spot-check first row only to avoid flooding output


# ===============================================================
# Stage 6 — AR forecast and edge-toggle interventions (optional)
# ===============================================================
if not args.skip_forecast:
    banner(6, "AR embedding forecast and interventions")

    torch.manual_seed(args.seed)

    # Build embedding history from adjacency snapshots
    # Each year's row-normalised adjacency serves as a structural embedding
    emb_dim = min(args.emb_dim, num_nodes)
    history_list = []
    for year in years:
        year_edges = data.yearly_edges[year]
        adj_all = build_adjacency(year_edges, num_nodes)
        emb_snap = adjacency_to_embeddings(adj_all)
        # Project or pad to target emb_dim
        if emb_snap.shape[1] > emb_dim:
            emb_snap = emb_snap[:, :emb_dim]
        elif emb_snap.shape[1] < emb_dim:
            pad = torch.zeros(emb_snap.shape[0], emb_dim - emb_snap.shape[1])
            emb_snap = torch.cat([emb_snap, pad], dim=1)
        history_list.append(emb_snap)

    history = torch.stack(history_list, dim=1)  # [num_nodes, n_years, emb_dim]
    print(f"  Embedding history shape: {tuple(history.shape)}")

    # Ensure enough sequence length
    seq_len = history.shape[1]
    if seq_len < 2:
        print("  WARNING: Only 1 year of data — cannot train AR model. Skipping forecast.")
    else:
        model = EmbeddingAR(emb_dim=emb_dim, hidden_dim=args.hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        train_input = history[:, :-1, :]
        train_target = history[:, -1, :]

        for epoch in range(args.train_epochs):
            optimizer.zero_grad()
            pred = model(train_input)
            loss = loss_fn(pred, train_target)
            loss.backward()
            optimizer.step()

        print(f"  Training loss: {loss.item():.6f}  ({args.train_epochs} epochs)")

        # --- Stage 6 flag checks (training) ---
        flag_check("FC-16", np.isfinite(loss.item()),
                   f"AR training loss={loss.item():.6f} is finite")

        ckpt_path = OUT_DIR / "ar_model_checkpoint.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

        # Rollout forecast
        model.eval()
        with torch.no_grad():
            forecasted = rollout_embeddings(model, history, steps=args.forecast_steps)

        print(f"  Forecasted shape: {tuple(forecasted.shape)}")

        # --- Stage 6 flag checks (forecast) ---
        flag_check("FC-17", torch.isfinite(forecasted).all().item(),
                   "Forecast tensor has no NaN/Inf")
        _expected_shape = (num_nodes, args.forecast_steps, emb_dim)
        flag_check("FC-18", tuple(forecasted.shape) == _expected_shape,
                   f"Forecast shape {tuple(forecasted.shape)} == {_expected_shape}")

        forecast_path = OUT_DIR / "embedding_forecast.pt"
        torch.save(forecasted, forecast_path)
        print(f"  Saved: {forecast_path}")

        # Decode adjacency from final forecasted embedding
        final_emb = forecasted[:, -1, :]
        adj_final = build_adjacency(
            data.yearly_edges[years[-1]], num_nodes
        )

        # Run edge-toggle interventions on the forecasted embedding
        for label, focal_cc in [("USA", USA_CCODE), ("China", CHN_CCODE)]:
            if focal_cc not in ccode_to_idx:
                continue
            partners = [cc for cc in ccode_to_idx if cc != focal_cc]
            results = simulate_edge_toggle(
                adj=adj_final,
                emb=final_emb,
                node_to_idx=ccode_to_idx,
                focal_ccode=focal_cc,
                partner_ccodes=partners,
            )
            idf = interventions_to_frame(results)
            ipath = OUT_DIR / f"forecast_interventions_{label.lower()}.csv"
            idf.to_csv(ipath, index=False)
            print(f"  Saved: {ipath}  ({len(idf)} rows)")

else:
    print("\n  (Skipping AR forecast — use without --skip-forecast to enable)")


# ===============================================================
# Stage 7 — Top edge changes summary
# ===============================================================
banner(7, "Summary")

print("\n  Top edge toggles by |embeddedness delta|:")
for label in ["USA", "China"]:
    subset = removal_df[removal_df["focal_country"] == label].head(10)
    if subset.empty:
        print(f"\n    {label}: no interventions available")
        continue
    print(f"\n    {label}:")
    for _, row in subset.iterrows():
        direction = "+" if row["embeddedness_delta"] >= 0 else ""
        print(
            f"      {row['operation']:6s} edge with {row['partner']:>10s} "
            f"(ccode {row['partner_ccode']:>4d})  =>  "
            f"{direction}{row['embeddedness_delta']:.6f}  [year {row['year']}]"
        )

# Artefact manifest
print("\n  Output files:")
for f in sorted(OUT_DIR.glob("*")):
    size_kb = f.stat().st_size / 1024
    print(f"    {f.name:45s} {size_kb:8.1f} KB")

elapsed = time.time() - t0
print(f"\n  Total runtime: {elapsed:.1f}s")
print(f"  All outputs in: {OUT_DIR}/")

# ===============================================================
# Flag check report
# ===============================================================
print(f"\n{'=' * 60}")
print("FLAG CHECK REPORT")
print("=" * 60)

_n_pass = sum(1 for f in _flag_results if f["status"] == "PASS")
_n_fail = sum(1 for f in _flag_results if f["status"] == "FAIL")
_n_total = len(_flag_results)

for f in _flag_results:
    marker = "OK" if f["status"] == "PASS" else "XX"
    print(f"  [{marker}] {f['code']}: {f['description']}")

print()

# Write flag report alongside outputs (before potential exit on failure)
flag_report_path = OUT_DIR / "flag_check_report.txt"
with open(flag_report_path, "w") as fh:
    fh.write(f"FLAG CHECK REPORT  (runtime: {elapsed:.1f}s)\n")
    fh.write(f"Data: {args.data_dir}\n")
    fh.write(f"{'=' * 60}\n\n")
    for f in _flag_results:
        marker = "PASS" if f["status"] == "PASS" else "FAIL"
        fh.write(f"[{marker}] {f['code']}: {f['description']}\n")
    fh.write(f"\nTotal: {_n_pass}/{_n_total} passed, {_n_fail} failed\n")
print(f"  Saved: {flag_report_path}")

if _n_fail == 0:
    print(f"  ALL FLAG CHECKS PASSED ({_n_pass}/{_n_total})")
else:
    print(f"  FLAG CHECKS: {_n_pass}/{_n_total} PASSED, {_n_fail} FAILED")
    print()
    print("  Failures:")
    for f in _flag_results:
        if f["status"] == "FAIL":
            print(f"    {f['code']}: {f['description']}")
    sys.exit(1)
