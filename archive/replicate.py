#!/usr/bin/env python3
"""
Replication script for GNN Forecast pipeline.

Exercises the full pipeline end-to-end using the test fixtures (tiny synthetic
data) so that correctness can be verified without the full peacesciencer
export.  Every stage prints a summary so the user can inspect outputs.

Usage:
    PYTHONPATH=src python scripts/replicate.py

Stages:
    1. Load canonical multiplex data from fixtures
    2. Build observed + residualized network layers
    3. Construct adjacency matrix from edge lists
    4. Train an AR embedding forecaster on synthetic embeddings
    5. Roll out 25-step forecasts (2026-2050)
    6. Run edge-toggle interventions for USA and China
    7. Export all artefacts to outputs/replication/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn_forecast.data_layer import CanonicalDataConfig, load_canonical_multiplex_data
from gnn_forecast.network_construction import (
    capital_distance_matrix,
    make_observed_and_residual_layers,
    residualize_ties_against_distance,
)
from gnn_forecast.forecast import EmbeddingAR, ForecastConfig, decode_edges, rollout_embeddings
from gnn_forecast.interventions import (
    embeddedness_score,
    interventions_to_frame,
    simulate_edge_toggle,
)

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
FIXTURE_DIR = Path("tests/fixtures/tiny_processed")
OUT_DIR = Path("outputs/replication")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_DIM = 8
HIDDEN_DIM = 16
SEQ_LEN = 5
FORECAST_STEPS = 25
TRAIN_EPOCHS = 50
LR = 1e-3
USA_CCODE = 2
CHN_CCODE = 710

errors: list[str] = []


def check(condition: bool, msg: str) -> None:
    """Record a failed check without aborting the script."""
    if not condition:
        errors.append(msg)
        print(f"  FAIL: {msg}")


# ===============================================================
# Stage 1 — Load canonical multiplex data
# ===============================================================
print("=" * 60)
print("Stage 1: Load canonical multiplex data from fixtures")
print("=" * 60)

config = CanonicalDataConfig(data_dir=str(FIXTURE_DIR))
data = load_canonical_multiplex_data(config)

print(f"  Nodes:       {len(data.nodes)} countries")
print(f"  ccode->idx:  {data.ccode_to_idx}")
print(f"  Years:       {sorted(data.yearly_edges.keys())}")
print(f"  Relations:   {sorted(next(iter(data.yearly_edges.values())).keys())}")

check(len(data.ccode_to_idx) == 3, "Expected 3 countries in fixture")
check(data.ccode_to_idx == {2: 0, 710: 1, 840: 2}, "ccode_to_idx mapping mismatch")
check(set(data.yearly_edges.keys()) == {2000, 2001}, "Expected years {2000, 2001}")

for year in data.yearly_edges:
    for rel, df in data.yearly_edges[year].items():
        check(
            {"source_ccode", "target_ccode", "tie", "source_idx", "target_idx"}.issubset(df.columns),
            f"Year {year} relation {rel} missing expected columns",
        )

print()

# ===============================================================
# Stage 2 — Build observed + residualized layers
# ===============================================================
print("=" * 60)
print("Stage 2: Build observed + residualized network layers")
print("=" * 60)

nodes_df = data.nodes.copy()
# Use cap_lat/cap_lon from fixture nodes
distances = capital_distance_matrix(nodes_df)
print(f"  Distance matrix rows: {len(distances)}")
check(len(distances) == 6, "Expected 6 directed pairs for 3 countries")

# Take year 2000 edges as example
year2000 = data.yearly_edges[2000]
alliance_edges = year2000["alliance"][["source_ccode", "target_ccode", "tie"]].copy()
resid = residualize_ties_against_distance(alliance_edges, distances)
print(f"  Residualized alliance edges (year 2000): {len(resid)} rows")
check({"pred_tie", "tie_resid"}.issubset(resid.columns), "Residualized output missing columns")

# Use the pipeline-level helper
layers_2000 = {rel: df[["source_ccode", "target_ccode", "tie"]].copy() for rel, df in year2000.items()}
bundle = make_observed_and_residual_layers(nodes_df, layers_2000)
print(f"  Observed layers:  {sorted(bundle.observed_layers.keys())}")
print(f"  Residual layers:  {sorted(bundle.residual_layers.keys())}")
check(set(bundle.observed_layers.keys()) == {"alliance", "igo", "trade"}, "Observed layer keys mismatch")
check(set(bundle.residual_layers.keys()) == {"alliance", "igo", "trade"}, "Residual layer keys mismatch")
print()

# ===============================================================
# Stage 3 — Construct adjacency matrix
# ===============================================================
print("=" * 60)
print("Stage 3: Construct adjacency matrix from edge lists")
print("=" * 60)

num_nodes = len(data.ccode_to_idx)
adj = np.zeros((num_nodes, num_nodes), dtype=float)

for rel, df in year2000.items():
    for _, row in df.iterrows():
        i, j = int(row["source_idx"]), int(row["target_idx"])
        adj[i, j] = max(adj[i, j], float(row["tie"]))
        adj[j, i] = max(adj[j, i], float(row["tie"]))

np.fill_diagonal(adj, 0.0)
print(f"  Adjacency shape: {adj.shape}")
print(f"  Non-zero entries: {int((adj > 0).sum())}")
check(adj.shape == (3, 3), "Adjacency shape should be (3, 3)")
check(np.allclose(adj, adj.T), "Adjacency should be symmetric")

adj_path = OUT_DIR / "adjacency_2000.npy"
np.save(adj_path, adj)
print(f"  Saved: {adj_path}")
print()

# ===============================================================
# Stage 4 — Train AR embedding forecaster
# ===============================================================
print("=" * 60)
print("Stage 4: Train AR embedding forecaster on synthetic data")
print("=" * 60)

torch.manual_seed(42)
# Synthetic embedding history: [num_nodes, seq_len, emb_dim]
history = torch.randn(num_nodes, SEQ_LEN, EMB_DIM)

model = EmbeddingAR(emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

# Train: predict last step from preceding steps
train_input = history[:, :-1, :]  # [N, T-1, D]
train_target = history[:, -1, :]  # [N, D]

for epoch in range(TRAIN_EPOCHS):
    optimizer.zero_grad()
    pred = model(train_input)
    loss = loss_fn(pred, train_target)
    loss.backward()
    optimizer.step()

print(f"  Final training loss: {loss.item():.6f}")
check(loss.item() < 1.0, "Training loss should converge below 1.0")

# Save checkpoint
ckpt_path = OUT_DIR / "ar_model_checkpoint.pt"
torch.save(model.state_dict(), ckpt_path)
print(f"  Saved checkpoint: {ckpt_path}")
print()

# ===============================================================
# Stage 5 — Rollout 25-step forecasts (2026-2050)
# ===============================================================
print("=" * 60)
print("Stage 5: Rollout 25-step embedding forecasts")
print("=" * 60)

forecast_cfg = ForecastConfig()
print(f"  Forecast window: {forecast_cfg.forecast_start_year}-{forecast_cfg.forecast_end_year}")

model.eval()
with torch.no_grad():
    forecasted = rollout_embeddings(model, history, steps=FORECAST_STEPS)

print(f"  Forecasted tensor shape: {tuple(forecasted.shape)}")
check(
    forecasted.shape == (num_nodes, FORECAST_STEPS, EMB_DIM),
    f"Expected forecast shape ({num_nodes}, {FORECAST_STEPS}, {EMB_DIM})",
)
check(torch.isfinite(forecasted).all().item(), "Forecasted embeddings contain NaN/Inf")

forecast_path = OUT_DIR / "embedding_forecast_2026_2050.pt"
torch.save(forecasted, forecast_path)
print(f"  Saved: {forecast_path}")

# Decode edges from final forecasted embedding
final_emb = forecasted[:, -1, :]
adj_pred = decode_edges(final_emb, threshold=0.0)
print(f"  Decoded adjacency shape: {tuple(adj_pred.shape)}")
check(adj_pred.shape == (num_nodes, num_nodes), "Decoded adjacency shape mismatch")
print()

# ===============================================================
# Stage 6 — Edge-toggle interventions
# ===============================================================
print("=" * 60)
print("Stage 6: Edge-toggle interventions (USA & China)")
print("=" * 60)

node_to_idx = data.ccode_to_idx
partners = [cc for cc in node_to_idx if cc not in (USA_CCODE, CHN_CCODE)]

# Baseline embeddedness scores
usa_baseline = embeddedness_score(final_emb, node_to_idx[USA_CCODE])
chn_baseline = embeddedness_score(final_emb, node_to_idx[CHN_CCODE])
print(f"  USA baseline embeddedness: {usa_baseline:.4f}")
print(f"  China baseline embeddedness: {chn_baseline:.4f}")

# USA interventions
usa_results = simulate_edge_toggle(
    adj=adj,
    emb=final_emb,
    node_to_idx=node_to_idx,
    focal_ccode=USA_CCODE,
    partner_ccodes=partners,
)
usa_df = interventions_to_frame(usa_results)
usa_path = OUT_DIR / "interventions_usa.csv"
usa_df.to_csv(usa_path, index=False)
print(f"  USA interventions: {len(usa_df)} partner(s)")
check(len(usa_df) >= 1, "Expected at least 1 USA intervention result")
print(usa_df.to_string(index=False))

# China interventions
chn_results = simulate_edge_toggle(
    adj=adj,
    emb=final_emb,
    node_to_idx=node_to_idx,
    focal_ccode=CHN_CCODE,
    partner_ccodes=partners,
)
chn_df = interventions_to_frame(chn_results)
chn_path = OUT_DIR / "interventions_china.csv"
chn_df.to_csv(chn_path, index=False)
print(f"  China interventions: {len(chn_df)} partner(s)")
check(len(chn_df) >= 1, "Expected at least 1 China intervention result")
print(chn_df.to_string(index=False))
print()

# ===============================================================
# Stage 7 — Summary
# ===============================================================
print("=" * 60)
print("Stage 7: Artefact summary")
print("=" * 60)

artefacts = sorted(OUT_DIR.glob("*"))
for a in artefacts:
    size_kb = a.stat().st_size / 1024
    print(f"  {a.name:45s} {size_kb:8.1f} KB")

print()
if errors:
    print(f"REPLICATION FINISHED WITH {len(errors)} ERROR(S):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("REPLICATION SUCCESSFUL — all checks passed.")
