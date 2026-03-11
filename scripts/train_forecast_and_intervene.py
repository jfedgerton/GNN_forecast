#!/usr/bin/env python3
from __future__ import annotations

"""
Minimal end-to-end training scaffold (sequential / no user-defined functions):
1) load precomputed embedding history tensor
2) fit/use AR forecaster (placeholder unless you load weights)
3) rollout embeddings 25 steps (2026–2050)
4) run edge-toggle interventions for USA (2) and China (710)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn_forecast.forecast import EmbeddingAR, rollout_embeddings
from gnn_forecast.interventions import interventions_to_frame, simulate_edge_toggle

# -----------------------------
# Constants
# -----------------------------
USA_CCODE = 2
CHN_CCODE = 710

# -----------------------------
# Parse CLI args (no main())
# -----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--embedding-file", required=True, help=".pt tensor [num_nodes, seq_len, emb_dim]")
ap.add_argument("--nodes-file", default="data/processed/nodes.csv")
ap.add_argument("--adj-file", required=True, help="npy adjacency for final observed year")
ap.add_argument("--out-dir", default="outputs")
args = ap.parse_args()

# -----------------------------
# Output directory
# -----------------------------
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load embedding history
# -----------------------------
embedding_file = args.embedding_file
history = torch.load(embedding_file)

# history expected shape: [num_nodes, seq_len, emb_dim]
if history.ndim != 3:
    raise ValueError(f"Expected history to be 3D [N, T, D], got shape={tuple(history.shape)}")

num_nodes, seq_len, emb_dim = history.shape

# -----------------------------
# Initialize AR model (placeholder)
# -----------------------------
# NOTE: this is untrained unless you load weights. If you have a checkpoint,
# you should load it here via model.load_state_dict(torch.load(...)).
model = EmbeddingAR(emb_dim=emb_dim, hidden_dim=64)

# -----------------------------
# Rollout embeddings 25 steps (2026–2050)
# -----------------------------
# pred expected shape: [num_nodes, seq_len + steps, emb_dim] OR similar depending on rollout_embeddings
pred = rollout_embeddings(model, history, steps=25)

torch.save(pred, out_dir / "embedding_forecast_2026_2050.pt")

# -----------------------------
# Load nodes + build ccode -> idx map
# -----------------------------
nodes_file = args.nodes_file
nodes = pd.read_csv(nodes_file)

if "ccode" not in nodes.columns:
    raise ValueError(f"nodes file must contain a 'ccode' column. Found: {list(nodes.columns)}")

# IMPORTANT: this assumes nodes.csv ordering matches the embedding tensor node ordering.
node_ccodes = [int(c) for c in nodes["ccode"].tolist()]
node_to_idx = {cc: i for i, cc in enumerate(node_ccodes)}

# Basic consistency check
if len(node_ccodes) != num_nodes:
    raise ValueError(
        f"Mismatch: nodes.csv has {len(node_ccodes)} nodes but history has {num_nodes} nodes. "
        "Your node ordering/indexing is not aligned."
    )

# -----------------------------
# Load adjacency for final observed year
# -----------------------------
adj_file = args.adj_file
adj = np.load(adj_file)

# Basic adjacency shape check: expect [N, N]
if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
    raise ValueError(f"Expected adj to be square [N,N], got shape={adj.shape}")

if adj.shape[0] != num_nodes:
    raise ValueError(
        f"Mismatch: adj has N={adj.shape[0]} but history has N={num_nodes}. "
        "Adjacency and embedding nodes are not aligned."
    )

# -----------------------------
# Select final embedding slice
# -----------------------------
# pred should have time dimension in axis=1; last time step is -1
# final_emb expected shape: [N, D]
final_emb = pred[:, -1, :]

if final_emb.ndim != 2 or final_emb.shape[0] != num_nodes or final_emb.shape[1] != emb_dim:
    raise ValueError(f"final_emb shape unexpected: {tuple(final_emb.shape)}")

# -----------------------------
# Build partner list (exclude focal nodes)
# -----------------------------
partners = [cc for cc in node_to_idx.keys() if cc not in (USA_CCODE, CHN_CCODE)]

# -----------------------------
# Run interventions (USA)
# -----------------------------
usa_results = simulate_edge_toggle(
    adj=adj,
    emb=final_emb,
    node_to_idx=node_to_idx,
    focal_ccode=USA_CCODE,
    partner_ccodes=partners,
)
usa_df = interventions_to_frame(usa_results)
usa_df.to_csv(out_dir / "interventions_usa.csv", index=False)

# -----------------------------
# Run interventions (China)
# -----------------------------
chn_results = simulate_edge_toggle(
    adj=adj,
    emb=final_emb,
    node_to_idx=node_to_idx,
    focal_ccode=CHN_CCODE,
    partner_ccodes=partners,
)
chn_df = interventions_to_frame(chn_results)
chn_df.to_csv(out_dir / "interventions_china.csv", index=False)

print("Saved forecasts and intervention rankings.")
print(f"- {out_dir / 'embedding_forecast_2026_2050.pt'}")
print(f"- {out_dir / 'interventions_usa.csv'}")
print(f"- {out_dir / 'interventions_china.csv'}")
