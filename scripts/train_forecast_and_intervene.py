#!/usr/bin/env python3
from __future__ import annotations

"""
Minimal end-to-end training scaffold:
1) loads preprocessed yearly edge tables,
2) fits a chosen GNN encoder and AR forecaster,
3) rolls embeddings 2026-2050,
4) evaluates edge-toggle interventions for USA (2) and China (710).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn_forecast.forecast import EmbeddingAR, rollout_embeddings
from gnn_forecast.interventions import interventions_to_frame, simulate_edge_toggle


USA_CCODE = 2
CHN_CCODE = 710


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-file", required=True, help=".pt tensor [num_nodes, seq_len, emb_dim]")
    ap.add_argument("--nodes-file", default="data/processed/nodes.csv")
    ap.add_argument("--adj-file", required=True, help="npy adjacency for final observed year")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    history = torch.load(args.embedding_file)
    model = EmbeddingAR(emb_dim=history.shape[-1], hidden_dim=64)

    # NOTE: plug in your trained model weights here; this is a placeholder untrained model.
    pred = rollout_embeddings(model, history, steps=25)
    torch.save(pred, out / "embedding_forecast_2026_2050.pt")

    nodes = pd.read_csv(args.nodes_file)
    node_to_idx = {int(c): i for i, c in enumerate(nodes["ccode"].tolist())}
    adj = np.load(args.adj_file)

    final_emb = pred[:, -1, :]
    partners = [c for c in node_to_idx.keys() if c not in (USA_CCODE, CHN_CCODE)]

    usa = interventions_to_frame(simulate_edge_toggle(adj, final_emb, node_to_idx, USA_CCODE, partners))
    chn = interventions_to_frame(simulate_edge_toggle(adj, final_emb, node_to_idx, CHN_CCODE, partners))

    usa.to_csv(out / "interventions_usa.csv", index=False)
    chn.to_csv(out / "interventions_china.csv", index=False)
    print("Saved forecasts and intervention rankings.")


if __name__ == "__main__":
    main()
