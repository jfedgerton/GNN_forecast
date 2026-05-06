"""Archived: SimpleGCNLayer (for-loop variant).

Removed from src/gnn_forecast/multiplex_model.py on 2026-05-05.

This was the original, non-vectorized fallback GCN layer used when
torch_geometric was unavailable. It iterates edge-by-edge in a Python
for-loop, which is O(num_edges) Python overhead per forward pass.

It was never imported or instantiated anywhere in the codebase --
LayerEncoder always used SimpleGCNLayerSparse (the scatter_add_-based
vectorized version) for the fallback path.

Kept here in case someone wants the simpler reference implementation
for debugging or pedagogical purposes. Do not import from production
code.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SimpleGCNLayer(nn.Module):
    """Minimal GCN layer using sparse adjacency multiplication.

    Implements: H' = D^{-1/2} A D^{-1/2} H W
    Falls back to this when torch_geometric is unavailable.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        if num_nodes is None:
            num_nodes = x.size(0)

        # Build sparse adjacency with self-loops
        if edge_index.numel() == 0:
            return self.weight(x)

        # Add self-loops
        self_loops = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
        ei = torch.cat([edge_index, self_loops], dim=1)

        if edge_weight is not None:
            ew = torch.cat([edge_weight, torch.ones(num_nodes, device=x.device)])
        else:
            ew = torch.ones(ei.size(1), device=x.device)

        # Symmetric normalization
        row, col = ei[0], ei[1]
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, ew)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm = deg_inv_sqrt[row] * ew * deg_inv_sqrt[col]

        # Sparse message passing
        out = torch.zeros_like(x)
        src_features = self.weight(x)
        for i in range(ei.size(1)):
            out[row[i]] += norm[i] * src_features[col[i]]

        return out
