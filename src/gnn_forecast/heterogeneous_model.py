"""Heterogeneous (R-GCN-lite) encoder for multiplex IR networks.

Why a new encoder? The diagnostic ablation (2026-05-07) showed our
hand-rolled multiplex GCN cannot escape chance-level link prediction
even with rich training. The replacement design here:

  1. Per-relation linear transforms (W_r per layer type) — first-class
     R-GCN style, parameter-shared across edges of the same type.
  2. Self-loop transform (W_self) added separately.
  3. Symmetric normalization within each relation type.
  4. Optional learnable node-identity embeddings concatenated to input
     features — gives the encoder country-specific structural slack
     that pure attribute features can't capture.
  5. Two layers, ReLU between them, dropout in between for regularization.

The encoder produces a single [num_nodes, emb_dim] tensor per snapshot
that downstream modules (GRU, link decoder, wedge counterfactual sweep)
can consume the same way they did the old multiplex encoder.

Naming: kept distinct from `multiplex_model.py` so both versions live in
parallel during the architecture transition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _symmetric_normalized_aggregate(
    x: torch.Tensor,
    edge_index: torch.LongTensor,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sum_{j in N(i)} (1 / sqrt(deg(i)*deg(j))) * w_ij * x_j (no self-loops)."""
    num_nodes = x.size(0)
    if edge_index.numel() == 0:
        return torch.zeros_like(x)
    row, col = edge_index[0], edge_index[1]
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=x.device)
    deg = torch.zeros(num_nodes, device=x.device)
    deg.scatter_add_(0, row, edge_weight.abs())
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    messages = x[col] * norm.unsqueeze(1)
    out = torch.zeros_like(x)
    out.scatter_add_(0, row.unsqueeze(1).expand_as(messages), messages)
    return out


class RGCNLayer(nn.Module):
    """One layer of relational graph convolution.

    h_i^{l+1} = sigma(W_self h_i^l + sum_r aggregate_r(W_r h_j^l))
    """

    def __init__(self, in_dim: int, out_dim: int, relation_names: List[str]):
        super().__init__()
        self.relation_names = list(relation_names)
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)
        self.W_rel = nn.ModuleDict({
            r: nn.Linear(in_dim, out_dim, bias=False) for r in self.relation_names
        })

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: Dict[str, torch.LongTensor],
        edge_weights: Dict[str, torch.FloatTensor],
        layer_mask: Dict[str, bool],
    ) -> torch.Tensor:
        out = self.W_self(x)
        for r in self.relation_names:
            if not layer_mask.get(r, False):
                continue
            ei = edge_indices.get(r)
            if ei is None or ei.numel() == 0:
                continue
            ew = edge_weights.get(r)
            transformed = self.W_rel[r](x)
            agg = _symmetric_normalized_aggregate(transformed, ei, ew)
            out = out + agg
        return out


@dataclass
class HeterogeneousEncoderConfig:
    relation_names: List[str]
    raw_feat_dim: int          # dimension of node attribute input (e.g., COW features)
    identity_dim: int = 16     # size of learnable per-node identity embedding (0 = disable)
    hidden_dim: int = 64
    emb_dim: int = 32
    dropout: float = 0.2


class HeterogeneousEncoder(nn.Module):
    """Two-layer R-GCN with optional learnable node-identity embeddings."""

    def __init__(self, num_nodes: int, config: HeterogeneousEncoderConfig):
        super().__init__()
        self.config = config
        if config.identity_dim > 0:
            self.identity = nn.Embedding(num_nodes, config.identity_dim)
            nn.init.normal_(self.identity.weight, std=0.1)
            in_dim = config.raw_feat_dim + config.identity_dim
        else:
            self.identity = None
            in_dim = config.raw_feat_dim
        self.layer1 = RGCNLayer(in_dim, config.hidden_dim, config.relation_names)
        self.layer2 = RGCNLayer(config.hidden_dim, config.emb_dim, config.relation_names)
        self.dropout = nn.Dropout(config.dropout)
        # Cache num_nodes for forward
        self.num_nodes = num_nodes

    def forward(
        self,
        node_features: torch.Tensor,
        edge_indices: Dict[str, torch.LongTensor],
        edge_weights: Dict[str, torch.FloatTensor],
        layer_mask: Dict[str, bool],
    ) -> torch.Tensor:
        if self.identity is not None:
            ids = self.identity(torch.arange(self.num_nodes, device=node_features.device))
            x = torch.cat([node_features, ids], dim=1)
        else:
            x = node_features
        h = self.layer1(x, edge_indices, edge_weights, layer_mask)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer2(h, edge_indices, edge_weights, layer_mask)
        return h


# ---------------------------------------------------------------
# Composite model: heterogeneous encoder + GRU temporal head
# ---------------------------------------------------------------

@dataclass
class HeteroTemporalConfig:
    encoder: HeterogeneousEncoderConfig
    temporal_hidden_dim: int = 64
    seq_len: int = 5


class HeterogeneousTemporalGNN(nn.Module):
    """Heterogeneous encoder + GRU temporal head.

    Drop-in replacement for MultiplexTemporalGNN. Same external interface
    (encode_snapshot, forward_temporal, decode_edges) so existing
    counterfactual machinery still works.
    """

    def __init__(self, num_nodes: int, config: HeteroTemporalConfig):
        super().__init__()
        self.config = config
        self.encoder = HeterogeneousEncoder(num_nodes, config.encoder)
        self.gru = nn.GRU(
            input_size=config.encoder.emb_dim,
            hidden_size=config.temporal_hidden_dim,
            batch_first=True,
        )
        self.head = nn.Linear(config.temporal_hidden_dim, config.encoder.emb_dim)
        self.layer_names = config.encoder.relation_names
        self.num_nodes = num_nodes

    def encode_snapshot(self, node_features, edge_indices, edge_weights, layer_mask):
        return self.encoder(node_features, edge_indices, edge_weights, layer_mask)

    def forward_temporal(self, embedding_history: List[torch.Tensor]) -> torch.Tensor:
        seq = torch.stack(embedding_history, dim=1)  # [N, T, D]
        h, _ = self.gru(seq)
        return self.head(h[:, -1, :])

    def decode_edges(self, emb: torch.Tensor) -> torch.Tensor:
        return emb @ emb.T

    def freeze_encoder(self) -> None:
        """Freeze encoder weights (used for two-stage pretrain → GRU training)."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True
