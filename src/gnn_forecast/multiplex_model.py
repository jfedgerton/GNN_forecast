"""Dynamic Multiplex GNN for forecasting international embedding space.

Architecture:
1. Per-layer GCN encoders (separate weights per layer)
2. Learned attention-based layer aggregation
3. GRU temporal model for next-period embedding prediction
4. Inner-product edge decoder for link reconstruction regularization

Two model sets use the same architecture with different input data:
- Model Set 1: Raw observed adjacency matrices
- Model Set 2: Strength-weighted (residualized) adjacency matrices
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


# ---------------------------------------------------------------
# Fallback GCN layer when torch_geometric is not available
# ---------------------------------------------------------------
# Note: an earlier non-vectorized SimpleGCNLayer (Python for-loop variant)
# was archived to archive/dead_simple_gcn_layer.py on 2026-05-05. It was
# never imported anywhere; SimpleGCNLayerSparse below is the live fallback.


class SimpleGCNLayerSparse(nn.Module):
    """GCN layer using scatter operations (vectorized, no for-loop)."""
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
        deg.scatter_add_(0, row, ew.abs())
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm = deg_inv_sqrt[row] * ew * deg_inv_sqrt[col]

        # Transform features
        h = self.weight(x)

        # Scatter-based message passing (vectorized)
        messages = h[col] * norm.unsqueeze(1)
        out = torch.zeros(num_nodes, h.size(1), device=x.device, dtype=h.dtype)
        out.scatter_add_(0, row.unsqueeze(1).expand_as(messages), messages)

        return out


# ---------------------------------------------------------------
# Per-layer GCN encoder
# ---------------------------------------------------------------
class LayerEncoder(nn.Module):
    """Two-layer GCN encoder for a single network layer."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, use_pyg: bool = False):
        super().__init__()
        self.use_pyg = use_pyg and HAS_TORCH_GEOMETRIC

        if self.use_pyg:
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        else:
            self.conv1 = SimpleGCNLayerSparse(in_dim, hidden_dim)
            self.conv2 = SimpleGCNLayerSparse(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_pyg:
            h = F.relu(self.conv1(x, edge_index, edge_weight))
            return self.conv2(h, edge_index, edge_weight)
        else:
            h = F.relu(self.conv1(x, edge_index, edge_weight, num_nodes=x.size(0)))
            return self.conv2(h, edge_index, edge_weight, num_nodes=x.size(0))


# ---------------------------------------------------------------
# Layer aggregation
# ---------------------------------------------------------------
class LayerAggregator(nn.Module):
    """Aggregate per-layer embeddings via learned attention weights.

    alpha_l = softmax(w_l) where w is a learnable parameter vector.
    H_t = sum_l alpha_l * H_t^(l)
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

    def forward(
        self,
        layer_embeddings: List[torch.Tensor],
        layer_mask: List[bool],
    ) -> torch.Tensor:
        """Aggregate layer embeddings, masking unavailable layers.

        Parameters
        ----------
        layer_embeddings : list of [num_nodes, emb_dim] tensors
        layer_mask : list of bool, True if layer is available
        """
        # Mask out unavailable layers by setting their logits to -inf
        logits = self.layer_logits.clone()
        for i, available in enumerate(layer_mask):
            if not available:
                logits[i] = float("-inf")

        # If no layers available, return zeros
        if not any(layer_mask):
            return torch.zeros_like(layer_embeddings[0])

        weights = F.softmax(logits, dim=0)

        out = torch.zeros_like(layer_embeddings[0])
        for i, (emb, available) in enumerate(zip(layer_embeddings, layer_mask)):
            if available:
                out = out + weights[i] * emb
        return out

    def get_layer_weights(self, layer_mask: Optional[List[bool]] = None) -> torch.Tensor:
        """Return current attention weights (for inspection)."""
        logits = self.layer_logits.clone()
        if layer_mask is not None:
            for i, available in enumerate(layer_mask):
                if not available:
                    logits[i] = float("-inf")
        return F.softmax(logits, dim=0)


# ---------------------------------------------------------------
# Temporal GRU model
# ---------------------------------------------------------------
class TemporalGRU(nn.Module):
    """GRU that maps a sequence of aggregated embeddings to next-period prediction.

    Input: sequence of [num_nodes, emb_dim] across time
    Output: predicted [num_nodes, emb_dim] for next period
    """

    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, emb_dim)

    def forward(self, embedding_sequence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embedding_sequence : [num_nodes, seq_len, emb_dim]

        Returns
        -------
        next_embedding : [num_nodes, emb_dim]
        """
        h, _ = self.gru(embedding_sequence)
        return self.head(h[:, -1, :])


# ---------------------------------------------------------------
# Full multiplex temporal GNN
# ---------------------------------------------------------------
@dataclass
class MultiplexGNNConfig:
    """Configuration for the multiplex temporal GNN."""
    num_layers: int = 5           # number of network layers
    in_dim: int = 5               # input node feature dimension (= num_layers for degree profile)
    hidden_dim: int = 64          # hidden dimension for GCN encoders
    emb_dim: int = 32             # output embedding dimension
    temporal_hidden_dim: int = 64 # GRU hidden dimension
    seq_len: int = 5              # number of historical years used for temporal prediction
    use_pyg: bool = False         # use torch_geometric GCN or fallback


class MultiplexTemporalGNN(nn.Module):
    """Full dynamic multiplex GNN model.

    Components:
    1. Per-layer GCN encoders
    2. Learned attention-based layer aggregation
    3. GRU temporal model
    4. Inner-product edge decoder (for link prediction regularization)
    """

    def __init__(self, config: MultiplexGNNConfig, layer_names: List[str]):
        super().__init__()
        self.config = config
        self.layer_names = layer_names
        self.num_layers = len(layer_names)

        # Per-layer encoders
        self.layer_encoders = nn.ModuleDict({
            name: LayerEncoder(
                in_dim=config.in_dim,
                hidden_dim=config.hidden_dim,
                out_dim=config.emb_dim,
                use_pyg=config.use_pyg,
            )
            for name in layer_names
        })

        # Layer aggregation
        self.aggregator = LayerAggregator(self.num_layers)

        # Temporal model
        self.temporal = TemporalGRU(
            emb_dim=config.emb_dim,
            hidden_dim=config.temporal_hidden_dim,
        )

    def encode_snapshot(
        self,
        node_features: torch.Tensor,
        edge_indices: Dict[str, torch.LongTensor],
        edge_weights: Dict[str, torch.FloatTensor],
        layer_mask: Dict[str, bool],
    ) -> torch.Tensor:
        """Encode a single time step: per-layer GCN -> layer aggregation.

        Returns aggregated embedding [num_nodes, emb_dim].
        """
        layer_embs = []
        mask_list = []
        for name in self.layer_names:
            available = layer_mask.get(name, False)
            mask_list.append(available)
            if available and name in edge_indices and edge_indices[name].numel() > 0:
                ei = edge_indices[name]
                ew = edge_weights.get(name)
                emb = self.layer_encoders[name](node_features, ei, ew)
            else:
                emb = torch.zeros(node_features.size(0), self.config.emb_dim,
                                  device=node_features.device)
            layer_embs.append(emb)

        return self.aggregator(layer_embs, mask_list)

    def forward_temporal(
        self,
        embedding_history: List[torch.Tensor],
    ) -> torch.Tensor:
        """Given a sequence of aggregated embeddings, predict next period.

        Parameters
        ----------
        embedding_history : list of [num_nodes, emb_dim] tensors (length = seq_len)

        Returns
        -------
        predicted embedding [num_nodes, emb_dim]
        """
        # Stack into [num_nodes, seq_len, emb_dim]
        seq = torch.stack(embedding_history, dim=1)
        return self.temporal(seq)

    def decode_edges(self, emb: torch.Tensor) -> torch.Tensor:
        """Inner product decoder: score_ij = emb_i . emb_j

        Returns [num_nodes, num_nodes] score matrix.
        """
        return emb @ emb.T

    def get_layer_weights(self, layer_mask: Optional[Dict[str, bool]] = None) -> Dict[str, float]:
        """Return layer attention weights for inspection."""
        mask_list = None
        if layer_mask is not None:
            mask_list = [layer_mask.get(name, False) for name in self.layer_names]
        weights = self.aggregator.get_layer_weights(mask_list)
        return {name: float(w.detach()) for name, w in zip(self.layer_names, weights)}
