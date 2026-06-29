from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
except Exception:  # optional dependency gate
    GATConv = GCNConv = SAGEConv = None


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        if GCNConv is None:
            raise ImportError("torch_geometric is required for GCNEncoder. Install with: pip install torch_geometric")
        self.c1 = GCNConv(in_dim, hidden_dim)
        self.c2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.c1(x, edge_index, edge_weight))
        x = self.dropout(x)
        return self.c2(x, edge_index, edge_weight)


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required for GraphSAGEEncoder. Install with: pip install torch_geometric")
        self.c1 = SAGEConv(in_dim, hidden_dim)
        self.c2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.c1(x, edge_index))
        x = self.dropout(x)
        return self.c2(x, edge_index)


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if GATConv is None:
            raise ImportError("torch_geometric is required for GATEncoder. Install with: pip install torch_geometric")
        self.c1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.c2 = GATConv(hidden_dim * heads, out_dim, heads=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.c1(x, edge_index))
        x = self.dropout(x)
        return self.c2(x, edge_index)


@dataclass
class ModelSpec:
    name: str
    hidden_dim: int
    out_dim: int


def build_candidate_models(in_dim: int, specs: List[ModelSpec]) -> List[nn.Module]:
    models = []
    for s in specs:
        if s.name.lower() == "gcn":
            models.append(GCNEncoder(in_dim, s.hidden_dim, s.out_dim))
        elif s.name.lower() == "graphsage":
            models.append(GraphSAGEEncoder(in_dim, s.hidden_dim, s.out_dim))
        elif s.name.lower() == "gat":
            models.append(GATEncoder(in_dim, s.hidden_dim, s.out_dim))
        else:
            raise ValueError(f"Unknown model name: {s.name}")
    return models
