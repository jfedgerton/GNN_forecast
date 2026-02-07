from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class EmbeddingAR(nn.Module):
    """Autoregressive forecaster over node embeddings (per node sequence)."""

    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        # x: [num_nodes, seq_len, emb_dim]
        h, _ = self.gru(x)
        return self.head(h[:, -1, :])


def decode_edges(emb: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Inner product decoder to produce adjacency predictions."""
    score = emb @ emb.T
    return (score > threshold).float()


@dataclass
class ForecastConfig:
    train_end_year: int = 2025
    forecast_start_year: int = 2026
    forecast_end_year: int = 2050


def rollout_embeddings(model: EmbeddingAR, history: torch.Tensor, steps: int) -> torch.Tensor:
    """
    history: [num_nodes, seq_len, emb_dim]
    returns: [num_nodes, steps, emb_dim]
    """
    seq = history.clone()
    preds = []
    for _ in range(steps):
        nxt = model(seq)
        preds.append(nxt)
        seq = torch.cat([seq[:, 1:, :], nxt.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)
