from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class InterventionResult:
    focal_ccode: int
    partner_ccode: int
    operation: str
    embeddedness_delta: float


def embeddedness_score(emb: torch.Tensor, idx: int) -> float:
    # Mean cosine similarity to all other states as a simple embeddedness measure.
    v = emb[idx]
    all_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
    v_norm = v / (v.norm() + 1e-8)
    cos = all_norm @ v_norm
    return float(cos.mean().item())


def simulate_edge_toggle(
    adj: np.ndarray,
    emb: torch.Tensor,
    node_to_idx: dict,
    focal_ccode: int,
    partner_ccodes: Iterable[int],
    add_if_missing: bool = True,
) -> List[InterventionResult]:
    res = []
    focal_idx = node_to_idx[focal_ccode]
    baseline = embeddedness_score(emb, focal_idx)

    for p in partner_ccodes:
        p_idx = node_to_idx[p]
        adj2 = adj.copy()
        exists = adj2[focal_idx, p_idx] > 0
        if exists:
            adj2[focal_idx, p_idx] = 0
            adj2[p_idx, focal_idx] = 0
            op = "remove"
        elif add_if_missing:
            adj2[focal_idx, p_idx] = 1
            adj2[p_idx, focal_idx] = 1
            op = "add"
        else:
            continue

        # Lightweight proxy: one-hop message passing from modified adjacency.
        a = torch.tensor(adj2, dtype=emb.dtype, device=emb.device)
        deg = a.sum(1, keepdim=True).clamp(min=1.0)
        emb2 = a @ emb / deg
        delta = embeddedness_score(emb2, focal_idx) - baseline
        res.append(InterventionResult(focal_ccode, p, op, delta))

    return sorted(res, key=lambda x: x.embeddedness_delta, reverse=True)


def interventions_to_frame(results: List[InterventionResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])
