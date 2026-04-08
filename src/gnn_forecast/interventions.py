from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class RandomDeletionResult:
    focal_ccode: int
    k: int
    sim_id: int
    deleted_partner_ccodes: List[int]
    embeddedness_baseline: float
    embeddedness_after: float
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
    if focal_ccode not in node_to_idx:
        raise ValueError(f"Focal country code {focal_ccode} not found in node index")
    missing_partners = [p for p in partner_ccodes if p not in node_to_idx]
    if missing_partners:
        raise ValueError(f"Partner country codes not found in node index: {missing_partners}")
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
        a = torch.zeros(adj2.shape[0], adj2.shape[1], dtype=emb.dtype, device=emb.device)
        a[:] = torch.as_tensor(adj2, dtype=emb.dtype)
        deg = a.sum(1, keepdim=True).clamp(min=1.0)
        emb2 = a @ emb / deg
        delta = embeddedness_score(emb2, focal_idx) - baseline
        res.append(InterventionResult(focal_ccode, p, op, delta))

    return sorted(res, key=lambda x: x.embeddedness_delta, reverse=True)


def interventions_to_frame(results: List[InterventionResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def simulate_random_multi_edge_deletion(
    adj: np.ndarray,
    emb: torch.Tensor,
    focal_idx: int,
    focal_ccode: int,
    idx_to_ccode: dict,
    k_values: Iterable[int] = range(2, 11),
    n_sims: int = 100,
    rng_seed: int = 42,
) -> List[RandomDeletionResult]:
    """Randomly delete *k* edges from the focal node and measure embeddedness change.

    For each *k* in *k_values*, runs *n_sims* simulations.  In each simulation
    *k* edges are sampled uniformly at random (without replacement) from the
    focal node's current neighbours and removed simultaneously.  The
    embeddedness delta is recorded for each draw.

    If the focal node's degree is less than *k*, that *k* value is skipped.

    Returns a list of :class:`RandomDeletionResult` sorted by (k, sim_id).
    """
    rng = np.random.default_rng(rng_seed)
    baseline = embeddedness_score(emb, focal_idx)

    # Identify all current neighbours of the focal node
    neighbours = np.nonzero(adj[focal_idx] > 0)[0]
    degree = len(neighbours)

    results: List[RandomDeletionResult] = []

    for k in k_values:
        if k > degree:
            continue

        for sim in range(n_sims):
            chosen = rng.choice(neighbours, size=k, replace=False)
            adj2 = adj.copy()
            for j in chosen:
                adj2[focal_idx, j] = 0
                adj2[j, focal_idx] = 0

            a = torch.zeros(adj2.shape[0], adj2.shape[1], dtype=emb.dtype, device=emb.device)
            a[:] = torch.as_tensor(adj2, dtype=emb.dtype)
            deg = a.sum(1, keepdim=True).clamp(min=1.0)
            emb2 = a @ emb / deg
            score_after = embeddedness_score(emb2, focal_idx)

            results.append(RandomDeletionResult(
                focal_ccode=focal_ccode,
                k=k,
                sim_id=sim,
                deleted_partner_ccodes=[int(idx_to_ccode[int(j)]) for j in chosen],
                embeddedness_baseline=baseline,
                embeddedness_after=score_after,
                embeddedness_delta=score_after - baseline,
            ))

    return results
