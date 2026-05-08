"""Multi-focal edge-intervention machinery for the R-GCN pipeline.

Counterpart to `feature_intervention.py` (which perturbs node features).
This module sweeps over (partner, layer, operation) edge perturbations,
re-encodes the perturbed graph at the same frozen R-GCN encoder weights,
and reports per-focal embeddedness deltas for an arbitrary set of focal
countries.

Why a new module instead of generalizing isolation_analysis.py? The old
isolation_analysis.dual_focal_simulation depends on MultiplexTemporalGNN
(the deprecated per-layer GCN + attention encoder). The new pipeline uses
HeterogeneousEncoder (R-GCN). Rather than touch a working file that
serves the old pipeline, we mirror the design here against the new
encoder. Both can coexist.

Design principles:
  - Long-form output: one row per (partner, layer, operation, focal).
    This makes pairwise wedge computation a pure DataFrame operation
    downstream and avoids the wide-format combinatorial explosion you
    get when adding focals.
  - Symmetric perturbations only (default k=5). The legacy "remove ALL /
    add up to 50" mode lives in the old isolation_analysis.py; for the
    paper's headline results we use k=5 add and k=5 remove so deltas
    are directly comparable.
  - Frozen encoder. Same weights for baseline and counterfactual passes.
  - No temporal head. Operates on year-of-perturbation embeddings only.
    Once we train the GRU on R-GCN embeddings, a sister function will
    extend this to forecasted-year deltas.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from gnn_forecast.heterogeneous_model import HeterogeneousEncoder
from gnn_forecast.diagnostic_v3 import MultiplexTemporalDataset
from gnn_forecast.node_features import NodeFeatureSet
from gnn_forecast.counterfactual import (
    USA_CCODE, CHN_CCODE, RUS_CCODE, IND_CCODE, FOUR_FOCAL_CCODES,
    MAJOR_POWER_CCODES,
)

SEED = 123


# ---------------------------------------------------------------
# Embeddedness metrics (compact reimplementation for new pipeline)
# ---------------------------------------------------------------

def _embeddedness_for_focals(
    emb: torch.Tensor,
    focal_indices: List[int],
    major_power_indices: List[int],
    knn_k: int = 5,
) -> Dict[int, Dict[str, float]]:
    """Compute centroid distance, mean cosine, mean distance to majors, kNN density
    for each focal. Returns {focal_idx: {metric_name: value}}.

    Single forward pass over the embedding matrix; per-focal metrics are
    cheap O(N) reductions. Cost is dominated by the kNN computation but
    we only need it for the few focals.
    """
    # Centroid (Euclidean): focal-to-mean-of-others
    n = emb.size(0)
    centroid = emb.mean(dim=0, keepdim=True)
    centroid_dist = torch.linalg.vector_norm(emb - centroid, dim=1)  # [N]

    # Cosine: focal-to-mean-cosine vs all others (proximity)
    norm = torch.linalg.vector_norm(emb, dim=1, keepdim=True).clamp_min(1e-8)
    unit = emb / norm
    cos_to_others = unit @ unit.T  # [N, N]
    cos_to_others.fill_diagonal_(0.0)
    mean_cos = cos_to_others.sum(dim=1) / max(n - 1, 1)  # [N]

    # Major-power distance: mean L2 from focal to each major-power embedding
    if major_power_indices:
        mp_emb = emb[major_power_indices]  # [M, D]
        # broadcast distances
        diff = emb.unsqueeze(1) - mp_emb.unsqueeze(0)  # [N, M, D]
        mp_dist = torch.linalg.vector_norm(diff, dim=2).mean(dim=1)  # [N]
    else:
        mp_dist = torch.zeros(n, device=emb.device)

    # k-NN density: mean L2 to k nearest neighbors (excluding self)
    pairwise = torch.cdist(emb, emb)  # [N, N]
    pairwise.fill_diagonal_(float("inf"))
    knn_vals, _ = pairwise.topk(min(knn_k, n - 1), dim=1, largest=False)
    knn_density = knn_vals.mean(dim=1)  # [N]

    out: Dict[int, Dict[str, float]] = {}
    for idx in focal_indices:
        out[idx] = {
            "centroid_distance": float(centroid_dist[idx].item()),
            "mean_cosine": float(mean_cos[idx].item()),
            "major_power_distance": float(mp_dist[idx].item()),
            "knn_density": float(knn_density[idx].item()),
        }
    return out


# ---------------------------------------------------------------
# Edge perturbation primitives
# ---------------------------------------------------------------

def _modify_edges_symmetric(
    edge_indices: Dict[str, torch.LongTensor],
    edge_weights: Dict[str, torch.FloatTensor],
    layer_name: str,
    partner_idx: int,
    operation: str,
    k: int,
    num_nodes: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.LongTensor], Dict[str, torch.FloatTensor]]:
    """Clone all layers and apply a symmetric add/remove of k edges
    involving partner_idx in the named layer.

    add: connect partner to k currently-unconnected nodes (deterministic).
    remove: drop the first k edges that touch partner.
    """
    mod_ei = {ln: edge_indices[ln].clone().to(device) for ln in edge_indices}
    mod_ew = {ln: edge_weights[ln].clone().to(device) for ln in edge_weights}

    if layer_name not in mod_ei:
        return mod_ei, mod_ew

    ei_l = mod_ei[layer_name]
    ew_l = mod_ew[layer_name]

    if operation == "remove" and ei_l.numel() > 0:
        partner_mask = (ei_l[0] == partner_idx) | (ei_l[1] == partner_idx)
        partner_edge_idx = partner_mask.nonzero(as_tuple=False).flatten()
        to_remove = partner_edge_idx[:k]
        keep_mask = torch.ones(ei_l.size(1), dtype=torch.bool, device=device)
        keep_mask[to_remove] = False
        mod_ei[layer_name] = ei_l[:, keep_mask]
        mod_ew[layer_name] = ew_l[keep_mask]
    elif operation == "add":
        existing = set()
        if ei_l.numel() > 0:
            src_mask = ei_l[0] == partner_idx
            existing.update(ei_l[1, src_mask].cpu().tolist())
            tgt_mask = ei_l[1] == partner_idx
            existing.update(ei_l[0, tgt_mask].cpu().tolist())
        unconnected = [
            n for n in range(num_nodes)
            if n != partner_idx and n not in existing
        ]
        new_targets = unconnected[:k]
        if new_targets:
            src = [partner_idx] * len(new_targets) + new_targets
            tgt = new_targets + [partner_idx] * len(new_targets)
            new_e = torch.tensor([src, tgt], dtype=torch.long, device=device)
            new_w = torch.ones(len(src), device=device)
            if ei_l.numel() > 0:
                mod_ei[layer_name] = torch.cat([ei_l, new_e], dim=1)
                mod_ew[layer_name] = torch.cat([ew_l, new_w])
            else:
                mod_ei[layer_name] = new_e
                mod_ew[layer_name] = new_w
    return mod_ei, mod_ew


# ---------------------------------------------------------------
# Multi-focal edge sweep
# ---------------------------------------------------------------

def multi_focal_edge_sweep(
    encoder: HeterogeneousEncoder,
    dataset: MultiplexTemporalDataset,
    feat_set: NodeFeatureSet,
    focal_ccodes: List[int] = FOUR_FOCAL_CCODES,
    partner_ccodes: Optional[List[int]] = None,
    layer_names: Optional[List[str]] = None,
    operations: Optional[List[str]] = None,
    focal_year: Optional[int] = None,
    symmetric_n_edges: int = 5,
    device: Optional[torch.device] = None,
    progress_every: int = 200,
) -> pd.DataFrame:
    """Sweep edge perturbations and report per-focal deltas.

    Returns a long-form DataFrame with one row per
    (partner_ccode, layer_name, operation, focal_ccode):

        partner_ccode | layer_name | operation | focal_ccode | focal_idx |
        delta_centroid | delta_cosine | delta_mp_dist | delta_knn |
        baseline_centroid | cf_centroid | (and same for the other 3 metrics)

    Pairwise wedges are computed downstream by `pairwise_wedges()`.
    """
    if device is None:
        device = next(encoder.parameters()).device
    encoder.eval()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if focal_year is None:
        focal_year = dataset.years[-1]
    if layer_names is None:
        layer_names = list(dataset.layer_names)
    if operations is None:
        operations = ["add", "remove"]

    ccode_to_idx = dataset.ccode_to_idx
    for cc in focal_ccodes:
        if cc not in ccode_to_idx:
            raise ValueError(f"focal_ccode {cc} not in dataset node index")

    focal_indices = [ccode_to_idx[cc] for cc in focal_ccodes]
    focal_set = set(focal_ccodes)

    if partner_ccodes is None:
        partner_ccodes = [cc for cc in ccode_to_idx.keys() if cc not in focal_set]

    major_power_indices = [
        ccode_to_idx[cc] for cc in MAJOR_POWER_CCODES if cc in ccode_to_idx
    ]

    snap = dataset.snapshots[focal_year]
    nf = feat_set.by_year[focal_year].to(device)
    base_ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
    base_ew = {k: v.to(device) for k, v in snap.edge_weights.items()}

    # Baseline embedding + per-focal metrics (one forward pass)
    with torch.no_grad():
        base_emb = encoder(nf, base_ei, base_ew, snap.layer_mask)
    baseline_metrics = _embeddedness_for_focals(
        base_emb, focal_indices, major_power_indices,
    )

    rows: List[Dict] = []
    total = len(partner_ccodes) * len(layer_names) * len(operations)
    done = 0
    print(f"[multi-focal] sweeping {total} perturbations over "
          f"{len(focal_ccodes)} focals at year {focal_year}")

    for partner_ccode in partner_ccodes:
        partner_idx = ccode_to_idx[partner_ccode]
        for layer_name in layer_names:
            if not snap.layer_mask.get(layer_name, False):
                continue
            for op in operations:
                mod_ei, mod_ew = _modify_edges_symmetric(
                    base_ei, base_ew, layer_name, partner_idx, op,
                    symmetric_n_edges, dataset.num_nodes, device,
                )
                with torch.no_grad():
                    cf_emb = encoder(nf, mod_ei, mod_ew, snap.layer_mask)
                cf_metrics = _embeddedness_for_focals(
                    cf_emb, focal_indices, major_power_indices,
                )
                for focal_ccode, focal_idx in zip(focal_ccodes, focal_indices):
                    base_m = baseline_metrics[focal_idx]
                    cf_m = cf_metrics[focal_idx]
                    rows.append({
                        "partner_ccode": partner_ccode,
                        "layer_name": layer_name,
                        "operation": op,
                        "focal_ccode": focal_ccode,
                        "focal_idx": focal_idx,
                        "delta_centroid": cf_m["centroid_distance"] - base_m["centroid_distance"],
                        "delta_cosine":   cf_m["mean_cosine"]        - base_m["mean_cosine"],
                        "delta_mp_dist":  cf_m["major_power_distance"] - base_m["major_power_distance"],
                        "delta_knn":      cf_m["knn_density"]         - base_m["knn_density"],
                        "baseline_centroid": base_m["centroid_distance"],
                        "cf_centroid":       cf_m["centroid_distance"],
                        "baseline_cosine":   base_m["mean_cosine"],
                        "cf_cosine":         cf_m["mean_cosine"],
                    })
                done += 1
                if done % progress_every == 0:
                    print(f"  ... {done}/{total}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Pairwise wedges
# ---------------------------------------------------------------

def pairwise_wedges(
    long_df: pd.DataFrame,
    metric: str = "delta_centroid",
) -> pd.DataFrame:
    """Given a long-form sweep DataFrame, compute wedges
    (delta_focal_a - delta_focal_b) for every ordered pair of focals.

    Returns one row per (partner_ccode, layer_name, operation,
    focal_a_ccode, focal_b_ccode) with the wedge value. For a 4-focal
    set this is C(4, 2) = 6 unordered pairs * 2 orderings = 12 rows
    per perturbation; we keep both orderings so callers can sort by
    signed wedge in either direction.
    """
    focal_ccodes = sorted(long_df["focal_ccode"].unique().tolist())
    out_rows: List[Dict] = []

    for (partner, layer, op), grp in long_df.groupby(
        ["partner_ccode", "layer_name", "operation"]
    ):
        deltas = dict(zip(grp["focal_ccode"], grp[metric]))
        for a in focal_ccodes:
            for b in focal_ccodes:
                if a == b:
                    continue
                out_rows.append({
                    "partner_ccode": partner,
                    "layer_name":    layer,
                    "operation":     op,
                    "focal_a_ccode": a,
                    "focal_b_ccode": b,
                    "metric":        metric,
                    f"delta_{a}":    deltas.get(a, np.nan),
                    f"delta_{b}":    deltas.get(b, np.nan),
                    "wedge":         deltas.get(a, np.nan) - deltas.get(b, np.nan),
                })
    return pd.DataFrame(out_rows)
