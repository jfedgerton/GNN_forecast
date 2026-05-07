"""Counterfactual edge simulation module.

CAUSAL INTERPRETATION — read before quoting any numbers.
The "interventions" implemented here are MODEL-BASED counterfactuals:
they ask "what would the trained encoder produce if the focal year's
graph had edge X added or removed?" This is not a do-calculus causal
effect on the real international system. Three things break a causal
interpretation:
  1. The encoder is trained on observed correlational data; learned
     message-passing weights reflect joint distributions, not structural
     mechanisms.
  2. SUTVA is violated by construction — every node's predicted
     embedding depends on every other node's edges, so removing one tie
     changes ALL nodes' outputs.
  3. The "next-year" prediction is conditional on the trained model's
     dynamics, which are themselves estimated from correlational data.

What these counterfactuals DO support:
  - Sensitivity rankings: which edges, if changed in the model, produce
    the largest predicted shifts in focal-state embeddedness?
  - Comparative leverage: which layers and partners matter most under
    the model's learned dynamics?
  - Scenario projection: assuming historical patterns persist, what
    does removing tie X imply for the embedding space?

These are useful but should be reported as model-implied sensitivities,
not as causal effects. The simulation study (simulation.run_recovery_
study) validates that the procedure recovers PLANTED effects in
synthetic data, which is the strongest claim available.

After training, simulate edge additions and edge knockouts in selected layers
to estimate how adding or removing specific ties changes the future embeddedness
or isolation of focal states (USA and China).

Supports:
1. Single-edge knockouts
2. Single-edge additions
3. Batch ranking of candidate tie changes
4. Comparison across layers
5. Separate analysis for USA and China

Embeddedness measures:
- mean_cosine_similarity: avg cosine sim to all other states
- centroid_distance: Euclidean distance to system centroid
- major_power_distance: avg distance to P5 states
- knn_density: avg distance to k nearest neighbors
- rank_shift: change in rank ordering by mean cosine similarity
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .multiplex_model import MultiplexTemporalGNN
from .multiplex_data import MultiplexSnapshot, MultiplexTemporalDataset

# COW codes for major powers (P5)
USA_CCODE = 2
CHN_CCODE = 710
MAJOR_POWER_CCODES = {
    2: "USA",
    200: "GBR",
    220: "FRA",
    365: "RUS",
    710: "CHN",
}


@dataclass
class EmbeddednessMetrics:
    """Embeddedness measures for a single state in the embedding space.

    Note on naming (post-simulation-validation, 2026-05-06):
      - `mean_cosine_similarity` measures EMBEDDING TYPICALITY — how
        similar the focal node's embedding is to the average of all
        other nodes. High values mean the focal looks like a "typical"
        country in the system. Removing a strong tie tends to INCREASE
        this (the focal becomes more generic), so it is NOT a measure
        of graph-theoretic isolation. Access this value via the
        equivalent property `centroid_proximity` for clearer naming
        in new code; the old field name is kept for backward compat.
      - `centroid_distance` measures TRUE ISOLATION — Euclidean
        distance from the focal to the centroid of all other
        embeddings. Removing a key tie pushes the focal away from
        the system center, so this value INCREASES under isolation.
        Use this metric when you want a graph-theoretic notion of
        "how peripheral did this country become."
    """
    ccode: int
    mean_cosine_similarity: float
    centroid_distance: float
    major_power_distance: float
    knn_density: float  # avg distance to k-nearest neighbors
    # Optional: cosine sim to specific partners
    partner_similarities: Optional[Dict[int, float]] = None

    @property
    def centroid_proximity(self) -> float:
        """Alias for `mean_cosine_similarity` with a more accurate name."""
        return self.mean_cosine_similarity


@dataclass
class CounterfactualResult:
    """Result of a single counterfactual edge intervention."""
    focal_ccode: int
    partner_ccode: int
    layer_name: str
    operation: str  # "add" or "remove"
    # Baseline metrics
    baseline_metrics: EmbeddednessMetrics
    # Counterfactual metrics
    counterfactual_metrics: EmbeddednessMetrics
    # Deltas
    delta_mean_cosine: float
    delta_centroid_distance: float
    delta_major_power_distance: float
    delta_knn_density: float


def compute_embeddedness_metrics(
    emb: torch.Tensor,
    focal_idx: int,
    ccode_to_idx: Dict[int, int],
    k_neighbors: int = 10,
) -> EmbeddednessMetrics:
    """Compute embeddedness metrics for a focal node.

    Parameters
    ----------
    emb : [num_nodes, emb_dim] tensor
    focal_idx : index of the focal node
    ccode_to_idx : mapping from country codes to indices
    k_neighbors : number of nearest neighbors for density measure
    """
    v = emb[focal_idx]
    num_nodes = emb.size(0)

    # 1. Mean cosine similarity to all other states
    norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = emb / norms
    v_normed = v / v.norm().clamp(min=1e-8)
    cos_sims = normed @ v_normed
    # Exclude self
    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[focal_idx] = False
    mean_cos = cos_sims[mask].mean().item()

    # 2. Distance to system centroid
    centroid = emb[mask].mean(dim=0)
    centroid_dist = torch.norm(v - centroid).item()

    # 3. Average distance to major powers
    mp_dists = []
    for mp_ccode in MAJOR_POWER_CCODES:
        if mp_ccode in ccode_to_idx:
            mp_idx = ccode_to_idx[mp_ccode]
            if mp_idx != focal_idx:
                mp_dists.append(torch.norm(v - emb[mp_idx]).item())
    major_power_dist = np.mean(mp_dists) if mp_dists else 0.0

    # 4. K-nearest neighbor density
    dists = torch.norm(emb - v.unsqueeze(0), dim=1)
    dists[focal_idx] = float("inf")  # exclude self
    k = min(k_neighbors, num_nodes - 1)
    topk_dists, _ = dists.topk(k, largest=False)
    knn_density = topk_dists.mean().item()

    # Focal ccode
    idx_to_ccode = {v_: k_ for k_, v_ in ccode_to_idx.items()}
    focal_ccode = idx_to_ccode.get(focal_idx, -1)

    return EmbeddednessMetrics(
        ccode=focal_ccode,
        mean_cosine_similarity=mean_cos,
        centroid_distance=centroid_dist,
        major_power_distance=major_power_dist,
        knn_density=knn_density,
    )


def simulate_single_edge(
    model: MultiplexTemporalGNN,
    dataset: MultiplexTemporalDataset,
    embedding_history: List[torch.Tensor],
    focal_ccode: int,
    partner_ccode: int,
    layer_name: str,
    operation: str,  # "add" or "remove"
    target_year_snap: MultiplexSnapshot,
    device: Optional[torch.device] = None,
) -> CounterfactualResult:
    """Simulate adding or removing one edge and measure the effect on focal state.

    This runs the full model forward pass with the modified graph to get
    accurate counterfactual predictions (not just a 1-hop proxy).

    Parameters
    ----------
    model : trained MultiplexTemporalGNN
    dataset : the dataset (for node mappings)
    embedding_history : list of [num_nodes, emb_dim] tensors for the temporal window
    focal_ccode : country code of the focal state
    partner_ccode : country code of the partner
    layer_name : which layer to modify
    operation : "add" or "remove"
    target_year_snap : the snapshot to modify
    device : torch device
    """
    if device is None:
        device = next(model.parameters()).device

    ccode_to_idx = dataset.ccode_to_idx
    focal_idx = ccode_to_idx[focal_ccode]
    partner_idx = ccode_to_idx[partner_ccode]

    model.eval()
    with torch.no_grad():
        # Baseline: predict from unmodified history
        baseline_pred = model.forward_temporal(embedding_history[-model.config.seq_len:])
        baseline_metrics = compute_embeddedness_metrics(
            baseline_pred, focal_idx, ccode_to_idx,
        )

        # Modify the target year's edge list for the specified layer
        snap = target_year_snap
        modified_ei = {}
        modified_ew = {}
        for ln in dataset.layer_names:
            modified_ei[ln] = snap.edge_indices[ln].clone().to(device)
            modified_ew[ln] = snap.edge_weights[ln].clone().to(device)

        if layer_name in modified_ei:
            ei = modified_ei[layer_name]
            ew = modified_ew[layer_name]

            if operation == "remove":
                # Remove edges between focal and partner (both directions)
                if ei.numel() > 0:
                    mask = ~(
                        ((ei[0] == focal_idx) & (ei[1] == partner_idx)) |
                        ((ei[0] == partner_idx) & (ei[1] == focal_idx))
                    )
                    modified_ei[layer_name] = ei[:, mask]
                    modified_ew[layer_name] = ew[mask]

            elif operation == "add":
                # Add edges in both directions
                new_edges = torch.tensor(
                    [[focal_idx, partner_idx], [partner_idx, focal_idx]],
                    dtype=torch.long, device=device,
                ).T
                new_weights = torch.ones(2, device=device)
                if ei.numel() > 0:
                    modified_ei[layer_name] = torch.cat([ei, new_edges], dim=1)
                    modified_ew[layer_name] = torch.cat([ew, new_weights])
                else:
                    modified_ei[layer_name] = new_edges
                    modified_ew[layer_name] = new_weights

        # Re-encode the modified snapshot
        nf = snap.node_features.to(device)
        modified_emb = model.encode_snapshot(nf, modified_ei, modified_ew, snap.layer_mask)

        # Replace the last entry in history with the modified embedding
        modified_history = list(embedding_history[:-1]) + [modified_emb]
        cf_pred = model.forward_temporal(modified_history[-model.config.seq_len:])

        cf_metrics = compute_embeddedness_metrics(cf_pred, focal_idx, ccode_to_idx)

    return CounterfactualResult(
        focal_ccode=focal_ccode,
        partner_ccode=partner_ccode,
        layer_name=layer_name,
        operation=operation,
        baseline_metrics=baseline_metrics,
        counterfactual_metrics=cf_metrics,
        delta_mean_cosine=cf_metrics.mean_cosine_similarity - baseline_metrics.mean_cosine_similarity,
        delta_centroid_distance=cf_metrics.centroid_distance - baseline_metrics.centroid_distance,
        delta_major_power_distance=cf_metrics.major_power_distance - baseline_metrics.major_power_distance,
        delta_knn_density=cf_metrics.knn_density - baseline_metrics.knn_density,
    )


def batch_counterfactual_analysis(
    model: MultiplexTemporalGNN,
    dataset: MultiplexTemporalDataset,
    focal_ccode: int,
    partner_ccodes: Optional[List[int]] = None,
    layer_names: Optional[List[str]] = None,
    operations: Optional[List[str]] = None,
    seq_len: int = 5,
    focal_year: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> List[CounterfactualResult]:
    """Run counterfactual analysis across multiple partners, layers, and operations.

    Parameters
    ----------
    model : trained MultiplexTemporalGNN
    dataset : MultiplexTemporalDataset
    focal_ccode : country code to analyze
    partner_ccodes : list of partner ccodes to test. If None, test all.
    layer_names : layers to modify. If None, test all available layers.
    operations : list of operations ("add", "remove"). If None, both.
    seq_len : temporal window length
    focal_year : the year to analyze. If None, use last year in dataset.
    device : torch device

    Returns
    -------
    List of CounterfactualResult sorted by abs(delta_mean_cosine) descending.
    """
    if device is None:
        device = next(model.parameters()).device

    if focal_year is None:
        focal_year = dataset.years[-1]

    if layer_names is None:
        layer_names = dataset.layer_names

    if operations is None:
        operations = ["add", "remove"]

    if partner_ccodes is None:
        partner_ccodes = [
            cc for cc in dataset.ccode_to_idx.keys()
            if cc != focal_ccode
        ]

    # Build embedding history up to focal year
    years = [y for y in dataset.years if y <= focal_year]
    if len(years) < seq_len:
        raise ValueError(f"Need at least {seq_len} years before focal year {focal_year}")

    model.eval()
    embedding_history: List[torch.Tensor] = []
    with torch.no_grad():
        for year in years[-seq_len:]:
            snap = dataset.snapshots[year]
            nf = snap.node_features.to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            emb = model.encode_snapshot(nf, ei, ew, snap.layer_mask)
            embedding_history.append(emb)

    target_snap = dataset.snapshots[focal_year]

    results: List[CounterfactualResult] = []
    total = len(partner_ccodes) * len(layer_names) * len(operations)
    done = 0

    for partner_ccode in partner_ccodes:
        if partner_ccode not in dataset.ccode_to_idx:
            continue
        for layer_name in layer_names:
            if not target_snap.layer_mask.get(layer_name, False):
                continue
            for op in operations:
                result = simulate_single_edge(
                    model=model,
                    dataset=dataset,
                    embedding_history=embedding_history,
                    focal_ccode=focal_ccode,
                    partner_ccode=partner_ccode,
                    layer_name=layer_name,
                    operation=op,
                    target_year_snap=target_snap,
                    device=device,
                )
                results.append(result)
                done += 1
                if done % 100 == 0:
                    print(f"  Counterfactual analysis: {done}/{total} simulations done")

    # Sort by absolute effect size on mean cosine similarity
    results.sort(key=lambda r: abs(r.delta_mean_cosine), reverse=True)
    return results


def results_to_dataframe(results: List[CounterfactualResult]) -> pd.DataFrame:
    """Convert counterfactual results to a DataFrame for export."""
    rows = []
    for r in results:
        rows.append({
            "focal_ccode": r.focal_ccode,
            "partner_ccode": r.partner_ccode,
            "layer": r.layer_name,
            "operation": r.operation,
            "baseline_mean_cosine": r.baseline_metrics.mean_cosine_similarity,
            "cf_mean_cosine": r.counterfactual_metrics.mean_cosine_similarity,
            "delta_mean_cosine": r.delta_mean_cosine,
            "baseline_centroid_dist": r.baseline_metrics.centroid_distance,
            "cf_centroid_dist": r.counterfactual_metrics.centroid_distance,
            "delta_centroid_dist": r.delta_centroid_distance,
            "baseline_mp_dist": r.baseline_metrics.major_power_distance,
            "cf_mp_dist": r.counterfactual_metrics.major_power_distance,
            "delta_mp_dist": r.delta_major_power_distance,
            "baseline_knn_density": r.baseline_metrics.knn_density,
            "cf_knn_density": r.counterfactual_metrics.knn_density,
            "delta_knn_density": r.delta_knn_density,
        })
    return pd.DataFrame(rows)


def summarize_by_layer(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize counterfactual effects by layer and operation."""
    return results_df.groupby(["layer", "operation"]).agg(
        mean_delta_cosine=("delta_mean_cosine", "mean"),
        abs_mean_delta_cosine=("delta_mean_cosine", lambda x: x.abs().mean()),
        max_delta_cosine=("delta_mean_cosine", "max"),
        min_delta_cosine=("delta_mean_cosine", "min"),
        mean_delta_centroid=("delta_centroid_dist", "mean"),
        n_simulations=("delta_mean_cosine", "count"),
    ).reset_index()


def top_interventions(
    results_df: pd.DataFrame,
    n: int = 20,
    metric: str = "delta_mean_cosine",
) -> pd.DataFrame:
    """Return the top N interventions by absolute effect size."""
    df = results_df.copy()
    df["abs_effect"] = df[metric].abs()
    return df.nlargest(n, "abs_effect").drop(columns=["abs_effect"])
