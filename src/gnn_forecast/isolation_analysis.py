"""Isolation analysis: paired USA–China counterfactual visualization.

Runs each edge perturbation once and measures the effect on BOTH USA and
China simultaneously.  This enables:

1. Dual-effect scatter: x = Δ USA embeddedness, y = Δ China embeddedness.
   Quadrants reveal asymmetric effects (edges that embed one power while
   isolating the other).

2. Isolation rankings: which edges most isolate each power?

3. Layer-level heatmaps: which layer types are most consequential for
   each power's isolation?

4. "Wedge" edges: edges where the effects on USA and China are most
   divergent (opposite sign), indicating strategic leverage points.

5. Multi-edge combinatorial analysis: test sets of edges together to
   capture interaction effects (e.g., "what if the US loses both its
   India trade tie and its India defense agreement?").

All outputs are DataFrames (CSV-ready) plus matplotlib figures when
matplotlib is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .multiplex_model import MultiplexTemporalGNN
from .multiplex_data import MultiplexSnapshot, MultiplexTemporalDataset
from .counterfactual import (
    compute_embeddedness_metrics,
    USA_CCODE,
    CHN_CCODE,
    MAJOR_POWER_CCODES,
)


# ---------------------------------------------------------------
# Core: dual-focal simulation
# ---------------------------------------------------------------

@dataclass
class DualFocalResult:
    """Effect of a single edge perturbation on both USA and China."""
    partner_ccode: int
    layer_name: str
    operation: str  # "add" or "remove"
    # USA deltas (positive = more embedded, negative = more isolated)
    usa_delta_cosine: float
    usa_delta_centroid: float
    usa_delta_mp_dist: float
    usa_delta_knn: float
    # China deltas
    chn_delta_cosine: float
    chn_delta_centroid: float
    chn_delta_mp_dist: float
    chn_delta_knn: float
    # Derived
    wedge_cosine: float   # usa_delta - chn_delta (positive = favors USA)
    wedge_centroid: float


def _modify_snapshot_edge(
    snap: MultiplexSnapshot,
    layer_name: str,
    src_idx: int,
    tgt_idx: int,
    operation: str,
    device: torch.device,
) -> Tuple[Dict[str, torch.LongTensor], Dict[str, torch.FloatTensor]]:
    """Clone a snapshot's edges and apply a single add/remove."""
    modified_ei = {}
    modified_ew = {}
    for ln in snap.edge_indices:
        modified_ei[ln] = snap.edge_indices[ln].clone().to(device)
        modified_ew[ln] = snap.edge_weights[ln].clone().to(device)

    if layer_name not in modified_ei:
        return modified_ei, modified_ew

    ei = modified_ei[layer_name]
    ew = modified_ew[layer_name]

    if operation == "remove":
        if ei.numel() > 0:
            mask = ~(
                ((ei[0] == src_idx) & (ei[1] == tgt_idx)) |
                ((ei[0] == tgt_idx) & (ei[1] == src_idx))
            )
            modified_ei[layer_name] = ei[:, mask]
            modified_ew[layer_name] = ew[mask]
    elif operation == "add":
        new_edges = torch.tensor(
            [[src_idx, tgt_idx], [tgt_idx, src_idx]],
            dtype=torch.long, device=device,
        ).T
        new_weights = torch.ones(2, device=device)
        if ei.numel() > 0:
            modified_ei[layer_name] = torch.cat([ei, new_edges], dim=1)
            modified_ew[layer_name] = torch.cat([ew, new_weights])
        else:
            modified_ei[layer_name] = new_edges
            modified_ew[layer_name] = new_weights

    return modified_ei, modified_ew


def _modify_snapshot_multi_edge(
    snap: MultiplexSnapshot,
    interventions: List[Tuple[str, int, int, str]],
    device: torch.device,
) -> Tuple[Dict[str, torch.LongTensor], Dict[str, torch.FloatTensor]]:
    """Apply multiple edge interventions to a snapshot.

    Parameters
    ----------
    interventions : list of (layer_name, src_idx, tgt_idx, operation)
    """
    modified_ei = {}
    modified_ew = {}
    for ln in snap.edge_indices:
        modified_ei[ln] = snap.edge_indices[ln].clone().to(device)
        modified_ew[ln] = snap.edge_weights[ln].clone().to(device)

    for layer_name, src_idx, tgt_idx, operation in interventions:
        if layer_name not in modified_ei:
            continue

        ei = modified_ei[layer_name]
        ew = modified_ew[layer_name]

        if operation == "remove":
            if ei.numel() > 0:
                mask = ~(
                    ((ei[0] == src_idx) & (ei[1] == tgt_idx)) |
                    ((ei[0] == tgt_idx) & (ei[1] == src_idx))
                )
                modified_ei[layer_name] = ei[:, mask]
                modified_ew[layer_name] = ew[mask]
        elif operation == "add":
            new_edges = torch.tensor(
                [[src_idx, tgt_idx], [tgt_idx, src_idx]],
                dtype=torch.long, device=device,
            ).T
            new_weights = torch.ones(2, device=device)
            if ei.numel() > 0:
                modified_ei[layer_name] = torch.cat([ei, new_edges], dim=1)
                modified_ew[layer_name] = torch.cat([ew, new_weights])
            else:
                modified_ei[layer_name] = new_edges
                modified_ew[layer_name] = new_weights

    return modified_ei, modified_ew


def dual_focal_simulation(
    model: MultiplexTemporalGNN,
    dataset: MultiplexTemporalDataset,
    partner_ccodes: Optional[List[int]] = None,
    layer_names: Optional[List[str]] = None,
    operations: Optional[List[str]] = None,
    seq_len: int = 5,
    focal_year: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> List[DualFocalResult]:
    """Run counterfactual analysis measuring effects on both USA and China.

    For each (partner, layer, operation): disconnect or fully connect the
    partner in that layer, then measure the resulting embedding change
    for both USA and China in the predicted next-period embedding.

    The perturbation is: remove (or add) ALL of the partner state's edges
    in the given layer. This captures the full system-wide effect of that
    state's participation in the layer — not just a single bilateral tie —
    which propagates through the GNN's message passing to affect every
    other state's embedding, including USA and China.

    This is more efficient than running batch_counterfactual_analysis twice
    because each perturbation only requires one forward pass to get both
    USA and China effects.
    """
    if device is None:
        device = next(model.parameters()).device

    if focal_year is None:
        focal_year = dataset.years[-1]
    if layer_names is None:
        layer_names = dataset.layer_names
    if operations is None:
        operations = ["add", "remove"]

    ccode_to_idx = dataset.ccode_to_idx

    if USA_CCODE not in ccode_to_idx or CHN_CCODE not in ccode_to_idx:
        raise ValueError("Both USA (2) and China (710) must be in the dataset")

    usa_idx = ccode_to_idx[USA_CCODE]
    chn_idx = ccode_to_idx[CHN_CCODE]

    if partner_ccodes is None:
        partner_ccodes = [
            cc for cc in ccode_to_idx.keys()
            if cc not in (USA_CCODE, CHN_CCODE)
        ]

    # Build embedding history
    years = [y for y in dataset.years if y <= focal_year]
    if len(years) < seq_len:
        raise ValueError(f"Need >= {seq_len} years, got {len(years)}")

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

    # Baseline prediction (one forward pass, reused for all perturbations)
    with torch.no_grad():
        baseline_pred = model.forward_temporal(embedding_history[-seq_len:])
        usa_baseline = compute_embeddedness_metrics(baseline_pred, usa_idx, ccode_to_idx)
        chn_baseline = compute_embeddedness_metrics(baseline_pred, chn_idx, ccode_to_idx)

    results: List[DualFocalResult] = []
    total = len(partner_ccodes) * len(layer_names) * len(operations)
    done = 0

    with torch.no_grad():
        for partner_ccode in partner_ccodes:
            if partner_ccode not in ccode_to_idx:
                continue
            partner_idx = ccode_to_idx[partner_ccode]

            for layer_name in layer_names:
                if not target_snap.layer_mask.get(layer_name, False):
                    continue

                for op in operations:
                    # Clone all layer edges
                    mod_ei = {}
                    mod_ew = {}
                    for ln in target_snap.edge_indices:
                        mod_ei[ln] = target_snap.edge_indices[ln].clone().to(device)
                        mod_ew[ln] = target_snap.edge_weights[ln].clone().to(device)

                    ei_l = mod_ei[layer_name]
                    ew_l = mod_ew[layer_name]

                    if op == "remove" and ei_l.numel() > 0:
                        # Remove ALL edges involving this partner in this layer
                        keep = ~((ei_l[0] == partner_idx) | (ei_l[1] == partner_idx))
                        mod_ei[layer_name] = ei_l[:, keep]
                        mod_ew[layer_name] = ew_l[keep]

                    elif op == "add":
                        # Find which nodes the partner is NOT connected to,
                        # add edges to a sample of them
                        existing = set()
                        if ei_l.numel() > 0:
                            src_mask = ei_l[0] == partner_idx
                            existing.update(ei_l[1, src_mask].cpu().tolist())
                            tgt_mask = ei_l[1] == partner_idx
                            existing.update(ei_l[0, tgt_mask].cpu().tolist())

                        unconnected = [
                            n for n in range(dataset.num_nodes)
                            if n != partner_idx and n not in existing
                        ]
                        # Sample up to 50 new connections
                        new_targets = unconnected[:50]
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

                    # Re-encode with modified graph, predict next period
                    nf = target_snap.node_features.to(device)
                    mod_emb = model.encode_snapshot(
                        nf, mod_ei, mod_ew, target_snap.layer_mask,
                    )
                    mod_history = list(embedding_history[:-1]) + [mod_emb]
                    cf_pred = model.forward_temporal(mod_history[-seq_len:])

                    # Measure effect on both USA and China
                    usa_cf = compute_embeddedness_metrics(cf_pred, usa_idx, ccode_to_idx)
                    chn_cf = compute_embeddedness_metrics(cf_pred, chn_idx, ccode_to_idx)

                    usa_d_cos = usa_cf.mean_cosine_similarity - usa_baseline.mean_cosine_similarity
                    usa_d_cen = usa_cf.centroid_distance - usa_baseline.centroid_distance
                    usa_d_mp = usa_cf.major_power_distance - usa_baseline.major_power_distance
                    usa_d_knn = usa_cf.knn_density - usa_baseline.knn_density

                    chn_d_cos = chn_cf.mean_cosine_similarity - chn_baseline.mean_cosine_similarity
                    chn_d_cen = chn_cf.centroid_distance - chn_baseline.centroid_distance
                    chn_d_mp = chn_cf.major_power_distance - chn_baseline.major_power_distance
                    chn_d_knn = chn_cf.knn_density - chn_baseline.knn_density

                    results.append(DualFocalResult(
                        partner_ccode=partner_ccode,
                        layer_name=layer_name,
                        operation=op,
                        usa_delta_cosine=usa_d_cos,
                        usa_delta_centroid=usa_d_cen,
                        usa_delta_mp_dist=usa_d_mp,
                        usa_delta_knn=usa_d_knn,
                        chn_delta_cosine=chn_d_cos,
                        chn_delta_centroid=chn_d_cen,
                        chn_delta_mp_dist=chn_d_mp,
                        chn_delta_knn=chn_d_knn,
                        wedge_cosine=usa_d_cos - chn_d_cos,
                        wedge_centroid=usa_d_cen - chn_d_cen,
                    ))

                    done += 1
                    if done % 100 == 0:
                        print(f"  Dual-focal analysis: {done}/{total}")

    # Sort by wedge magnitude (most asymmetric first)
    results.sort(key=lambda r: abs(r.wedge_cosine), reverse=True)
    return results


# ---------------------------------------------------------------
# Multi-edge combinatorial analysis
# ---------------------------------------------------------------

@dataclass
class MultiEdgeResult:
    """Effect of a set of simultaneous edge changes on USA and China."""
    label: str
    interventions: List[Tuple[str, int, int, str]]
    usa_delta_cosine: float
    usa_delta_centroid: float
    chn_delta_cosine: float
    chn_delta_centroid: float
    wedge_cosine: float


def multi_edge_simulation(
    model: MultiplexTemporalGNN,
    dataset: MultiplexTemporalDataset,
    intervention_sets: Dict[str, List[Tuple[int, str, str]]],
    seq_len: int = 5,
    focal_year: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> List[MultiEdgeResult]:
    """Test multiple simultaneous edge changes.

    Parameters
    ----------
    intervention_sets : dict mapping label -> list of (partner_ccode, layer, operation)
        Example: {
            "US loses India trade + defense": [
                (750, "trade", "remove"),
                (750, "defensive_alliances", "remove"),
            ],
            "China gains Brazil trade + IGO": [
                (140, "trade", "add"),
                (140, "igo", "add"),
            ],
        }
    """
    if device is None:
        device = next(model.parameters()).device
    if focal_year is None:
        focal_year = dataset.years[-1]

    ccode_to_idx = dataset.ccode_to_idx
    usa_idx = ccode_to_idx[USA_CCODE]
    chn_idx = ccode_to_idx[CHN_CCODE]

    # Build history
    years = [y for y in dataset.years if y <= focal_year]
    model.eval()
    embedding_history: List[torch.Tensor] = []
    with torch.no_grad():
        for year in years[-seq_len:]:
            snap = dataset.snapshots[year]
            nf = snap.node_features.to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            embedding_history.append(model.encode_snapshot(nf, ei, ew, snap.layer_mask))

    target_snap = dataset.snapshots[focal_year]

    # Baseline
    with torch.no_grad():
        baseline_pred = model.forward_temporal(embedding_history[-seq_len:])
        usa_base = compute_embeddedness_metrics(baseline_pred, usa_idx, ccode_to_idx)
        chn_base = compute_embeddedness_metrics(baseline_pred, chn_idx, ccode_to_idx)

    results = []
    with torch.no_grad():
        for label, interventions in intervention_sets.items():
            # Convert ccodes to indices
            idx_interventions = []
            for partner_cc, layer, op in interventions:
                if partner_cc in ccode_to_idx:
                    pidx = ccode_to_idx[partner_cc]
                    idx_interventions.append((layer, pidx, pidx, op))

            # For multi-edge: we need src and tgt for each edge.
            # The current design removes ALL edges of the partner in that layer,
            # so we handle it differently:
            mod_ei = {}
            mod_ew = {}
            for ln in target_snap.edge_indices:
                mod_ei[ln] = target_snap.edge_indices[ln].clone().to(device)
                mod_ew[ln] = target_snap.edge_weights[ln].clone().to(device)

            for partner_cc, layer, op in interventions:
                if partner_cc not in ccode_to_idx or layer not in mod_ei:
                    continue
                pidx = ccode_to_idx[partner_cc]
                ei_l = mod_ei[layer]
                ew_l = mod_ew[layer]

                if op == "remove" and ei_l.numel() > 0:
                    mask = ~((ei_l[0] == pidx) | (ei_l[1] == pidx))
                    mod_ei[layer] = ei_l[:, mask]
                    mod_ew[layer] = ew_l[mask]
                elif op == "add":
                    new_targets = list(range(min(50, dataset.num_nodes)))
                    new_targets = [t for t in new_targets if t != pidx]
                    if new_targets:
                        src = [pidx] * len(new_targets) + new_targets
                        tgt = new_targets + [pidx] * len(new_targets)
                        new_e = torch.tensor([src, tgt], dtype=torch.long, device=device)
                        new_w = torch.ones(len(src), device=device)
                        if ei_l.numel() > 0:
                            mod_ei[layer] = torch.cat([ei_l, new_e], dim=1)
                            mod_ew[layer] = torch.cat([ew_l, new_w])
                        else:
                            mod_ei[layer] = new_e
                            mod_ew[layer] = new_w

            nf = target_snap.node_features.to(device)
            mod_emb = model.encode_snapshot(nf, mod_ei, mod_ew, target_snap.layer_mask)
            mod_history = list(embedding_history[:-1]) + [mod_emb]
            cf_pred = model.forward_temporal(mod_history[-seq_len:])

            usa_cf = compute_embeddedness_metrics(cf_pred, usa_idx, ccode_to_idx)
            chn_cf = compute_embeddedness_metrics(cf_pred, chn_idx, ccode_to_idx)

            usa_d = usa_cf.mean_cosine_similarity - usa_base.mean_cosine_similarity
            chn_d = chn_cf.mean_cosine_similarity - chn_base.mean_cosine_similarity

            results.append(MultiEdgeResult(
                label=label,
                interventions=[(l, ccode_to_idx.get(cc, -1), ccode_to_idx.get(cc, -1), o)
                               for cc, l, o in interventions],
                usa_delta_cosine=usa_d,
                usa_delta_centroid=usa_cf.centroid_distance - usa_base.centroid_distance,
                chn_delta_cosine=chn_d,
                chn_delta_centroid=chn_cf.centroid_distance - chn_base.centroid_distance,
                wedge_cosine=usa_d - chn_d,
            ))

    return results


# ---------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------

def dual_results_to_dataframe(results: List[DualFocalResult]) -> pd.DataFrame:
    """Convert dual-focal results to a DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "partner_ccode": r.partner_ccode,
            "layer": r.layer_name,
            "operation": r.operation,
            "usa_delta_cosine": r.usa_delta_cosine,
            "usa_delta_centroid": r.usa_delta_centroid,
            "usa_delta_mp_dist": r.usa_delta_mp_dist,
            "usa_delta_knn": r.usa_delta_knn,
            "chn_delta_cosine": r.chn_delta_cosine,
            "chn_delta_centroid": r.chn_delta_centroid,
            "chn_delta_mp_dist": r.chn_delta_mp_dist,
            "chn_delta_knn": r.chn_delta_knn,
            "wedge_cosine": r.wedge_cosine,
            "wedge_centroid": r.wedge_centroid,
            "abs_wedge": abs(r.wedge_cosine),
        })
    return pd.DataFrame(rows)


def multi_edge_results_to_dataframe(results: List[MultiEdgeResult]) -> pd.DataFrame:
    """Convert multi-edge results to a DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "scenario": r.label,
            "usa_delta_cosine": r.usa_delta_cosine,
            "usa_delta_centroid": r.usa_delta_centroid,
            "chn_delta_cosine": r.chn_delta_cosine,
            "chn_delta_centroid": r.chn_delta_centroid,
            "wedge_cosine": r.wedge_cosine,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Quadrant classification
# ---------------------------------------------------------------

def classify_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'quadrant' column classifying each edge perturbation.

    Quadrants (based on delta_cosine — positive = more embedded):
      Q1: USA embedded, China embedded     (both gain)
      Q2: USA isolated, China embedded     (China gains at US expense)
      Q3: USA isolated, China isolated     (both lose)
      Q4: USA embedded, China isolated     (US gains at China expense)
    """
    df = df.copy()

    def _classify(row):
        u = row["usa_delta_cosine"]
        c = row["chn_delta_cosine"]
        if u >= 0 and c >= 0:
            return "Q1: Both embedded"
        elif u < 0 and c >= 0:
            return "Q2: USA isolated, China embedded"
        elif u < 0 and c < 0:
            return "Q3: Both isolated"
        else:
            return "Q4: USA embedded, China isolated"

    df["quadrant"] = df.apply(_classify, axis=1)
    return df


def quadrant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize edge perturbations by quadrant and layer."""
    if "quadrant" not in df.columns:
        df = classify_quadrants(df)

    return df.groupby(["quadrant", "layer", "operation"]).agg(
        n=("partner_ccode", "count"),
        mean_usa_delta=("usa_delta_cosine", "mean"),
        mean_chn_delta=("chn_delta_cosine", "mean"),
        mean_wedge=("wedge_cosine", "mean"),
        max_abs_wedge=("abs_wedge", "max"),
    ).reset_index().sort_values("max_abs_wedge", ascending=False)


# ---------------------------------------------------------------
# Top isolating edges
# ---------------------------------------------------------------

def top_usa_isolating(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Edges that most isolate the USA (most negative usa_delta_cosine)."""
    return df.nsmallest(n, "usa_delta_cosine")


def top_china_isolating(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Edges that most isolate China (most negative chn_delta_cosine)."""
    return df.nsmallest(n, "chn_delta_cosine")


def top_wedge_edges(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Edges with the most asymmetric USA-vs-China effect."""
    return df.nlargest(n, "abs_wedge")


def top_pro_usa(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Edges that most favor USA over China (positive wedge)."""
    return df.nlargest(n, "wedge_cosine")


def top_pro_china(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Edges that most favor China over USA (negative wedge)."""
    return df.nsmallest(n, "wedge_cosine")


# ---------------------------------------------------------------
# Layer-level heatmap data
# ---------------------------------------------------------------

def layer_isolation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate isolation effects by layer x operation.

    Returns a table suitable for rendering as a heatmap with layers as rows
    and metrics as columns.
    """
    return df.groupby(["layer", "operation"]).agg(
        mean_usa_delta=("usa_delta_cosine", "mean"),
        mean_chn_delta=("chn_delta_cosine", "mean"),
        mean_wedge=("wedge_cosine", "mean"),
        std_usa_delta=("usa_delta_cosine", "std"),
        std_chn_delta=("chn_delta_cosine", "std"),
        max_usa_isolation=("usa_delta_cosine", "min"),
        max_chn_isolation=("chn_delta_cosine", "min"),
        n=("partner_ccode", "count"),
    ).reset_index()


# ---------------------------------------------------------------
# Matplotlib figures
# ---------------------------------------------------------------

def plot_isolation_scatter(
    df: pd.DataFrame,
    out_path: Optional[str | Path] = None,
    title: str = "Edge Perturbation Effects on USA vs China Embeddedness",
) -> Optional[object]:
    """Scatter plot: x = USA Δ cosine, y = China Δ cosine.

    Each point is one (partner, layer, operation) perturbation.
    Points are colored by layer, shaped by operation.
    Quadrant lines at (0, 0) divide embed/isolate regions.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — scatter plot skipped")
        return None

    if "quadrant" not in df.columns:
        df = classify_quadrants(df)

    fig, ax = plt.subplots(figsize=(10, 8))

    layers = df["layer"].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(layers), 3)))
    layer_colors = {l: colors[i] for i, l in enumerate(layers)}

    markers = {"add": "^", "remove": "v"}

    for layer in layers:
        for op in df["operation"].unique():
            subset = df[(df["layer"] == layer) & (df["operation"] == op)]
            if subset.empty:
                continue
            ax.scatter(
                subset["usa_delta_cosine"],
                subset["chn_delta_cosine"],
                c=[layer_colors[layer]],
                marker=markers.get(op, "o"),
                alpha=0.6,
                s=40,
                label=f"{layer} ({op})",
                edgecolors="white",
                linewidths=0.3,
            )

    # Quadrant lines and labels
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    pad_x = (xlim[1] - xlim[0]) * 0.02
    pad_y = (ylim[1] - ylim[0]) * 0.02

    ax.text(xlim[1] - pad_x, ylim[1] - pad_y, "Q1: Both embedded",
            ha="right", va="top", fontsize=8, color="gray")
    ax.text(xlim[0] + pad_x, ylim[1] - pad_y, "Q2: USA isolated\nChina embedded",
            ha="left", va="top", fontsize=8, color="gray")
    ax.text(xlim[0] + pad_x, ylim[0] + pad_y, "Q3: Both isolated",
            ha="left", va="bottom", fontsize=8, color="gray")
    ax.text(xlim[1] - pad_x, ylim[0] + pad_y, "Q4: USA embedded\nChina isolated",
            ha="right", va="bottom", fontsize=8, color="gray")

    ax.set_xlabel("USA Δ Mean Cosine Similarity (→ more embedded)")
    ax.set_ylabel("China Δ Mean Cosine Similarity (↑ more embedded)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  Saved scatter: {out_path}")
    return fig


def plot_layer_heatmap(
    df: pd.DataFrame,
    out_path: Optional[str | Path] = None,
    title: str = "Mean Isolation Effect by Layer",
) -> Optional[object]:
    """Heatmap: layers × {USA delta, China delta, wedge}."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — heatmap skipped")
        return None

    hm = layer_isolation_heatmap(df)
    # Pivot for removal effects (most interpretable)
    remove_only = hm[hm["operation"] == "remove"].copy()
    if remove_only.empty:
        remove_only = hm.copy()

    layers = remove_only["layer"].values
    metrics = np.column_stack([
        remove_only["mean_usa_delta"].values,
        remove_only["mean_chn_delta"].values,
        remove_only["mean_wedge"].values,
    ])
    metric_names = ["USA Δ cosine", "China Δ cosine", "Wedge (USA − China)"]

    fig, ax = plt.subplots(figsize=(8, max(3, len(layers) * 0.6 + 1)))
    vmax = np.abs(metrics).max()
    im = ax.imshow(metrics, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=30, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)

    # Annotate cells
    for i in range(len(layers)):
        for j in range(len(metric_names)):
            ax.text(j, i, f"{metrics[i, j]:.4f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(metrics[i, j]) > vmax * 0.5 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  Saved heatmap: {out_path}")
    return fig


def plot_wedge_bar(
    df: pd.DataFrame,
    n: int = 20,
    out_path: Optional[str | Path] = None,
    title: str = "Top Asymmetric Edges (USA vs China)",
) -> Optional[object]:
    """Horizontal bar chart of the most asymmetric edge perturbations.

    Bars go left (favors China) or right (favors USA).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — bar chart skipped")
        return None

    top = top_wedge_edges(df, n).copy()
    top["label"] = top.apply(
        lambda r: f"ccode {int(r['partner_ccode'])} | {r['layer']} | {r['operation']}",
        axis=1,
    )
    top = top.sort_values("wedge_cosine")

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.35)))
    colors = ["#2166ac" if w > 0 else "#b2182b" for w in top["wedge_cosine"]]
    ax.barh(range(len(top)), top["wedge_cosine"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Wedge: USA Δcosine − China Δcosine\n(← favors China | favors USA →)")
    ax.set_title(title)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  Saved bar chart: {out_path}")
    return fig


def plot_multi_edge_scenarios(
    results: List[MultiEdgeResult],
    out_path: Optional[str | Path] = None,
    title: str = "Multi-Edge Scenario Comparison",
) -> Optional[object]:
    """Grouped bar chart comparing multi-edge scenarios.

    For each scenario, shows USA delta, China delta, and wedge side by side.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — multi-edge plot skipped")
        return None

    labels = [r.label for r in results]
    usa_vals = [r.usa_delta_cosine for r in results]
    chn_vals = [r.chn_delta_cosine for r in results]
    wedge_vals = [r.wedge_cosine for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.bar(x - width, usa_vals, width, label="USA Δ cosine", color="#2166ac")
    ax.bar(x, chn_vals, width, label="China Δ cosine", color="#b2182b")
    ax.bar(x + width, wedge_vals, width, label="Wedge (USA − China)", color="#7fbc41")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ Mean Cosine Similarity")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  Saved scenario chart: {out_path}")
    return fig


# ---------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------

def run_isolation_analysis(
    model: MultiplexTemporalGNN,
    dataset: MultiplexTemporalDataset,
    out_dir: str | Path,
    partner_ccodes: Optional[List[int]] = None,
    seq_len: int = 5,
    focal_year: Optional[int] = None,
    multi_edge_scenarios: Optional[Dict[str, List[Tuple[int, str, str]]]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, pd.DataFrame]:
    """Run the full isolation analysis pipeline and save all outputs.

    Parameters
    ----------
    model : trained MultiplexTemporalGNN
    dataset : MultiplexTemporalDataset
    out_dir : directory for outputs
    partner_ccodes : which partners to test (None = all non-USA/non-China)
    seq_len : temporal window
    focal_year : year to analyze (None = last year)
    multi_edge_scenarios : named sets of simultaneous edge interventions.
        Example: {"US loses India": [(750, "trade", "remove"),
                                     (750, "defensive_alliances", "remove")]}
    device : torch device

    Returns
    -------
    Dict of output table name -> DataFrame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    print("=" * 60)
    print("ISOLATION ANALYSIS: USA vs China Edge Effects")
    print("=" * 60)

    # 1. Dual-focal simulation
    print("\n1. Running dual-focal edge simulations...")
    dual_results = dual_focal_simulation(
        model, dataset, partner_ccodes=partner_ccodes,
        seq_len=seq_len, focal_year=focal_year, device=device,
    )
    df = dual_results_to_dataframe(dual_results)
    df = classify_quadrants(df)
    df.to_csv(out_dir / "dual_focal_all.csv", index=False)
    outputs["dual_focal_all"] = df
    print(f"   {len(df)} perturbations analyzed")

    # 2. Quadrant summary
    print("\n2. Quadrant classification...")
    qs = quadrant_summary(df)
    qs.to_csv(out_dir / "quadrant_summary.csv", index=False)
    outputs["quadrant_summary"] = qs
    print(qs[["quadrant", "layer", "n", "mean_wedge"]].to_string(index=False))

    # 3. Top isolating edges
    print("\n3. Top isolating edges...")
    usa_iso = top_usa_isolating(df, 30)
    usa_iso.to_csv(out_dir / "top_usa_isolating.csv", index=False)
    outputs["top_usa_isolating"] = usa_iso

    chn_iso = top_china_isolating(df, 30)
    chn_iso.to_csv(out_dir / "top_china_isolating.csv", index=False)
    outputs["top_china_isolating"] = chn_iso

    wedge = top_wedge_edges(df, 30)
    wedge.to_csv(out_dir / "top_wedge_edges.csv", index=False)
    outputs["top_wedge_edges"] = wedge

    pro_usa = top_pro_usa(df, 30)
    pro_usa.to_csv(out_dir / "top_pro_usa.csv", index=False)
    outputs["top_pro_usa"] = pro_usa

    pro_chn = top_pro_china(df, 30)
    pro_chn.to_csv(out_dir / "top_pro_china.csv", index=False)
    outputs["top_pro_china"] = pro_chn

    # 4. Layer heatmap
    print("\n4. Layer-level heatmap data...")
    hm = layer_isolation_heatmap(df)
    hm.to_csv(out_dir / "layer_isolation_heatmap.csv", index=False)
    outputs["layer_heatmap"] = hm

    # 5. Figures
    print("\n5. Generating figures...")
    plot_isolation_scatter(df, out_dir / "scatter_usa_vs_china.png")
    plot_layer_heatmap(df, out_dir / "heatmap_layer_isolation.png")
    plot_wedge_bar(df, 30, out_dir / "bar_top_wedge_edges.png")

    # 6. Multi-edge scenarios (if provided)
    if multi_edge_scenarios:
        print(f"\n6. Multi-edge scenarios ({len(multi_edge_scenarios)} scenarios)...")
        me_results = multi_edge_simulation(
            model, dataset, multi_edge_scenarios,
            seq_len=seq_len, focal_year=focal_year, device=device,
        )
        me_df = multi_edge_results_to_dataframe(me_results)
        me_df.to_csv(out_dir / "multi_edge_scenarios.csv", index=False)
        outputs["multi_edge"] = me_df
        plot_multi_edge_scenarios(me_results, out_dir / "bar_multi_edge.png")
        print(me_df.to_string(index=False))

    print(f"\nSaved {len(outputs)} tables and figures to {out_dir}")
    return outputs
