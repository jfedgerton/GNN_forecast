"""Feature-intervention counterfactual machinery.

Counterpart to `interventions.py` (which perturbs edges). This module
perturbs a single (focal_country, feature, year_range) cell of the
node-feature panel, re-encodes the entire history at the same frozen
encoder weights, and reports the per-hop cascade of embedding shifts.

The substantive question this answers is "what if country i's domestic
configuration were different?" — the natural sibling of the edge
intervention's "what if relation (i, j, ell) didn't exist?".

Both intervention types ride on the same R-GCN encoder because the
encoder takes both edges and node features as inputs. AME and latent-
space models can't do feature interventions cleanly because they have
no parameterized propagation operator.

Design principles (concrete, sequential implementations, seed=123):
  - The encoder weights are frozen during intervention. We never retrain
    on the perturbed inputs; we only re-evaluate at frozen weights.
  - Feature values entering this module are in RAW units (e.g., polity2
    = -10), and we re-apply the NodeFeatureSet's z-score statistics
    before substitution. This means callers don't need to know the
    z-score parameters — they specify domain-meaningful values.
  - Cascade hops are measured on the merged-multiplex alliance graph at
    the year of the perturbation start. We use undirected BFS from the
    focal node.
  - Centroid distance is the headline metric. Per-hop deltas are
    reported as both (a) mean absolute change in focal-relative centroid
    distance and (b) mean L2 displacement of embedding vector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from gnn_forecast.heterogeneous_model import HeterogeneousEncoder
from gnn_forecast.diagnostic_v3 import MultiplexTemporalDataset
from gnn_forecast.node_features import NodeFeatureSet

SEED = 123


# ---------------------------------------------------------------
# Configuration object
# ---------------------------------------------------------------

@dataclass
class FeatureInterventionConfig:
    """One regime-shock scenario.

    Attributes
    ----------
    focal_ccode : int
        COW code of the focal country.
    feature_name : str
        Column name in the NodeFeatureSet (e.g., 'polity2', 'cinc').
    target_value_raw : float
        Counterfactual value in RAW (un-z-scored) units. The module
        re-applies the NodeFeatureSet's z-score stats internally.
    year_range : tuple[int, int]
        Inclusive (start, end) of the perturbation window.
    label : str
        Short human-readable scenario name for output rows.
    """
    focal_ccode: int
    feature_name: str
    target_value_raw: float
    year_range: Tuple[int, int]
    label: str


@dataclass
class CascadeResult:
    """Per-hop cascade decomposition for one scenario."""
    scenario_label: str
    focal_ccode: int
    focal_idx: int
    feature_name: str
    target_value_raw: float
    year_range: Tuple[int, int]
    # per-hop deltas at the final year of the panel
    hop_0_focal_displacement: float
    hop_1_mean_displacement: float
    hop_2_mean_displacement: float
    hop_3plus_mean_displacement: float
    hop_0_centroid_delta: float
    hop_1_mean_centroid_delta: float
    hop_2_mean_centroid_delta: float
    hop_3plus_mean_centroid_delta: float
    n_hop_1: int
    n_hop_2: int
    n_hop_3plus: int
    # per-state per-year embedding shift table (long-form), for downstream plots
    per_state_table: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------
# Feature perturbation
# ---------------------------------------------------------------

def apply_feature_intervention(
    feat_set: NodeFeatureSet,
    dataset: MultiplexTemporalDataset,
    config: FeatureInterventionConfig,
) -> Dict[int, torch.Tensor]:
    """Build a year -> z-scored feature tensor dictionary that differs from
    `feat_set.by_year` only in the targeted (focal_idx, feature_idx) cell
    over the configured year range.

    Returns a fresh dict; does not mutate `feat_set`.
    """
    if config.focal_ccode not in dataset.ccode_to_idx:
        raise ValueError(
            f"focal_ccode {config.focal_ccode} not in dataset node index"
        )
    if config.feature_name not in feat_set.feature_names:
        raise ValueError(
            f"feature {config.feature_name!r} not in NodeFeatureSet "
            f"(have {feat_set.feature_names})"
        )

    focal_idx = dataset.ccode_to_idx[config.focal_ccode]
    feat_idx = feat_set.feature_names.index(config.feature_name)
    mu = float(feat_set.feature_means[feat_idx])
    sd = float(feat_set.feature_stds[feat_idx])
    if sd <= 0:
        raise ValueError(
            f"std for feature {config.feature_name!r} is non-positive ({sd})"
        )
    target_z = (config.target_value_raw - mu) / sd

    y_start, y_end = config.year_range
    perturbed = {}
    for year, base in feat_set.by_year.items():
        clone = base.clone()
        if y_start <= year <= y_end:
            clone[focal_idx, feat_idx] = target_z
        perturbed[year] = clone
    return perturbed


# ---------------------------------------------------------------
# Re-encoding under frozen weights
# ---------------------------------------------------------------

def encode_history(
    encoder: HeterogeneousEncoder,
    dataset: MultiplexTemporalDataset,
    features_by_year: Dict[int, torch.Tensor],
    device: Optional[torch.device] = None,
) -> Dict[int, torch.Tensor]:
    """Run the (already-trained) encoder over every snapshot, returning
    {year: [num_nodes, emb_dim]} embeddings.

    Encoder is set to eval mode and grad is disabled. The same encoder
    weights are reused for both baseline and counterfactual runs.
    """
    if device is None:
        device = next(encoder.parameters()).device
    encoder.eval()
    out: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for year in dataset.years:
            snap = dataset.snapshots[year]
            nf = features_by_year[year].to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            z = encoder(nf, ei, ew, snap.layer_mask)
            out[year] = z.detach().cpu()
    return out


# ---------------------------------------------------------------
# Cascade decomposition (k-hop neighborhoods at the final year)
# ---------------------------------------------------------------

def k_hop_partition(
    dataset: MultiplexTemporalDataset,
    focal_idx: int,
    year: int,
    max_hop: int = 3,
    only_layers: Optional[List[str]] = None,
) -> Dict[int, List[int]]:
    """BFS partition of nodes into k-hop sets from focal in the merged
    edge graph at `year`. Returns {0: [focal_idx], 1: [...], 2: [...], 3+: [...]}.

    only_layers : restrict the merged edge graph to these layers.
        Default is to use all available layers in the snapshot.
    """
    snap = dataset.snapshots[year]
    edges = []
    for ln, available in snap.layer_mask.items():
        if not available:
            continue
        if only_layers is not None and ln not in only_layers:
            continue
        ei = snap.edge_indices.get(ln)
        if ei is None or ei.numel() == 0:
            continue
        edges.append(ei.cpu().numpy())
    if not edges:
        # No edges at all; everything is hop-3+
        partition = {0: [focal_idx], 1: [], 2: [], 3: []}
        partition[3] = [i for i in range(dataset.num_nodes) if i != focal_idx]
        return partition

    merged = np.concatenate(edges, axis=1)
    # Build undirected adjacency lookup
    adj: Dict[int, set] = {i: set() for i in range(dataset.num_nodes)}
    for s, t in zip(merged[0].tolist(), merged[1].tolist()):
        adj[s].add(t)
        adj[t].add(s)

    # BFS
    hop_of: Dict[int, int] = {focal_idx: 0}
    frontier = {focal_idx}
    for h in range(1, max_hop + 1):
        next_frontier = set()
        for u in frontier:
            for v in adj[u]:
                if v not in hop_of:
                    hop_of[v] = h
                    next_frontier.add(v)
        frontier = next_frontier
        if not frontier:
            break

    partition: Dict[int, List[int]] = {0: [focal_idx], 1: [], 2: [], 3: []}
    for node, h in hop_of.items():
        if node == focal_idx:
            continue
        bucket = h if h < 3 else 3
        partition[bucket].append(node)
    # Anyone not reached goes in the 3+ bucket
    reached = set(hop_of.keys())
    for i in range(dataset.num_nodes):
        if i not in reached:
            partition[3].append(i)
    return partition


def centroid_distance(emb: torch.Tensor) -> torch.Tensor:
    """Per-node Euclidean distance to the centroid of all embeddings."""
    centroid = emb.mean(dim=0, keepdim=True)
    return torch.linalg.vector_norm(emb - centroid, dim=1)


# ---------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------

def run_feature_intervention(
    encoder: HeterogeneousEncoder,
    dataset: MultiplexTemporalDataset,
    feat_set: NodeFeatureSet,
    config: FeatureInterventionConfig,
    cascade_year: Optional[int] = None,
    cascade_layers: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> CascadeResult:
    """Run one regime-shock scenario end to end.

    1. Build perturbed feature panel.
    2. Re-encode baseline and counterfactual histories at frozen weights.
    3. Compute per-state displacement and centroid-distance shifts at the
       end year of the perturbation window (or `cascade_year` if specified).
    4. Aggregate per-hop on the merged alliance graph.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    focal_idx = dataset.ccode_to_idx[config.focal_ccode]

    # Step 1 — build counterfactual feature panel
    perturbed_features = apply_feature_intervention(feat_set, dataset, config)

    # Step 2 — re-encode under both feature panels at frozen weights
    baseline_emb = encode_history(encoder, dataset, feat_set.by_year, device=device)
    cf_emb = encode_history(encoder, dataset, perturbed_features, device=device)

    # Step 3 — pick the year at which to compute cascade
    if cascade_year is None:
        cascade_year = config.year_range[1]
    if cascade_year not in baseline_emb:
        # Fall back to the latest year of the dataset
        cascade_year = max(baseline_emb.keys())

    z_base = baseline_emb[cascade_year]   # [N, D]
    z_cf = cf_emb[cascade_year]           # [N, D]

    # Per-state L2 displacement and centroid-distance shift
    displacement = torch.linalg.vector_norm(z_cf - z_base, dim=1).numpy()
    cd_base = centroid_distance(z_base).numpy()
    cd_cf = centroid_distance(z_cf).numpy()
    cd_delta = cd_cf - cd_base

    # Step 4 — k-hop partition (from focal in merged alliance graph at cascade_year)
    partition = k_hop_partition(
        dataset, focal_idx, cascade_year, max_hop=3, only_layers=cascade_layers,
    )

    def mean_or_zero(values: np.ndarray, idx: List[int]) -> float:
        if not idx:
            return 0.0
        return float(np.mean(values[idx]))

    hop_0_disp = float(displacement[focal_idx])
    hop_1_disp = mean_or_zero(displacement, partition[1])
    hop_2_disp = mean_or_zero(displacement, partition[2])
    hop_3p_disp = mean_or_zero(displacement, partition[3])

    hop_0_cd = float(cd_delta[focal_idx])
    hop_1_cd = mean_or_zero(cd_delta, partition[1])
    hop_2_cd = mean_or_zero(cd_delta, partition[2])
    hop_3p_cd = mean_or_zero(cd_delta, partition[3])

    # Per-state long-form table (for downstream plotting / inspection)
    rows = []
    for idx in range(dataset.num_nodes):
        if idx == focal_idx:
            hop = 0
        elif idx in partition[1]:
            hop = 1
        elif idx in partition[2]:
            hop = 2
        else:
            hop = 3
        rows.append({
            "scenario": config.label,
            "focal_ccode": config.focal_ccode,
            "ccode": dataset.idx_to_ccode[idx],
            "node_idx": idx,
            "hop": hop,
            "displacement": float(displacement[idx]),
            "centroid_delta": float(cd_delta[idx]),
            "centroid_baseline": float(cd_base[idx]),
            "centroid_cf": float(cd_cf[idx]),
        })
    per_state = pd.DataFrame(rows)

    return CascadeResult(
        scenario_label=config.label,
        focal_ccode=config.focal_ccode,
        focal_idx=focal_idx,
        feature_name=config.feature_name,
        target_value_raw=config.target_value_raw,
        year_range=config.year_range,
        hop_0_focal_displacement=hop_0_disp,
        hop_1_mean_displacement=hop_1_disp,
        hop_2_mean_displacement=hop_2_disp,
        hop_3plus_mean_displacement=hop_3p_disp,
        hop_0_centroid_delta=hop_0_cd,
        hop_1_mean_centroid_delta=hop_1_cd,
        hop_2_mean_centroid_delta=hop_2_cd,
        hop_3plus_mean_centroid_delta=hop_3p_cd,
        n_hop_1=len(partition[1]),
        n_hop_2=len(partition[2]),
        n_hop_3plus=len(partition[3]),
        per_state_table=per_state,
    )


# ---------------------------------------------------------------
# Pre-baked scenario set for the four focal countries (USA, China,
# Russia, India). COW codes: USA=2, China=710, Russia=365, India=750.
# ---------------------------------------------------------------

POLITY_SHOCKS: List[FeatureInterventionConfig] = [
    FeatureInterventionConfig(
        focal_ccode=2, feature_name="polity2",
        target_value_raw=-10.0, year_range=(2010, 2016),
        label="USA_polity_-10",
    ),
    FeatureInterventionConfig(
        focal_ccode=710, feature_name="polity2",
        target_value_raw=6.0, year_range=(2010, 2016),
        label="CHN_polity_+6",
    ),
    FeatureInterventionConfig(
        focal_ccode=365, feature_name="polity2",
        target_value_raw=8.0, year_range=(2010, 2016),
        label="RUS_polity_+8",
    ),
    FeatureInterventionConfig(
        focal_ccode=750, feature_name="polity2",
        target_value_raw=-6.0, year_range=(2010, 2016),
        label="IND_polity_-6",
    ),
]

CINC_SHOCKS: List[FeatureInterventionConfig] = [
    FeatureInterventionConfig(
        focal_ccode=2, feature_name="cinc",
        target_value_raw=0.38, year_range=(2010, 2016),
        label="USA_cinc_1948level",
    ),
    FeatureInterventionConfig(
        focal_ccode=710, feature_name="cinc",
        target_value_raw=0.05, year_range=(2010, 2016),
        label="CHN_cinc_1980level",
    ),
    FeatureInterventionConfig(
        focal_ccode=365, feature_name="cinc",
        target_value_raw=0.18, year_range=(2010, 2016),
        label="RUS_cinc_1990USSR",
    ),
    FeatureInterventionConfig(
        focal_ccode=750, feature_name="cinc",
        target_value_raw=0.15, year_range=(2010, 2016),
        label="IND_cinc_higher_growth",
    ),
]

ALL_REGIME_SHOCKS: List[FeatureInterventionConfig] = POLITY_SHOCKS + CINC_SHOCKS
