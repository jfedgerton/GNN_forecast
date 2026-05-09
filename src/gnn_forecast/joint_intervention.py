"""Joint feature + edge intervention for the R-GCN pipeline.

For each (focal, feature_perturbation, edge_perturbation) triple, run:
  - feature_only:   apply feature change, encode, measure delta
  - edge_only:      apply edge change, encode, measure delta
  - joint:          apply BOTH, encode, measure delta
  - additive:       feature_only_delta + edge_only_delta (theoretical sum)
  - interaction:    joint - additive  (positive => super-additive synergy,
                                       negative => sub-additive)

This is the §6.7 / appendix interaction-term decomposition. The point is
to test whether two interventions combine linearly, or whether the
encoder discovers super-additive ("the polity shift makes the alliance
removal hurt MORE than either alone") or sub-additive ("the polity shift
buffers against the alliance removal") effects.

Because the encoder is non-linear (R-GCN message passing + ReLU), we
should expect non-zero interaction terms. The substantive question is
whether they're systematically signed and substantively meaningful.
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
from gnn_forecast.feature_intervention import (
    FeatureInterventionConfig, apply_feature_intervention, centroid_distance,
)
from gnn_forecast.edge_intervention import _modify_edges_symmetric

SEED = 123


@dataclass
class JointInterventionConfig:
    """One joint scenario."""
    label: str
    focal_ccode: int
    # Feature side
    feature_name: str
    feature_target_raw: float
    feature_year_range: Tuple[int, int]
    # Edge side
    partner_ccode: int
    layer_name: str
    operation: str               # "add" or "remove"
    symmetric_n_edges: int = 5


@dataclass
class JointInterventionResult:
    label: str
    focal_ccode: int
    focal_idx: int
    cascade_year: int
    # Per-condition focal-centroid-distance deltas
    delta_feature_only: float
    delta_edge_only: float
    delta_joint: float
    delta_additive: float        # = feature_only + edge_only
    interaction_term: float      # = joint - additive
    interaction_pct: float       # (joint - additive) / |additive|, capped at +/- 200%


def _focal_delta(
    encoder: HeterogeneousEncoder,
    dataset: MultiplexTemporalDataset,
    feat_set: NodeFeatureSet,
    focal_idx: int,
    focal_year: int,
    perturbed_features_year: torch.Tensor,
    perturbed_ei: Dict[str, torch.LongTensor],
    perturbed_ew: Dict[str, torch.FloatTensor],
    layer_mask: Dict[str, bool],
    baseline_centroid_dist: float,
    device: torch.device,
) -> float:
    """Run encoder once on perturbed inputs, return centroid-distance
    delta for the focal vs. baseline."""
    encoder.eval()
    with torch.no_grad():
        emb = encoder(perturbed_features_year, perturbed_ei, perturbed_ew, layer_mask)
        cd = centroid_distance(emb)[focal_idx].item()
    return float(cd - baseline_centroid_dist)


def run_joint_intervention(
    encoder: HeterogeneousEncoder,
    dataset: MultiplexTemporalDataset,
    feat_set: NodeFeatureSet,
    config: JointInterventionConfig,
    device: Optional[torch.device] = None,
) -> JointInterventionResult:
    """Run the four conditions (feature_only, edge_only, joint, baseline)
    and compute the interaction term.

    All conditions evaluate at config.feature_year_range[1] (the last
    year of the feature perturbation window).
    """
    if device is None:
        device = next(encoder.parameters()).device
    encoder.eval()
    torch.manual_seed(SEED)

    if config.focal_ccode not in dataset.ccode_to_idx:
        raise ValueError(f"focal_ccode {config.focal_ccode} not in dataset")
    if config.partner_ccode not in dataset.ccode_to_idx:
        raise ValueError(f"partner_ccode {config.partner_ccode} not in dataset")
    focal_idx = dataset.ccode_to_idx[config.focal_ccode]
    partner_idx = dataset.ccode_to_idx[config.partner_ccode]

    cascade_year = config.feature_year_range[1]
    if cascade_year not in dataset.snapshots:
        cascade_year = dataset.years[-1]

    snap = dataset.snapshots[cascade_year]
    base_nf = feat_set.by_year[cascade_year].to(device)
    base_ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
    base_ew = {k: v.to(device) for k, v in snap.edge_weights.items()}

    # Baseline centroid distance
    with torch.no_grad():
        base_emb = encoder(base_nf, base_ei, base_ew, snap.layer_mask)
        base_cd = float(centroid_distance(base_emb)[focal_idx].item())

    # FEATURE-ONLY: perturb feature, keep edges
    feat_cfg = FeatureInterventionConfig(
        focal_ccode=config.focal_ccode,
        feature_name=config.feature_name,
        target_value_raw=config.feature_target_raw,
        year_range=config.feature_year_range,
        label=config.label + "_featonly",
    )
    perturbed_feats_by_year = apply_feature_intervention(feat_set, dataset, feat_cfg)
    nf_feat_only = perturbed_feats_by_year[cascade_year].to(device)
    delta_feat = _focal_delta(
        encoder, dataset, feat_set, focal_idx, cascade_year,
        nf_feat_only, base_ei, base_ew, snap.layer_mask, base_cd, device,
    )

    # EDGE-ONLY: keep features, perturb edges
    mod_ei, mod_ew = _modify_edges_symmetric(
        base_ei, base_ew, config.layer_name, partner_idx,
        config.operation, config.symmetric_n_edges, dataset.num_nodes, device,
    )
    delta_edge = _focal_delta(
        encoder, dataset, feat_set, focal_idx, cascade_year,
        base_nf, mod_ei, mod_ew, snap.layer_mask, base_cd, device,
    )

    # JOINT: both perturbations
    delta_joint = _focal_delta(
        encoder, dataset, feat_set, focal_idx, cascade_year,
        nf_feat_only, mod_ei, mod_ew, snap.layer_mask, base_cd, device,
    )

    delta_additive = delta_feat + delta_edge
    interaction = delta_joint - delta_additive
    if abs(delta_additive) > 1e-6:
        interaction_pct = 100.0 * interaction / abs(delta_additive)
        interaction_pct = max(min(interaction_pct, 200.0), -200.0)
    else:
        interaction_pct = 0.0

    return JointInterventionResult(
        label=config.label,
        focal_ccode=config.focal_ccode,
        focal_idx=focal_idx,
        cascade_year=cascade_year,
        delta_feature_only=delta_feat,
        delta_edge_only=delta_edge,
        delta_joint=delta_joint,
        delta_additive=delta_additive,
        interaction_term=interaction,
        interaction_pct=interaction_pct,
    )


# ---------------------------------------------------------------
# Pre-baked joint scenarios
# ---------------------------------------------------------------
# These compose the four-focal regime shocks with a relevant edge
# perturbation. The edge partner is chosen to be substantively
# meaningful for each focal: removal of a key trade tie or alliance.

JOINT_SCENARIOS: List[JointInterventionConfig] = [
    # USA polity collapse + remove US-South Korea defensive alliance
    JointInterventionConfig(
        label="USA_polity-10_AND_remove_USA-KOR_alliance",
        focal_ccode=2, feature_name="polity2", feature_target_raw=-10.0,
        feature_year_range=(2010, 2016),
        partner_ccode=732, layer_name="defensive_alliances", operation="remove",
    ),
    # CHN democratization + add CHN-USA FTA
    JointInterventionConfig(
        label="CHN_polity+6_AND_add_CHN-USA_FTA",
        focal_ccode=710, feature_name="polity2", feature_target_raw=6.0,
        feature_year_range=(2010, 2016),
        partner_ccode=2, layer_name="fta", operation="add",
    ),
    # RUS democratization + remove RUS defensive alliances entirely (block via partner=USA proxy)
    JointInterventionConfig(
        label="RUS_polity+8_AND_remove_RUS-CHN_alliance",
        focal_ccode=365, feature_name="polity2", feature_target_raw=8.0,
        feature_year_range=(2010, 2016),
        partner_ccode=710, layer_name="defensive_alliances", operation="remove",
    ),
    # IND backsliding + remove IND-USA defense cooperation
    JointInterventionConfig(
        label="IND_polity-6_AND_remove_IND-USA_dca",
        focal_ccode=750, feature_name="polity2", feature_target_raw=-6.0,
        feature_year_range=(2010, 2016),
        partner_ccode=2, layer_name="dca", operation="remove",
    ),
]
