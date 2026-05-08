"""Planted-feature-shock SBM validation for the R-GCN feature-intervention pipeline.

Mirrors the planted-wedge SBM validation in `simulation.py` (which validates
edge interventions on the old multiplex encoder), but for the new pipeline:
R-GCN encoder + feature-intervention machinery.

Validation question: when we shift a node's feature by k SDs, does the
encoder propagate that shock to the focal's k-hop neighborhood with
monotonically decaying magnitude in hop distance, and does the null
condition (no planted shift) show no propagation?

Workflow per replicate:
  1. Generate a 60-node, 3-block, 3-layer multiplex SBM with synthetic
     node features. Features are block-correlated so the encoder has
     real signal to learn.
  2. Train a small R-GCN on the SBM (~50 epochs of InfoNCE — fast).
  3. Pick a focal node from block 2. In the *planted* condition, shift
     its feature value by +3 SDs. In the *null* condition, leave it
     unchanged.
  4. Re-encode at frozen weights for both conditions.
  5. Compute per-hop cascade deltas (centroid distance shift + L2
     displacement), partitioned by k-hop graph distance from focal.
  6. Record: per-hop deltas, monotonic-decay check, max-hop magnitude.

Pass criteria:
  - Planted condition: hop-0 displacement > hop-1 > hop-2 > hop-3+
    (monotonic decay in expected magnitude).
  - Planted condition: hop-0 displacement is at least 3x larger than
    null condition's hop-0 displacement (effect size > noise floor).
  - Null condition: all hops show small (< 0.1) displacement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from gnn_forecast.heterogeneous_model import (
    HeterogeneousEncoder, HeterogeneousEncoderConfig,
)
from gnn_forecast.diagnostic_v3 import (
    MultiplexTemporalDataset, MultiplexSnapshot, infonce_loss,
    _merge_edge_indices,
)
from gnn_forecast.feature_intervention import (
    apply_feature_intervention, encode_history, k_hop_partition,
    centroid_distance, FeatureInterventionConfig,
)
from gnn_forecast.node_features import NodeFeatureSet

SEED = 123


# ---------------------------------------------------------------
# SBM generator with synthetic node features
# ---------------------------------------------------------------

@dataclass
class SBMReplicate:
    """One synthetic SBM with assigned focal and feature configuration."""
    dataset: MultiplexTemporalDataset
    feat_set: NodeFeatureSet
    focal_idx: int
    focal_ccode: int
    block_assignment: Dict[int, int]


def generate_sbm_with_features(
    num_nodes: int = 60,
    num_years: int = 30,
    num_layers: int = 3,
    num_blocks: int = 3,
    intra_block: float = 0.30,
    inter_block: float = 0.05,
    num_features: int = 4,
    seed: int = SEED,
) -> SBMReplicate:
    """Build a multiplex SBM with block-correlated node features.

    Features are 4-dim per node: 2 block-correlated (informative) +
    2 random (noise). Block-correlated dims have mean = block_id and
    SD = 1, so blocks are linearly separable in feature space.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Block assignment
    block_size = num_nodes // num_blocks
    block_assignment: Dict[int, int] = {}
    for i in range(num_nodes):
        b = min(i // block_size, num_blocks - 1)
        block_assignment[i] = b

    # ccode assignment: synthetic codes 1000+i, no special USA/China
    idx_to_ccode = {i: 1000 + i for i in range(num_nodes)}
    ccode_to_idx = {cc: idx for idx, cc in idx_to_ccode.items()}

    # Per-year, per-layer adjacency
    layer_names = [f"layer_{l}" for l in range(num_layers)]
    base_year = 2000
    snapshots: Dict[int, MultiplexSnapshot] = {}

    for t in range(num_years):
        year = base_year + t
        edge_indices: Dict[str, torch.LongTensor] = {}
        edge_weights: Dict[str, torch.FloatTensor] = {}
        layer_mask: Dict[str, bool] = {}

        for l_idx, ln in enumerate(layer_names):
            intra = intra_block * (0.85 + 0.15 * (l_idx + 1) / num_layers)
            inter = inter_block * (0.85 + 0.15 * (l_idx + 1) / num_layers)

            edges_src: List[int] = []
            edges_tgt: List[int] = []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    p = intra if block_assignment[i] == block_assignment[j] else inter
                    if rng.random() < p:
                        edges_src.append(i); edges_tgt.append(j)
                        edges_src.append(j); edges_tgt.append(i)

            ei = (
                torch.tensor([edges_src, edges_tgt], dtype=torch.long)
                if edges_src else torch.zeros((2, 0), dtype=torch.long)
            )
            ew = torch.ones(ei.size(1)) if ei.numel() > 0 else torch.zeros(0)
            edge_indices[ln] = ei
            edge_weights[ln] = ew
            layer_mask[ln] = True

        # Node features: block_id + small noise, plus pure noise dims
        feat = np.zeros((num_nodes, num_features), dtype=np.float32)
        for i in range(num_nodes):
            b = block_assignment[i]
            feat[i, 0] = b + rng.normal(0, 0.5)
            feat[i, 1] = b + rng.normal(0, 0.5)
            feat[i, 2] = rng.normal(0, 1.0)
            feat[i, 3] = rng.normal(0, 1.0)

        snapshots[year] = MultiplexSnapshot(
            year=year,
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            layer_mask=layer_mask,
        )

    years = sorted(snapshots.keys())
    nodes_df = pd.DataFrame({
        "ccode": [idx_to_ccode[i] for i in range(num_nodes)],
        "block": [block_assignment[i] for i in range(num_nodes)],
    })

    dataset = MultiplexTemporalDataset(
        years=years, num_nodes=num_nodes, layer_names=layer_names,
        ccode_to_idx=ccode_to_idx, idx_to_ccode=idx_to_ccode,
        snapshots=snapshots, nodes_df=nodes_df,
    )

    # NodeFeatureSet — same value for every year (features are static
    # over the SBM lifetime; this keeps the validation simple).
    feat_t = torch.tensor(feat, dtype=torch.float32)
    by_year = {y: feat_t.clone() for y in years}
    feature_names = [f"f_{i}" for i in range(num_features)]
    means = feat.mean(axis=0)
    stds = feat.std(axis=0).clip(min=1e-6)
    # Z-score
    feat_z = (feat - means) / stds
    feat_z_t = torch.tensor(feat_z, dtype=torch.float32)
    by_year_z = {y: feat_z_t.clone() for y in years}

    feat_set = NodeFeatureSet(
        feature_names=feature_names,
        num_features=num_features,
        by_year=by_year_z,
        feature_means=means,
        feature_stds=stds,
        raw_df=pd.DataFrame(),
    )

    # Pick a default focal from block 2 (mirrors the planted-wedge SBM)
    block2_indices = [i for i in range(num_nodes) if block_assignment[i] == 2]
    focal_idx = int(block2_indices[0])

    return SBMReplicate(
        dataset=dataset,
        feat_set=feat_set,
        focal_idx=focal_idx,
        focal_ccode=idx_to_ccode[focal_idx],
        block_assignment=block_assignment,
    )


# ---------------------------------------------------------------
# Train a small encoder on the SBM
# ---------------------------------------------------------------

def train_sbm_encoder(
    rep: SBMReplicate,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    hidden_dim: int = 32,
    emb_dim: int = 16,
    identity_dim: int = 8,
    num_neg_per_pos: int = 10,
    temperature: float = 0.5,
    device: Optional[torch.device] = None,
) -> HeterogeneousEncoder:
    """Quick R-GCN training run on the SBM. Smaller than the empirical
    encoder (less data, simpler structure)."""
    if device is None:
        device = torch.device("cpu")
    cfg = HeterogeneousEncoderConfig(
        relation_names=list(rep.dataset.layer_names),
        raw_feat_dim=rep.feat_set.num_features,
        identity_dim=identity_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        dropout=0.2,
    )
    encoder = HeterogeneousEncoder(rep.dataset.num_nodes, cfg).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        encoder.train()
        for year in rep.dataset.years:
            snap = rep.dataset.snapshots[year]
            nf = rep.feat_set.by_year[year].to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            merged = _merge_edge_indices(snap, device)
            if merged.numel() == 0:
                continue
            emb = encoder(nf, ei, ew, snap.layer_mask)
            loss = infonce_loss(emb, merged, num_neg_per_pos, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
    encoder.eval()
    return encoder


# ---------------------------------------------------------------
# Single-replicate planted-shock test
# ---------------------------------------------------------------

@dataclass
class PlantedShockResult:
    replicate_id: int
    scenario: str  # "planted" or "null"
    focal_idx: int
    feature_name: str
    target_z_shift: float
    hop_0_displacement: float
    hop_1_mean_displacement: float
    hop_2_mean_displacement: float
    hop_3plus_mean_displacement: float
    hop_0_centroid_delta: float
    n_hop_1: int
    n_hop_2: int
    n_hop_3plus: int
    monotonic_decay: bool


def _planted_shift_to_raw(
    feat_set: NodeFeatureSet,
    feature_name: str,
    z_shift: float,
    focal_idx: int,
) -> float:
    """Convert a desired z-shift into the equivalent RAW value to pass
    to FeatureInterventionConfig (which expects raw units)."""
    fi = feat_set.feature_names.index(feature_name)
    mu = float(feat_set.feature_means[fi])
    sd = float(feat_set.feature_stds[fi])
    # Current raw value of the focal at any year (features are static here)
    any_year = next(iter(feat_set.by_year))
    cur_z = float(feat_set.by_year[any_year][focal_idx, fi].item())
    cur_raw = cur_z * sd + mu
    return cur_raw + z_shift * sd


def run_planted_shock_replicate(
    replicate_id: int,
    scenario: str,
    z_shift: float = 3.0,
    feature_name: str = "f_0",
    seed: int = SEED,
    num_epochs: int = 50,
) -> PlantedShockResult:
    """One SBM replicate. scenario in {"planted", "null"}.

    "planted": shift focal's f_0 by +z_shift SDs.
    "null": no shift (target_value_raw == current value, so no perturbation).
    """
    rep = generate_sbm_with_features(seed=seed)
    encoder = train_sbm_encoder(rep, num_epochs=num_epochs)

    if scenario == "planted":
        target_raw = _planted_shift_to_raw(
            rep.feat_set, feature_name, z_shift, rep.focal_idx,
        )
        actual_z_shift = z_shift
    elif scenario == "null":
        # No shift — target equals current value
        fi = rep.feat_set.feature_names.index(feature_name)
        any_year = next(iter(rep.feat_set.by_year))
        cur_z = float(rep.feat_set.by_year[any_year][rep.focal_idx, fi].item())
        target_raw = cur_z * float(rep.feat_set.feature_stds[fi]) \
                     + float(rep.feat_set.feature_means[fi])
        actual_z_shift = 0.0
    else:
        raise ValueError(f"unknown scenario: {scenario!r}")

    cfg = FeatureInterventionConfig(
        focal_ccode=rep.focal_ccode,
        feature_name=feature_name,
        target_value_raw=target_raw,
        year_range=(rep.dataset.years[0], rep.dataset.years[-1]),
        label=f"sbm_{scenario}_rep{replicate_id}",
    )

    # Re-encode under both conditions
    perturbed_features = apply_feature_intervention(rep.feat_set, rep.dataset, cfg)
    base_emb_by_year = encode_history(encoder, rep.dataset, rep.feat_set.by_year)
    cf_emb_by_year = encode_history(encoder, rep.dataset, perturbed_features)

    cascade_year = rep.dataset.years[-1]
    z_base = base_emb_by_year[cascade_year]
    z_cf = cf_emb_by_year[cascade_year]
    displacement = torch.linalg.vector_norm(z_cf - z_base, dim=1).numpy()
    cd_delta = (centroid_distance(z_cf) - centroid_distance(z_base)).numpy()

    partition = k_hop_partition(rep.dataset, rep.focal_idx, cascade_year, max_hop=3)

    def mean_or_zero(arr, idx):
        return float(np.mean(arr[idx])) if idx else 0.0

    hop_0_disp = float(displacement[rep.focal_idx])
    hop_1_disp = mean_or_zero(displacement, partition[1])
    hop_2_disp = mean_or_zero(displacement, partition[2])
    hop_3p_disp = mean_or_zero(displacement, partition[3])
    monotonic = hop_0_disp >= hop_1_disp >= hop_2_disp >= hop_3p_disp

    return PlantedShockResult(
        replicate_id=replicate_id,
        scenario=scenario,
        focal_idx=rep.focal_idx,
        feature_name=feature_name,
        target_z_shift=actual_z_shift,
        hop_0_displacement=hop_0_disp,
        hop_1_mean_displacement=hop_1_disp,
        hop_2_mean_displacement=hop_2_disp,
        hop_3plus_mean_displacement=hop_3p_disp,
        hop_0_centroid_delta=float(cd_delta[rep.focal_idx]),
        n_hop_1=len(partition[1]),
        n_hop_2=len(partition[2]),
        n_hop_3plus=len(partition[3]),
        monotonic_decay=monotonic,
    )


# ---------------------------------------------------------------
# Full study: R replicates each of {planted, null}
# ---------------------------------------------------------------

def run_planted_shock_study(
    n_replicates: int = 10,
    z_shift: float = 3.0,
    feature_name: str = "f_0",
    base_seed: int = SEED,
    num_epochs: int = 50,
) -> pd.DataFrame:
    """Run n_replicates each of planted and null. Returns long-form
    DataFrame ready for downstream aggregation."""
    rows = []
    for r in range(n_replicates):
        for scenario in ("planted", "null"):
            res = run_planted_shock_replicate(
                replicate_id=r,
                scenario=scenario,
                z_shift=z_shift,
                feature_name=feature_name,
                seed=base_seed + r,
                num_epochs=num_epochs,
            )
            rows.append({
                "replicate_id": res.replicate_id,
                "scenario": res.scenario,
                "focal_idx": res.focal_idx,
                "feature_name": res.feature_name,
                "target_z_shift": res.target_z_shift,
                "hop_0_displacement": res.hop_0_displacement,
                "hop_1_displacement": res.hop_1_mean_displacement,
                "hop_2_displacement": res.hop_2_mean_displacement,
                "hop_3plus_displacement": res.hop_3plus_mean_displacement,
                "hop_0_centroid_delta": res.hop_0_centroid_delta,
                "n_hop_1": res.n_hop_1,
                "n_hop_2": res.n_hop_2,
                "n_hop_3plus": res.n_hop_3plus,
                "monotonic_decay": res.monotonic_decay,
            })
            print(f"  rep {r:2d} {scenario:>7s}: "
                  f"hop-0={res.hop_0_displacement:.3f}, "
                  f"hop-1={res.hop_1_mean_displacement:.3f}, "
                  f"hop-2={res.hop_2_mean_displacement:.3f}, "
                  f"hop-3+={res.hop_3plus_mean_displacement:.3f}, "
                  f"monotonic={res.monotonic_decay}")
    return pd.DataFrame(rows)
