"""GRU temporal head + 2017-2040 forecast rollout for the R-GCN pipeline.

Pipeline:
  1. Train R-GCN encoder (already done, weights saved by run_diagnostic_v3.py)
  2. With encoder frozen, generate per-year embeddings for 1948-2016
  3. Train GRU on (z_{t-K+1:t}) -> z_{t+1} via embedding-MSE loss
  4. For forecasting: roll the GRU forward from the 2012-2016 embedding
     sequence to 2017-2040. Pure GRU rollout (no future edges, no future
     features). Per-scenario forecasts swap the encoder's last-window
     embeddings for the counterfactual encoding before rolling forward.

Why pure GRU rollout (option 2 from the design discussion):
  - We have no future edges or features post-2016.
  - GRU was trained on encoder-output sequences, so it operates entirely
    in embedding space and doesn't need future inputs to keep rolling.
  - Interventions at 2010-2016 propagate forward via the GRU's learned
    dynamics; the gap between baseline and counterfactual rollouts at
    2040 is the long-run effect of the intervention.

Caveats (paper-honest framing):
  - This is a model-based scenario projection, not a causal forecast.
  - Embedding trajectories don't decode into specific future ties without
    a separate edge predictor (out of scope here).
  - 24 years out from 2016 (i.e., 2040) is past the longest backtested
    horizon (20 years). Treat with appropriate caution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gnn_forecast.heterogeneous_model import (
    HeterogeneousEncoder, HeterogeneousEncoderConfig,
)
from gnn_forecast.diagnostic_v3 import MultiplexTemporalDataset
from gnn_forecast.node_features import NodeFeatureSet
from gnn_forecast.feature_intervention import (
    FeatureInterventionConfig, apply_feature_intervention, encode_history,
    centroid_distance,
)

SEED = 123


# ---------------------------------------------------------------
# GRU model + training
# ---------------------------------------------------------------

class EmbeddingGRU(nn.Module):
    """GRU that takes [N, K, D] (N nodes, K-window, D-emb) and predicts [N, D].

    Trained per-node (each node is a row of the sequence batch). Same
    GRU weights apply to all nodes.
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=emb_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """sequences: [N, K, D]. Returns [N, D]."""
        h, _ = self.gru(sequences)
        return self.head(h[:, -1, :])


def _build_embedding_history(
    encoder: HeterogeneousEncoder,
    dataset: MultiplexTemporalDataset,
    feat_set: NodeFeatureSet,
    device: Optional[torch.device] = None,
) -> Dict[int, torch.Tensor]:
    """Run the frozen encoder over every snapshot, return {year: [N, D]}.
    Same machinery as feature_intervention.encode_history but listed here
    explicitly so this module is self-sufficient."""
    return encode_history(encoder, dataset, feat_set.by_year, device=device)


def train_gru(
    embeddings_by_year: Dict[int, torch.Tensor],
    seq_len: int = 5,
    hidden_dim: int = 64,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = SEED,
    verbose: bool = True,
) -> Tuple[EmbeddingGRU, Dict[str, List[float]]]:
    """Train the GRU on embedding sequences.

    Build (input_seq, target) pairs: for each year t with at least
    seq_len prior years available, use [z_{t-K+1}, ..., z_t] -> z_{t+1}.
    The last 20% of years are held out for validation.

    Returns (trained_gru, history_dict).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    years = sorted(embeddings_by_year.keys())
    if len(years) < seq_len + 2:
        raise ValueError(f"Need >= {seq_len + 2} years, got {len(years)}")

    sample_emb = embeddings_by_year[years[0]]
    num_nodes, emb_dim = sample_emb.shape

    # Build (input, target) for each predict-year t (where t spans
    # seq_len-1 .. len(years)-2 inclusive; target is t+1)
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for ti in range(seq_len - 1, len(years) - 1):
        input_years = years[ti - seq_len + 1 : ti + 1]
        target_year = years[ti + 1]
        seq = torch.stack([embeddings_by_year[y] for y in input_years], dim=1)
        # seq: [N, K, D]
        target = embeddings_by_year[target_year]   # [N, D]
        pairs.append((seq, target))

    # Train/val split: hold out the last `val_frac` of pairs (last years)
    n_val = max(1, int(len(pairs) * val_frac))
    train_pairs = pairs[:-n_val]
    val_pairs = pairs[-n_val:]

    if verbose:
        print(f"  GRU training: {len(pairs)} (input,target) pairs, "
              f"split {len(train_pairs)} train / {len(val_pairs)} val")

    device = sample_emb.device
    gru = EmbeddingGRU(emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(gru.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    history = {"train_mse": [], "val_mse": [], "best_epoch": [], "best_val": []}
    best_val = float("inf")
    best_state = {k: v.clone().cpu() for k, v in gru.state_dict().items()}
    best_epoch = 0

    for epoch in range(num_epochs):
        gru.train()
        train_loss_sum = 0.0
        for seq, target in train_pairs:
            pred = gru(seq.to(device))
            loss = loss_fn(pred, target.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss_sum += float(loss.detach().cpu())
        train_loss = train_loss_sum / max(len(train_pairs), 1)

        gru.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            for seq, target in val_pairs:
                pred = gru(seq.to(device))
                val_loss_sum += float(loss_fn(pred, target.to(device)).cpu())
            val_loss = val_loss_sum / max(len(val_pairs), 1)

        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.clone().cpu() for k, v in gru.state_dict().items()}

        if verbose and ((epoch + 1) % 25 == 0 or epoch == 0):
            print(f"    epoch {epoch+1:>3d}/{num_epochs} train_mse={train_loss:.5f} "
                  f"val_mse={val_loss:.5f} (best={best_val:.5f} @ ep {best_epoch})")

    gru.load_state_dict(best_state)
    history["best_epoch"] = [best_epoch]
    history["best_val"] = [best_val]
    if verbose:
        print(f"  GRU trained. Best val MSE = {best_val:.5f} at epoch "
              f"{best_epoch}/{num_epochs}.")
    return gru, history


# ---------------------------------------------------------------
# Forecast rollout
# ---------------------------------------------------------------

def rollout_forecast(
    gru: EmbeddingGRU,
    embeddings_by_year: Dict[int, torch.Tensor],
    forecast_until_year: int,
    seq_len: int = 5,
) -> Dict[int, torch.Tensor]:
    """Roll the GRU forward from the last seq_len observed embeddings
    out to forecast_until_year (inclusive). Returns
    {year: [N, D]} for forecasted years only (does NOT include observed years).
    """
    gru.eval()
    years = sorted(embeddings_by_year.keys())
    last_year = years[-1]
    if forecast_until_year <= last_year:
        return {}

    device = next(gru.parameters()).device
    # Initialize sliding window with the last seq_len observed embeddings
    window: List[torch.Tensor] = [
        embeddings_by_year[y].to(device) for y in years[-seq_len:]
    ]
    forecast: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for y in range(last_year + 1, forecast_until_year + 1):
            seq = torch.stack(window, dim=1)      # [N, K, D]
            pred = gru(seq)                       # [N, D]
            forecast[y] = pred.detach().cpu()
            # Slide the window: drop oldest, append new prediction
            window = window[1:] + [pred]
    return forecast


# ---------------------------------------------------------------
# Per-focal trajectory extraction (centroid distance over time)
# ---------------------------------------------------------------

def trajectory_centroid_distance(
    embeddings_by_year: Dict[int, torch.Tensor],
    focal_idx: int,
) -> pd.DataFrame:
    """Return a DataFrame with one row per year: focal centroid distance,
    mean centroid distance across all nodes."""
    rows = []
    for y in sorted(embeddings_by_year.keys()):
        z = embeddings_by_year[y]
        cd = centroid_distance(z)
        rows.append({
            "year": y,
            "focal_centroid_dist": float(cd[focal_idx].item()),
            "mean_centroid_dist": float(cd.mean().item()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Per-scenario paired baseline / counterfactual rollout
# ---------------------------------------------------------------

@dataclass
class ForecastScenarioResult:
    scenario_label: str
    focal_ccode: int
    focal_idx: int
    # Combined observed + forecasted embeddings (long-form)
    trajectory_baseline: pd.DataFrame
    trajectory_counterfactual: pd.DataFrame


def run_scenario_forecast(
    encoder: HeterogeneousEncoder,
    gru: EmbeddingGRU,
    dataset: MultiplexTemporalDataset,
    feat_set: NodeFeatureSet,
    feature_cfg: FeatureInterventionConfig,
    forecast_until_year: int = 2040,
    seq_len: int = 5,
    device: Optional[torch.device] = None,
) -> ForecastScenarioResult:
    """For one feature-intervention scenario, produce paired baseline +
    counterfactual trajectories (observed years + GRU rollout to
    forecast_until_year)."""
    if device is None:
        device = next(encoder.parameters()).device
    focal_idx = dataset.ccode_to_idx[feature_cfg.focal_ccode]

    # Baseline encoding
    base_emb = encode_history(encoder, dataset, feat_set.by_year, device=device)
    # Counterfactual encoding
    cf_feats = apply_feature_intervention(feat_set, dataset, feature_cfg)
    cf_emb = encode_history(encoder, dataset, cf_feats, device=device)

    # GRU rollout
    base_fcst = rollout_forecast(gru, base_emb, forecast_until_year, seq_len)
    cf_fcst = rollout_forecast(gru, cf_emb, forecast_until_year, seq_len)

    # Combine observed + forecasted years
    base_combined = {**base_emb, **base_fcst}
    cf_combined = {**cf_emb, **cf_fcst}

    base_traj = trajectory_centroid_distance(base_combined, focal_idx)
    cf_traj = trajectory_centroid_distance(cf_combined, focal_idx)
    base_traj["scenario"] = "baseline"
    cf_traj["scenario"] = feature_cfg.label

    return ForecastScenarioResult(
        scenario_label=feature_cfg.label,
        focal_ccode=feature_cfg.focal_ccode,
        focal_idx=focal_idx,
        trajectory_baseline=base_traj,
        trajectory_counterfactual=cf_traj,
    )


# ---------------------------------------------------------------
# Edge-counterfactual forecast scenarios
# ---------------------------------------------------------------

@dataclass
class EdgeScenarioConfig:
    """One edge-perturbation forecast scenario.

    The perturbation is applied at perturb_year (within the observed
    window), and the GRU then rolls embeddings forward to the
    forecast horizon. This is the paper #1 edge-counterfactual story
    (no future edges; perturb in observed years, roll dynamics forward).
    """
    label: str
    focal_ccode: int
    partner_ccode: int
    layer_name: str
    operation: str               # "remove" or "add"
    perturb_year: int = 2016     # year at which to apply the edge perturbation
    symmetric_n_edges: int = 5


def run_edge_scenario_forecast(
    encoder: HeterogeneousEncoder,
    gru: EmbeddingGRU,
    dataset: MultiplexTemporalDataset,
    feat_set: NodeFeatureSet,
    config: EdgeScenarioConfig,
    forecast_until_year: int = 2040,
    seq_len: int = 5,
    device: Optional[torch.device] = None,
) -> ForecastScenarioResult:
    """Edge-perturbation analog of run_scenario_forecast.

    Builds the baseline embedding sequence by encoding observed years.
    For the counterfactual: at perturb_year, replaces that year's
    edges in the targeted layer with a symmetric add/remove of k edges
    involving partner_idx, re-encodes that single snapshot, substitutes
    it into the embedding sequence, then rolls the GRU forward to
    forecast_until_year. Pre-perturb-year embeddings are unchanged.
    """
    from gnn_forecast.edge_intervention import _modify_edges_symmetric

    if device is None:
        device = next(encoder.parameters()).device
    if config.focal_ccode not in dataset.ccode_to_idx:
        raise ValueError(f"focal_ccode {config.focal_ccode} not in dataset")
    if config.partner_ccode not in dataset.ccode_to_idx:
        raise ValueError(f"partner_ccode {config.partner_ccode} not in dataset")
    focal_idx = dataset.ccode_to_idx[config.focal_ccode]
    partner_idx = dataset.ccode_to_idx[config.partner_ccode]

    perturb_year = config.perturb_year
    if perturb_year not in dataset.snapshots:
        perturb_year = dataset.years[-1]

    # Baseline encoding (no perturbation)
    base_emb = encode_history(encoder, dataset, feat_set.by_year, device=device)

    # Counterfactual: re-encode perturb_year with modified edges; copy other years
    snap = dataset.snapshots[perturb_year]
    nf = feat_set.by_year[perturb_year].to(device)
    base_ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
    base_ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
    mod_ei, mod_ew = _modify_edges_symmetric(
        base_ei, base_ew, config.layer_name, partner_idx, config.operation,
        config.symmetric_n_edges, dataset.num_nodes, device,
    )
    encoder.eval()
    with torch.no_grad():
        cf_perturb_emb = encoder(nf, mod_ei, mod_ew, snap.layer_mask).detach().cpu()

    cf_emb = dict(base_emb)
    cf_emb[perturb_year] = cf_perturb_emb

    base_fcst = rollout_forecast(gru, base_emb, forecast_until_year, seq_len)
    cf_fcst = rollout_forecast(gru, cf_emb, forecast_until_year, seq_len)
    base_combined = {**base_emb, **base_fcst}
    cf_combined = {**cf_emb, **cf_fcst}

    base_traj = trajectory_centroid_distance(base_combined, focal_idx)
    cf_traj = trajectory_centroid_distance(cf_combined, focal_idx)
    base_traj["scenario"] = "baseline"
    cf_traj["scenario"] = config.label

    return ForecastScenarioResult(
        scenario_label=config.label,
        focal_ccode=config.focal_ccode,
        focal_idx=focal_idx,
        trajectory_baseline=base_traj,
        trajectory_counterfactual=cf_traj,
    )


# Pre-baked edge scenarios for the four-focal set. Each removes a key
# bilateral tie at year 2016 and rolls forward to 2040.
EDGE_SCENARIOS: List[EdgeScenarioConfig] = [
    EdgeScenarioConfig(
        label="USA_loses_KOR_alliance_2016",
        focal_ccode=2, partner_ccode=732,
        layer_name="defensive_alliances", operation="remove",
    ),
    EdgeScenarioConfig(
        label="USA_loses_JPN_alliance_2016",
        focal_ccode=2, partner_ccode=740,
        layer_name="defensive_alliances", operation="remove",
    ),
    EdgeScenarioConfig(
        label="CHN_gains_USA_FTA_2016",
        focal_ccode=710, partner_ccode=2,
        layer_name="fta", operation="add",
    ),
    EdgeScenarioConfig(
        label="RUS_loses_CHN_alliance_2016",
        focal_ccode=365, partner_ccode=710,
        layer_name="defensive_alliances", operation="remove",
    ),
    EdgeScenarioConfig(
        label="IND_gains_USA_dca_2016",
        focal_ccode=750, partner_ccode=2,
        layer_name="dca", operation="add",
    ),
]
