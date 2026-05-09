"""Walk-forward backtest at 5/10/15/20-year horizons.

For each split year t_split, train R-GCN on data up to t_split, then
hold out years t_split+1 ... t_split+max_horizon. For each horizon h,
the forecast is the GRU rollout from the trained R-GCN's encoded
embeddings for years t_split-K+1 ... t_split. Scoring metrics are
computed against the (held-out) R-GCN encoder's outputs at the target
year, treated as the validation target.

Per-split, per-horizon metrics:
  embedding_mse:    mean squared error between forecasted embedding
                    and the held-out encoder output at year t_split+h
  link_pred_auc:    stratified-negatives AUC of the inner-product
                    decoder applied to forecasted embeddings against
                    actual edges at year t_split+h
  centroid_drift:   mean |centroid_dist(forecast) - centroid_dist(actual)|

The honest paper story: clean to 5-10y, predictable degradation to 15y,
near-baseline at 20y.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from gnn_forecast.heterogeneous_model import (
    HeterogeneousEncoder, HeterogeneousEncoderConfig,
)
from gnn_forecast.diagnostic_v3 import (
    MultiplexTemporalDataset, _merge_edge_indices, infonce_loss,
    stratified_link_auc,
)
from gnn_forecast.node_features import NodeFeatureSet
from gnn_forecast.forecast_v3 import (
    EmbeddingGRU, train_gru, _build_embedding_history,
    rollout_forecast, centroid_distance,
)

SEED = 123


def filter_dataset_to_split(
    dataset: MultiplexTemporalDataset,
    feat_set: NodeFeatureSet,
    end_year: int,
) -> Tuple[MultiplexTemporalDataset, NodeFeatureSet]:
    """Build a copy of the dataset truncated at end_year (inclusive)."""
    years = [y for y in dataset.years if y <= end_year]
    snapshots = {y: dataset.snapshots[y] for y in years}
    truncated_dataset = MultiplexTemporalDataset(
        years=years,
        num_nodes=dataset.num_nodes,
        layer_names=dataset.layer_names,
        ccode_to_idx=dataset.ccode_to_idx,
        idx_to_ccode=dataset.idx_to_ccode,
        snapshots=snapshots,
        nodes_df=dataset.nodes_df,
    )
    truncated_feats = NodeFeatureSet(
        feature_names=feat_set.feature_names,
        num_features=feat_set.num_features,
        by_year={y: feat_set.by_year[y] for y in years if y in feat_set.by_year},
        feature_means=feat_set.feature_means,
        feature_stds=feat_set.feature_stds,
        raw_df=feat_set.raw_df,
    )
    return truncated_dataset, truncated_feats


def train_split_encoder(
    train_dataset: MultiplexTemporalDataset,
    train_feats: NodeFeatureSet,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    emb_dim: int = 32,
    identity_dim: int = 16,
    num_neg_per_pos: int = 10,
    temperature: float = 0.5,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> Tuple[HeterogeneousEncoder, int, float]:
    """Train an R-GCN encoder on a truncated dataset; return (encoder,
    best_epoch, best_auc) using the same early-stopping protocol as
    the headline diagnostic."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    cfg = HeterogeneousEncoderConfig(
        relation_names=list(train_dataset.layer_names),
        raw_feat_dim=train_feats.num_features,
        identity_dim=identity_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        dropout=0.2,
    )
    encoder = HeterogeneousEncoder(train_dataset.num_nodes, cfg).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_auc = -1.0
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}

    for epoch in range(num_epochs):
        encoder.train()
        for year in train_dataset.years:
            snap = train_dataset.snapshots[year]
            nf = train_feats.by_year[year].to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            merged = _merge_edge_indices(snap, device)
            if merged.numel() == 0:
                continue
            emb = encoder(nf, ei, ew, snap.layer_mask)
            loss = infonce_loss(emb, merged, num_neg_per_pos, temperature)
            opt.zero_grad(); loss.backward(); opt.step()

        # Eval on the most recent training year's stratified AUC
        encoder.eval()
        with torch.no_grad():
            year = train_dataset.years[-1]
            snap = train_dataset.snapshots[year]
            nf = train_feats.by_year[year].to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            emb = encoder(nf, ei, ew, snap.layer_mask)
            merged = _merge_edge_indices(snap, device)
            auc = stratified_link_auc(emb, merged, train_dataset.num_nodes)

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}

    encoder.load_state_dict(best_state)
    encoder.eval()
    return encoder, best_epoch, best_auc


def encode_target_years(
    encoder: HeterogeneousEncoder,
    full_dataset: MultiplexTemporalDataset,
    full_feats: NodeFeatureSet,
    target_years: List[int],
    device: Optional[torch.device] = None,
) -> Dict[int, torch.Tensor]:
    """Run the trained encoder on full_dataset's snapshots at target_years.
    Used to score forecasts against actual encoded embeddings."""
    if device is None:
        device = next(encoder.parameters()).device
    encoder.eval()
    out = {}
    with torch.no_grad():
        for y in target_years:
            if y not in full_dataset.snapshots:
                continue
            snap = full_dataset.snapshots[y]
            nf = full_feats.by_year[y].to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            out[y] = encoder(nf, ei, ew, snap.layer_mask).detach().cpu()
    return out


@dataclass
class WalkForwardResult:
    split_year: int
    horizon: int
    target_year: int
    embedding_mse: float
    link_pred_auc: float
    centroid_drift: float
    n_target_edges: int


def run_walk_forward_split(
    full_dataset: MultiplexTemporalDataset,
    full_feats: NodeFeatureSet,
    split_year: int,
    horizons: List[int],
    seq_len: int = 5,
    encoder_epochs: int = 200,
    gru_epochs: int = 200,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> List[WalkForwardResult]:
    """For one split_year:
      1. Truncate dataset/features at split_year.
      2. Train encoder on truncated set.
      3. Build embedding history; train GRU.
      4. For each horizon h, roll GRU forward to split_year + h.
      5. Score forecasted embedding against actual encoded embedding
         at split_year + h (using the truncated encoder applied to
         the full dataset's later year)."""
    train_dataset, train_feats = filter_dataset_to_split(
        full_dataset, full_feats, split_year,
    )
    if len(train_dataset.years) < seq_len + 2:
        print(f"  SKIP split={split_year}: too few training years")
        return []

    print(f"\n=== split_year={split_year} (training on {len(train_dataset.years)} years) ===")
    encoder, best_ep, best_auc = train_split_encoder(
        train_dataset, train_feats, num_epochs=encoder_epochs, seed=seed, device=device,
    )
    print(f"  encoder best_auc={best_auc:.4f} at epoch {best_ep}")

    # Build embedding history on training years; train GRU
    train_emb = _build_embedding_history(encoder, train_dataset, train_feats, device=device)
    gru, _ = train_gru(train_emb, seq_len=seq_len, num_epochs=gru_epochs, verbose=False)
    print(f"  GRU trained")

    # Encode the held-out target years using THE SAME truncated encoder
    target_years = [split_year + h for h in horizons]
    actual_emb = encode_target_years(encoder, full_dataset, full_feats, target_years, device=device)

    # Roll forecast forward to the maximum target year
    max_target = max(target_years)
    fcst = rollout_forecast(gru, train_emb, max_target, seq_len)

    results = []
    for h in horizons:
        ty = split_year + h
        if ty not in actual_emb or ty not in fcst:
            continue
        f = fcst[ty]
        a = actual_emb[ty]
        mse = float(((f - a) ** 2).mean().item())
        actual_snap = full_dataset.snapshots[ty]
        actual_merged = _merge_edge_indices(actual_snap, torch.device("cpu"))
        if actual_merged.numel() > 0:
            auc = stratified_link_auc(f, actual_merged.cpu(), full_dataset.num_nodes)
            n_edges = int(actual_merged.size(1))
        else:
            auc = float("nan"); n_edges = 0
        cd_f = centroid_distance(f); cd_a = centroid_distance(a)
        cd_drift = float((cd_f - cd_a).abs().mean().item())
        results.append(WalkForwardResult(
            split_year=split_year, horizon=h, target_year=ty,
            embedding_mse=mse, link_pred_auc=auc, centroid_drift=cd_drift,
            n_target_edges=n_edges,
        ))
        print(f"  h={h:>2d} (year {ty}): mse={mse:.4f}, auc={auc:.4f}, cd_drift={cd_drift:.4f}")
    return results


def run_walk_forward_study(
    full_dataset: MultiplexTemporalDataset,
    full_feats: NodeFeatureSet,
    split_years: List[int],
    horizons: List[int],
    seq_len: int = 5,
    encoder_epochs: int = 200,
    gru_epochs: int = 200,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Run all (split, horizon) combinations. Returns long-form DataFrame."""
    rows = []
    for split in split_years:
        results = run_walk_forward_split(
            full_dataset, full_feats, split_year=split,
            horizons=horizons, seq_len=seq_len,
            encoder_epochs=encoder_epochs, gru_epochs=gru_epochs,
            seed=seed, device=device,
        )
        for r in results:
            rows.append({
                "split_year": r.split_year, "horizon": r.horizon,
                "target_year": r.target_year,
                "embedding_mse": r.embedding_mse,
                "link_pred_auc": r.link_pred_auc,
                "centroid_drift": r.centroid_drift,
                "n_target_edges": r.n_target_edges,
            })
    return pd.DataFrame(rows)
