"""Two-stage training pipeline for the heterogeneous temporal GNN.

Stage 1: pretrain the encoder alone with InfoNCE on per-year link
         reconstruction. No temporal model. Encoder gets direct, strong
         supervision to produce link-predictive embeddings.

Stage 2: freeze the encoder, train the GRU on top of frozen embeddings
         to predict next-year representations.

Optional Stage 3: unfreeze and joint-fine-tune at a low learning rate.

This sidesteps the encoder-collapse failure mode that plagued the
single-stage end-to-end training (training.py:train_model). The
diagnostic ablation (2026-05-07) showed the encoder cannot learn
useful embeddings under coupled training; pretraining decouples them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .heterogeneous_model import (
    HeterogeneousEncoder,
    HeterogeneousEncoderConfig,
    HeterogeneousTemporalGNN,
    HeteroTemporalConfig,
)
from .multiplex_data import MultiplexTemporalDataset, MultiplexSnapshot
from .training import _merge_edge_indices

SEED = 123


# ---------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------

def infonce_loss(
    emb: torch.Tensor,
    edge_index: torch.LongTensor,
    num_neg_per_pos: int = 10,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Sampled InfoNCE / NT-Xent for graph link prediction.

    For each positive (i, j) edge:
        L_ij = -log( exp(z_i . z_j / T) /
                     [exp(z_i . z_j / T) + sum_k exp(z_i . z_k / T)] )
    where k ranges over K random negative nodes.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=emb.device, requires_grad=True)
    num_nodes = emb.size(0)
    src = edge_index[0]
    tgt = edge_index[1]
    n_pos = src.size(0)

    pos_scores = (emb[src] * emb[tgt]).sum(dim=1) / temperature
    neg_tgt = torch.randint(0, num_nodes, (n_pos, num_neg_per_pos), device=emb.device)
    neg_scores = (emb[src].unsqueeze(1) * emb[neg_tgt]).sum(dim=2) / temperature

    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
    log_denom = torch.logsumexp(all_scores, dim=1)
    return -(pos_scores - log_denom).mean()


# ---------------------------------------------------------------
# Stage 1 — encoder pretraining
# ---------------------------------------------------------------

@dataclass
class PretrainConfig:
    """Hyperparameters for stage-1 encoder pretraining."""
    num_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_neg_per_pos: int = 10
    temperature: float = 0.5
    print_every: int = 25
    # If True, train on every year's edges (treats each year as one
    # training sample). If False, only train on a single target year.
    multi_year: bool = True


@dataclass
class PretrainResult:
    encoder: HeterogeneousEncoder
    loss_history: List[float]
    auc_history: List[float]
    final_link_auc: float
    final_mean_norm: float


def _approx_link_auc(emb: torch.Tensor, edge_index: torch.LongTensor, num_nodes: int, n_neg: int = 1000) -> float:
    """AUC of inner-product scores at distinguishing positives from random negatives."""
    if edge_index.numel() == 0:
        return 0.5
    scores = emb @ emb.T
    src, tgt = edge_index[0], edge_index[1]
    n_pos = min(len(src), n_neg)
    perm = torch.randperm(len(src))[:n_pos]
    pos = scores[src[perm], tgt[perm]].detach().cpu().numpy()
    neg_src = torch.randint(0, num_nodes, (n_neg,))
    neg_tgt = torch.randint(0, num_nodes, (n_neg,))
    neg = scores[neg_src, neg_tgt].detach().cpu().numpy()
    all_s = np.concatenate([pos, neg])
    all_l = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-all_s)
    sorted_l = all_l[order]
    n_p = sorted_l.sum()
    n_n = len(sorted_l) - n_p
    if n_p == 0 or n_n == 0:
        return 0.5
    cum_neg = np.cumsum(1 - sorted_l)
    return float(np.sum(sorted_l * cum_neg) / (n_p * n_n))


def pretrain_encoder(
    encoder: HeterogeneousEncoder,
    dataset: MultiplexTemporalDataset,
    node_features_by_year: Dict[int, torch.Tensor],
    config: Optional[PretrainConfig] = None,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> PretrainResult:
    """Stage 1: pretrain the encoder with InfoNCE on link reconstruction.

    Each epoch iterates over all years (multi_year=True) or one year
    (multi_year=False). Within each year, encode the snapshot and apply
    InfoNCE on that year's merged edge index.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config is None:
        config = PretrainConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if config.multi_year:
        train_years = list(dataset.years)
    else:
        train_years = [dataset.years[-1]]

    print(f"[Pretrain] encoder on {len(train_years)} years × {config.num_epochs} epochs")

    loss_history: List[float] = []
    auc_history: List[float] = []

    for epoch in range(config.num_epochs):
        encoder.train()
        epoch_loss = 0.0
        n_batches = 0
        for year in train_years:
            snap = dataset.snapshots[year]
            nf = node_features_by_year[year].to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            merged_ei = _merge_edge_indices(ei, snap.layer_mask)
            if merged_ei.numel() == 0:
                continue
            emb = encoder(nf, ei, ew, snap.layer_mask)
            loss = infonce_loss(
                emb, merged_ei,
                num_neg_per_pos=config.num_neg_per_pos,
                temperature=config.temperature,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        # Evaluate AUC on the final training year
        encoder.eval()
        with torch.no_grad():
            year = train_years[-1]
            snap = dataset.snapshots[year]
            nf = node_features_by_year[year].to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            emb = encoder(nf, ei, ew, snap.layer_mask)
            merged_ei = _merge_edge_indices(ei, snap.layer_mask)
            auc = _approx_link_auc(emb, merged_ei, dataset.num_nodes)
            mean_norm = float(emb.norm(dim=1).mean().cpu())
        auc_history.append(auc)

        if (epoch + 1) % config.print_every == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{config.num_epochs} "
                  f"infonce={avg_loss:.4f} link_auc={auc:.4f} ||z||={mean_norm:.4f}")

    return PretrainResult(
        encoder=encoder,
        loss_history=loss_history,
        auc_history=auc_history,
        final_link_auc=auc_history[-1],
        final_mean_norm=mean_norm,
    )


# ---------------------------------------------------------------
# Stage 2 — train GRU on frozen encoder
# ---------------------------------------------------------------

@dataclass
class GRUTrainConfig:
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    seq_len: int = 5
    print_every: int = 10
    patience: int = 20
    min_delta: float = 1e-4


@dataclass
class GRUTrainResult:
    model: HeterogeneousTemporalGNN
    loss_history: List[float]
    n_epochs_run: int


def train_gru_on_frozen_encoder(
    model: HeterogeneousTemporalGNN,
    dataset: MultiplexTemporalDataset,
    node_features_by_year: Dict[int, torch.Tensor],
    config: Optional[GRUTrainConfig] = None,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> GRUTrainResult:
    """Stage 2: with encoder frozen, train the GRU to predict next-year embeddings.

    Loss = MSE(GRU output, next-year encoded embedding).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config is None:
        config = GRUTrainConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    model.freeze_encoder()
    # Pre-compute encoded snapshots for every year (since encoder is frozen)
    encoded_by_year: Dict[int, torch.Tensor] = {}
    model.encoder.eval()
    with torch.no_grad():
        for y in dataset.years:
            snap = dataset.snapshots[y]
            nf = node_features_by_year[y].to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            encoded_by_year[y] = model.encoder(nf, ei, ew, snap.layer_mask).detach()

    print(f"[GRU train] {len(dataset.years)} years cached, training GRU + head only")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )

    seq_len = config.seq_len
    years = list(dataset.years)
    if len(years) < seq_len + 1:
        raise ValueError(f"Need at least {seq_len + 1} years, got {len(years)}")

    loss_history: List[float] = []
    best = float("inf")
    patience_left = config.patience

    for epoch in range(config.num_epochs):
        model.train()
        # Encoder stays in eval mode even when training
        model.encoder.eval()
        epoch_loss = 0.0
        n_windows = 0
        for i in range(len(years) - seq_len):
            window_years = years[i: i + seq_len]
            target_year = years[i + seq_len]
            history = [encoded_by_year[y] for y in window_years]
            target = encoded_by_year[target_year]
            pred = model.forward_temporal(history)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_windows += 1
        avg = epoch_loss / max(n_windows, 1)
        loss_history.append(avg)
        if (epoch + 1) % config.print_every == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{config.num_epochs} mse={avg:.6f}")
        if avg < best - config.min_delta:
            best = avg
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {epoch+1}")
                break

    return GRUTrainResult(
        model=model,
        loss_history=loss_history,
        n_epochs_run=len(loss_history),
    )


# ---------------------------------------------------------------
# Convenience: pretrain → freeze → GRU in one call
# ---------------------------------------------------------------

@dataclass
class TwoStageResult:
    encoder_result: PretrainResult
    gru_result: GRUTrainResult
    yearly_embeddings: Dict[int, torch.Tensor]


def two_stage_train(
    dataset: MultiplexTemporalDataset,
    node_features_by_year: Dict[int, torch.Tensor],
    raw_feat_dim: int,
    pretrain_config: Optional[PretrainConfig] = None,
    gru_config: Optional[GRUTrainConfig] = None,
    encoder_hidden_dim: int = 64,
    encoder_emb_dim: int = 32,
    identity_dim: int = 16,
    temporal_hidden_dim: int = 64,
    seq_len: int = 5,
    seed: int = SEED,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> TwoStageResult:
    """Build the heterogeneous model, pretrain encoder, freeze, train GRU."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    enc_cfg = HeterogeneousEncoderConfig(
        relation_names=list(dataset.layer_names),
        raw_feat_dim=raw_feat_dim,
        identity_dim=identity_dim,
        hidden_dim=encoder_hidden_dim,
        emb_dim=encoder_emb_dim,
    )
    model_cfg = HeteroTemporalConfig(
        encoder=enc_cfg,
        temporal_hidden_dim=temporal_hidden_dim,
        seq_len=seq_len,
    )
    model = HeterogeneousTemporalGNN(dataset.num_nodes, model_cfg)

    print("=" * 60)
    print("STAGE 1: Encoder pretraining (InfoNCE on link reconstruction)")
    print("=" * 60)
    pre = pretrain_encoder(
        model.encoder, dataset, node_features_by_year,
        config=pretrain_config, seed=seed, device=device,
    )

    print("\n" + "=" * 60)
    print("STAGE 2: GRU on frozen encoder (MSE on next-year embedding)")
    print("=" * 60)
    gru_res = train_gru_on_frozen_encoder(
        model, dataset, node_features_by_year,
        config=gru_config, seed=seed, device=device,
    )

    # Snapshot embeddings for downstream consumers
    model.eval()
    yearly_embeddings: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for y in dataset.years:
            snap = dataset.snapshots[y]
            nf = node_features_by_year[y].to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            yearly_embeddings[y] = model.encode_snapshot(nf, ei, ew, snap.layer_mask).cpu()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path / "two_stage_model.pt")
        torch.save(yearly_embeddings, save_path / "yearly_embeddings.pt")
        torch.save({
            "pretrain_loss_history": pre.loss_history,
            "pretrain_auc_history": pre.auc_history,
            "gru_loss_history": gru_res.loss_history,
            "final_link_auc": pre.final_link_auc,
            "final_mean_norm": pre.final_mean_norm,
        }, save_path / "two_stage_diagnostics.pt")
        print(f"\nSaved model + embeddings to {save_path}")

    return TwoStageResult(
        encoder_result=pre,
        gru_result=gru_res,
        yearly_embeddings=yearly_embeddings,
    )
