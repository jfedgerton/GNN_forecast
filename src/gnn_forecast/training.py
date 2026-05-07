"""Training loop for the multiplex temporal GNN.

Loss function:
  L = L_embedding + lambda_link * L_link + lambda_smooth * L_smooth

where:
  L_embedding = MSE between predicted Z^{t+1} and actual H_{t+1}
  L_link = BCE on edge reconstruction from embeddings (regularizer)
  L_smooth = MSE penalty on temporal embedding changes (smoothness)

Training proceeds sequentially over years: for each window of seq_len years,
encode each year, feed the sequence to the GRU, and predict the next year.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiplex_data import MultiplexTemporalDataset, MultiplexSnapshot
from .multiplex_model import MultiplexTemporalGNN, MultiplexGNNConfig


@dataclass
class TrainingConfig:
    """Hyperparameters for training.

    NOTE on lambda_link and lambda_anti_collapse (added 2026-05-06 after
    encoder-collapse incident on the full Roar dataset): with the original
    lambda_link=0.1 and no anti-collapse term, the model converged to the
    trivial all-zero embedding solution on 204-node real data — link loss
    barely dropped below 0.693 (chance baseline for BCE on uniform 0.5).
    Bumping lambda_link to 1.0 strengthens the link-reconstruction signal
    sufficiently to push the encoder away from zero. The anti-collapse
    term explicitly penalizes mean embedding norms below 1.0.
    """
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    seq_len: int = 5              # window of years for temporal model
    lambda_link: float = 5.0      # weight for link reconstruction loss (raised 0.1->1.0->5.0; 2026-05-06)
    lambda_smooth: float = 0.01   # weight for temporal smoothness penalty
    lambda_anti_collapse: float = 0.2  # penalty for mean embedding norm below target_norm
    anti_collapse_target_norm: float = 2.0  # push mean ||z|| toward at least this value
    link_sample_size: int = 500   # number of node pairs to sample for link loss
    patience: int = 20            # early stopping patience
    min_delta: float = 1e-4       # minimum improvement for early stopping
    print_every: int = 10         # print loss every N epochs
    save_dir: Optional[str] = None


@dataclass
class TrainingResult:
    """Results from training."""
    model: MultiplexTemporalGNN
    config: TrainingConfig
    loss_history: List[float]
    epoch_details: List[Dict]
    yearly_embeddings: Dict[int, torch.Tensor]  # year -> [num_nodes, emb_dim]
    layer_weights: Dict[str, float]


def _compute_link_loss(
    emb: torch.Tensor,
    edge_index: torch.LongTensor,
    num_nodes: int,
    sample_size: int = 500,
) -> torch.Tensor:
    """Binary cross-entropy link reconstruction loss.

    Samples positive edges from the actual graph and negative edges randomly.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=emb.device)

    scores = emb @ emb.T  # [N, N]
    sigmoid_scores = torch.sigmoid(scores)

    # Positive edges
    pos_src, pos_tgt = edge_index[0], edge_index[1]
    n_pos = min(len(pos_src), sample_size)
    if n_pos == 0:
        return torch.tensor(0.0, device=emb.device)

    perm = torch.randperm(len(pos_src))[:n_pos]
    pos_scores = sigmoid_scores[pos_src[perm], pos_tgt[perm]]
    pos_labels = torch.ones(n_pos, device=emb.device)

    # Negative edges (random pairs)
    neg_src = torch.randint(0, num_nodes, (n_pos,), device=emb.device)
    neg_tgt = torch.randint(0, num_nodes, (n_pos,), device=emb.device)
    neg_scores = sigmoid_scores[neg_src, neg_tgt]
    neg_labels = torch.zeros(n_pos, device=emb.device)

    all_scores = torch.cat([pos_scores, neg_scores])
    all_labels = torch.cat([pos_labels, neg_labels])

    return F.binary_cross_entropy(all_scores.clamp(1e-7, 1 - 1e-7), all_labels)


def _merge_edge_indices(
    edge_indices: Dict[str, torch.LongTensor],
    layer_mask: Dict[str, bool],
) -> torch.LongTensor:
    """Merge edge indices from all available layers into one edge_index for link loss."""
    parts = []
    for name, ei in edge_indices.items():
        if layer_mask.get(name, False) and ei.numel() > 0:
            parts.append(ei)
    if not parts:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.cat(parts, dim=1)


def train_model(
    dataset: MultiplexTemporalDataset,
    model_config: Optional[MultiplexGNNConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
) -> TrainingResult:
    """Train the multiplex temporal GNN on a dataset.

    Parameters
    ----------
    dataset : MultiplexTemporalDataset
    model_config : MultiplexGNNConfig or None (uses defaults)
    train_config : TrainingConfig or None (uses defaults)
    device : torch.device or None (auto-detect)

    Returns
    -------
    TrainingResult with trained model, embeddings, and diagnostics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_config is None:
        model_config = MultiplexGNNConfig(
            num_layers=len(dataset.layer_names),
            in_dim=len(dataset.layer_names),
            emb_dim=32,
        )

    if train_config is None:
        train_config = TrainingConfig()

    print(f"Training on device: {device}")
    print(f"Layers: {dataset.layer_names}")
    print(f"Years: {dataset.years[0]}-{dataset.years[-1]} ({len(dataset.years)} years)")
    print(f"Nodes: {dataset.num_nodes}")
    print(f"Seq len: {train_config.seq_len}")

    model = MultiplexTemporalGNN(model_config, dataset.layer_names).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    years = dataset.years
    seq_len = train_config.seq_len

    # We need at least seq_len + 1 years (seq_len for input, 1 for target)
    if len(years) < seq_len + 1:
        raise ValueError(
            f"Need at least {seq_len + 1} years for training, got {len(years)}"
        )

    loss_history = []
    epoch_details = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(train_config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_emb_loss = 0.0
        epoch_link_loss = 0.0
        epoch_smooth_loss = 0.0
        epoch_mean_norm = 0.0
        n_windows = 0

        # Slide window over years. Re-encode within each window to avoid
        # in-place modification conflicts during backprop.
        for i in range(len(years) - seq_len):
            window_years = years[i: i + seq_len]
            target_year = years[i + seq_len]

            # Encode window years + target year fresh each iteration
            emb_seq = []
            for y in window_years:
                snap = dataset.snapshots[y]
                nf = snap.node_features.to(device)
                ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
                ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
                emb_seq.append(model.encode_snapshot(nf, ei, ew, snap.layer_mask))

            target_snap = dataset.snapshots[target_year]
            nf_t = target_snap.node_features.to(device)
            ei_t = {k: v.to(device) for k, v in target_snap.edge_indices.items()}
            ew_t = {k: v.to(device) for k, v in target_snap.edge_weights.items()}
            target_emb = model.encode_snapshot(nf_t, ei_t, ew_t, target_snap.layer_mask)

            # Predict next embedding
            pred_emb = model.forward_temporal(emb_seq)

            # Embedding prediction loss (primary).
            #
            # NOTE on target_emb.detach():
            # The target embedding is detached so the GCN encoder is NOT
            # updated by the embedding-MSE term. The encoder is trained
            # only by (a) the link-reconstruction loss and (b) the
            # smoothness penalty. The GRU temporal head is the only
            # component updated by l_emb.
            #
            # Rationale: with joint backprop the encoder can collapse to a
            # near-constant representation, which trivially minimizes the
            # year-to-year prediction MSE but produces uninformative
            # embeddings. Detaching keeps the encoder grounded in the link-
            # reconstruction objective and forces the GRU to do the actual
            # temporal forecasting work. See Methods § "Two-stage learning
            # signal" for the formal argument.
            l_emb = F.mse_loss(pred_emb, target_emb.detach())

            # Link reconstruction loss (regularizer)
            merged_ei = _merge_edge_indices(
                {k: v.to(device) for k, v in target_snap.edge_indices.items()},
                target_snap.layer_mask,
            )
            l_link = _compute_link_loss(
                pred_emb, merged_ei, dataset.num_nodes, train_config.link_sample_size,
            )

            # Temporal smoothness penalty
            l_smooth = F.mse_loss(pred_emb, emb_seq[-1].detach())

            # Anti-collapse penalty: penalize when mean embedding norm
            # falls below `anti_collapse_target_norm`. Uses the predicted
            # embedding (the GRU output) so that gradient flows through
            # both the GRU head and the encoder via the input history.
            mean_norm = pred_emb.norm(dim=1).mean()
            l_anti_collapse = F.relu(
                train_config.anti_collapse_target_norm - mean_norm
            )

            # Total loss
            loss = (
                l_emb
                + train_config.lambda_link * l_link
                + train_config.lambda_smooth * l_smooth
                + train_config.lambda_anti_collapse * l_anti_collapse
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_emb_loss += l_emb.item()
            epoch_link_loss += l_link.item()
            epoch_smooth_loss += l_smooth.item()
            epoch_mean_norm += float(mean_norm.detach().cpu())
            n_windows += 1

        avg_loss = epoch_loss / max(n_windows, 1)
        loss_history.append(avg_loss)
        epoch_details.append({
            "epoch": epoch,
            "total_loss": avg_loss,
            "emb_loss": epoch_emb_loss / max(n_windows, 1),
            "link_loss": epoch_link_loss / max(n_windows, 1),
            "smooth_loss": epoch_smooth_loss / max(n_windows, 1),
            "mean_emb_norm": epoch_mean_norm / max(n_windows, 1),
        })

        if (epoch + 1) % train_config.print_every == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{train_config.num_epochs}: "
                  f"loss={avg_loss:.6f} "
                  f"(emb={epoch_details[-1]['emb_loss']:.6f}, "
                  f"link={epoch_details[-1]['link_loss']:.6f}, "
                  f"smooth={epoch_details[-1]['smooth_loss']:.6f}, "
                  f"||z||={epoch_details[-1]['mean_emb_norm']:.4f})")

        # Early stopping
        if avg_loss < best_loss - train_config.min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Final embeddings for all years
    model.eval()
    yearly_embeddings: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for year in years:
            snap = dataset.snapshots[year]
            nf = snap.node_features.to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            emb = model.encode_snapshot(nf, ei, ew, snap.layer_mask)
            yearly_embeddings[year] = emb.cpu()

    # Layer weights
    layer_weights = model.get_layer_weights()

    # Save if requested
    if train_config.save_dir is not None:
        save_dir = Path(train_config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model_weights.pt")
        torch.save(yearly_embeddings, save_dir / "yearly_embeddings.pt")
        torch.save({
            "loss_history": loss_history,
            "epoch_details": epoch_details,
            "layer_weights": layer_weights,
            "config": {
                "model": model_config.__dict__,
                "training": train_config.__dict__,
            },
        }, save_dir / "training_diagnostics.pt")
        print(f"  Saved model and diagnostics to {save_dir}")

    return TrainingResult(
        model=model,
        config=train_config,
        loss_history=loss_history,
        epoch_details=epoch_details,
        yearly_embeddings=yearly_embeddings,
        layer_weights=layer_weights,
    )


def forecast_embeddings(
    model: MultiplexTemporalGNN,
    dataset: MultiplexTemporalDataset,
    n_steps: int = 25,
    seq_len: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[int, torch.Tensor]:
    """Autoregressive rollout of embeddings into the future.

    Uses the last seq_len years of data to predict the next n_steps years.

    Returns dict of future_year -> [num_nodes, emb_dim].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    years = dataset.years
    if len(years) < seq_len:
        raise ValueError(f"Need at least {seq_len} years, got {len(years)}")

    # Encode last seq_len years
    history: List[torch.Tensor] = []
    with torch.no_grad():
        for year in years[-seq_len:]:
            snap = dataset.snapshots[year]
            nf = snap.node_features.to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            emb = model.encode_snapshot(nf, ei, ew, snap.layer_mask)
            history.append(emb)

    # Autoregressive rollout
    last_year = years[-1]
    forecasts: Dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for step in range(n_steps):
            pred = model.forward_temporal(history[-seq_len:])
            future_year = last_year + step + 1
            forecasts[future_year] = pred.cpu()
            history.append(pred)

    return forecasts
