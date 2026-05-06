"""Baseline models for comparison with the multiplex temporal GNN.

Implements:
1. Static GNN (no temporal component) — encodes one year, predicts next directly
2. Pure autoregressive on degree features (no GNN, just GRU on raw features)
3. Mean baseline (predict the mean embedding across training years)
4. Bilinear latent space model (AME-lite) — z_i, z_j latent positions per
   layer with logit p_ij = z_i' z_j; the IR analogue of Hoff's AME with
   no additive sender/receiver random effects. Trained per-year and
   stitched into a sequence so the full pipeline can compare it head-to-
   head with the GNN.

Each baseline returns the same output structure as the main model for fair comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiplex_data import MultiplexTemporalDataset
from .multiplex_model import MultiplexGNNConfig, MultiplexTemporalGNN, LayerEncoder, LayerAggregator
from .training import TrainingConfig, TrainingResult, _compute_link_loss, _merge_edge_indices


@dataclass
class BaselineResult:
    """Results from a baseline model."""
    model_name: str
    loss_history: List[float]
    yearly_embeddings: Dict[int, torch.Tensor]
    test_mse: Optional[Dict[int, float]] = None


# ---------------------------------------------------------------
# Baseline 1: Static GNN (encode year t, directly predict year t+1)
# ---------------------------------------------------------------
class StaticGNN(nn.Module):
    """GNN encoder + linear projection, no temporal recurrence.

    Encodes year t's graph, then applies a learned linear map to predict
    year t+1's embedding. No memory or sequential context.
    """
    def __init__(self, config: MultiplexGNNConfig, layer_names: List[str]):
        super().__init__()
        self.config = config
        self.layer_names = layer_names

        self.layer_encoders = nn.ModuleDict({
            name: LayerEncoder(config.in_dim, config.hidden_dim, config.emb_dim)
            for name in layer_names
        })
        self.aggregator = LayerAggregator(len(layer_names))
        # Simple linear projection for next-step prediction
        self.proj = nn.Linear(config.emb_dim, config.emb_dim)

    def forward(self, node_features, edge_indices, edge_weights, layer_mask):
        layer_embs = []
        mask_list = []
        for name in self.layer_names:
            available = layer_mask.get(name, False)
            mask_list.append(available)
            if available and name in edge_indices and edge_indices[name].numel() > 0:
                emb = self.layer_encoders[name](node_features, edge_indices[name], edge_weights.get(name))
            else:
                emb = torch.zeros(node_features.size(0), self.config.emb_dim, device=node_features.device)
            layer_embs.append(emb)

        agg = self.aggregator(layer_embs, mask_list)
        return self.proj(agg)


def train_static_gnn(
    dataset: MultiplexTemporalDataset,
    model_config: Optional[MultiplexGNNConfig] = None,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> BaselineResult:
    """Train the static GNN baseline."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_config is None:
        model_config = MultiplexGNNConfig(
            num_layers=len(dataset.layer_names),
            in_dim=len(dataset.layer_names),
        )

    model = StaticGNN(model_config, dataset.layer_names).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    years = dataset.years
    loss_history = []

    print(f"Training Static GNN baseline ({num_epochs} epochs)...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for i in range(len(years) - 1):
            snap = dataset.snapshots[years[i]]
            nf = snap.node_features.to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}

            pred = model(nf, ei, ew, snap.layer_mask)

            # Target: encode year t+1
            next_snap = dataset.snapshots[years[i + 1]]
            nf_next = next_snap.node_features.to(device)
            ei_next = {k: v.to(device) for k, v in next_snap.edge_indices.items()}
            ew_next = {k: v.to(device) for k, v in next_snap.edge_weights.items()}
            # Use same encoder for target (shared weights)
            with torch.no_grad():
                layer_embs = []
                mask_list = []
                for name in dataset.layer_names:
                    available = next_snap.layer_mask.get(name, False)
                    mask_list.append(available)
                    if available and name in ei_next and ei_next[name].numel() > 0:
                        emb = model.layer_encoders[name](nf_next, ei_next[name], ew_next.get(name))
                    else:
                        emb = torch.zeros(nf_next.size(0), model_config.emb_dim, device=device)
                    layer_embs.append(emb)
                target = model.aggregator(layer_embs, mask_list).detach()

            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1

        avg = epoch_loss / max(n, 1)
        loss_history.append(avg)
        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            print(f"  Epoch {epoch+1}: loss={avg:.6f}")

    # Extract embeddings
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for year in years:
            snap = dataset.snapshots[year]
            nf = snap.node_features.to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            # Use the encoder output (before projection) as the embedding
            layer_embs = []
            mask_list = []
            for name in dataset.layer_names:
                available = snap.layer_mask.get(name, False)
                mask_list.append(available)
                if available and name in ei and ei[name].numel() > 0:
                    emb = model.layer_encoders[name](nf, ei[name], ew.get(name))
                else:
                    emb = torch.zeros(nf.size(0), model_config.emb_dim, device=device)
                layer_embs.append(emb)
            embeddings[year] = model.aggregator(layer_embs, mask_list).cpu()

    return BaselineResult(
        model_name="static_gnn",
        loss_history=loss_history,
        yearly_embeddings=embeddings,
    )


# ---------------------------------------------------------------
# Baseline 2: Pure AR on degree features (no GNN)
# ---------------------------------------------------------------
class DegreeAR(nn.Module):
    """Autoregressive model on raw degree-profile features, no graph encoding."""
    def __init__(self, feat_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, feature_sequence: torch.Tensor) -> torch.Tensor:
        # feature_sequence: [num_nodes, seq_len, feat_dim]
        h, _ = self.gru(feature_sequence)
        return self.head(h[:, -1, :])


def train_degree_ar(
    dataset: MultiplexTemporalDataset,
    out_dim: int = 32,
    hidden_dim: int = 64,
    seq_len: int = 5,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> BaselineResult:
    """Train the degree-AR baseline (GRU on node features, no GNN)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_dim = len(dataset.layer_names)
    model = DegreeAR(feat_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    years = dataset.years
    loss_history = []

    print(f"Training Degree-AR baseline ({num_epochs} epochs)...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n = 0

        for i in range(len(years) - seq_len):
            window_years = years[i: i + seq_len]
            target_year = years[i + seq_len]

            # Build feature sequence
            feat_seq = torch.stack([
                dataset.snapshots[y].node_features for y in window_years
            ], dim=1).to(device)  # [N, seq_len, feat_dim]

            target_feat = dataset.snapshots[target_year].node_features.to(device)

            pred = model(feat_seq)
            loss = F.mse_loss(pred, target_feat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1

        avg = epoch_loss / max(n, 1)
        loss_history.append(avg)
        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            print(f"  Epoch {epoch+1}: loss={avg:.6f}")

    # Extract "embeddings" (output of AR model)
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for i in range(seq_len - 1, len(years)):
            window_years = years[max(0, i - seq_len + 1): i + 1]
            if len(window_years) < seq_len:
                # Pad with first year
                pad = [window_years[0]] * (seq_len - len(window_years))
                window_years = pad + list(window_years)

            feat_seq = torch.stack([
                dataset.snapshots[y].node_features for y in window_years
            ], dim=1).to(device)

            embeddings[years[i]] = model(feat_seq).cpu()

    return BaselineResult(
        model_name="degree_ar",
        loss_history=loss_history,
        yearly_embeddings=embeddings,
    )


# ---------------------------------------------------------------
# Baseline 3: Mean baseline (predict training mean)
# ---------------------------------------------------------------
def mean_baseline(
    dataset: MultiplexTemporalDataset,
    train_end_year: int,
    model_config: Optional[MultiplexGNNConfig] = None,
) -> BaselineResult:
    """Mean baseline: predict the average embedding from training years.

    Uses a simple GNN to encode each year, then averages across training years.
    The prediction for every future year is the same mean embedding.
    """
    # For fair comparison, use the node features as "embeddings"
    train_years = [y for y in dataset.years if y <= train_end_year]
    if not train_years:
        raise ValueError("No training years")

    # Average degree features across training years
    feats = [dataset.snapshots[y].node_features for y in train_years]
    mean_feat = torch.stack(feats).mean(dim=0)  # [N, feat_dim]

    embeddings = {y: mean_feat.clone() for y in dataset.years}

    return BaselineResult(
        model_name="mean_baseline",
        loss_history=[],
        yearly_embeddings=embeddings,
    )


# ---------------------------------------------------------------
# Baseline 4: Bilinear latent space model (AME-lite)
# ---------------------------------------------------------------
class BilinearLatentSpace(nn.Module):
    """Per-layer latent positions; logit p_ij = z_i' z_j.

    This is the GNN-free analogue of the dyadic latent space model
    (Hoff & Ward / Hoff's AME without additive sender/receiver effects).
    Each node has a learned latent position per layer; aggregated across
    layers via mean. Trained by binary cross-entropy on observed edges
    plus negative samples.

    Intentionally simple — the point is a fair latent-space baseline that
    PA reviewers will recognize, not a re-implementation of full AME.
    """

    def __init__(self, num_nodes: int, num_layers: int, emb_dim: int):
        super().__init__()
        # One latent matrix per layer
        self.layer_z = nn.ParameterDict({
            f"layer_{i}": nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
            for i in range(num_layers)
        })
        self.num_layers = num_layers
        self.emb_dim = emb_dim

    def aggregated(self) -> torch.Tensor:
        """Mean across layers — the embedding fed into downstream metrics."""
        zs = [self.layer_z[f"layer_{i}"] for i in range(self.num_layers)]
        return torch.stack(zs, dim=0).mean(dim=0)

    def edge_logits(self, layer_idx: int, src: torch.LongTensor, tgt: torch.LongTensor) -> torch.Tensor:
        z = self.layer_z[f"layer_{layer_idx}"]
        return (z[src] * z[tgt]).sum(dim=1)


def train_bilinear_latent_space(
    dataset: MultiplexTemporalDataset,
    emb_dim: int = 32,
    num_epochs: int = 200,
    lr: float = 1e-2,
    n_neg_samples: int = 1000,
    seed: int = 123,
    device: Optional[torch.device] = None,
) -> BaselineResult:
    """Train a per-year bilinear latent space model and stitch into a sequence.

    For each year, fit fresh latent positions to that year's edges. Embeddings
    for downstream comparison are the mean across layers.

    Sequential per-year fitting is intentionally simple — each year is its
    own latent-space recovery problem. This matches how AME is typically run
    on cross-sectional IR data.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    num_layers = len(dataset.layer_names)
    embeddings: Dict[int, torch.Tensor] = {}
    loss_history: List[float] = []

    print(f"Training Bilinear Latent Space baseline ({len(dataset.years)} years × {num_epochs} epochs)...")

    for y_idx, year in enumerate(dataset.years):
        snap = dataset.snapshots[year]
        model = BilinearLatentSpace(snap.num_nodes, num_layers, emb_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for l_idx, ln in enumerate(dataset.layer_names):
                if not snap.layer_mask.get(ln, False):
                    continue
                ei = snap.edge_indices[ln].to(device)
                if ei.numel() == 0:
                    continue
                # Positive edges
                src_pos, tgt_pos = ei[0], ei[1]
                logits_pos = model.edge_logits(l_idx, src_pos, tgt_pos)
                # Negative samples (random pairs)
                src_neg = torch.randint(0, snap.num_nodes, (n_neg_samples,), device=device)
                tgt_neg = torch.randint(0, snap.num_nodes, (n_neg_samples,), device=device)
                logits_neg = model.edge_logits(l_idx, src_neg, tgt_neg)
                # BCE
                labels = torch.cat([
                    torch.ones_like(logits_pos), torch.zeros_like(logits_neg),
                ])
                logits = torch.cat([logits_pos, logits_neg])
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch == num_epochs - 1:
                loss_history.append(total_loss)

        with torch.no_grad():
            embeddings[year] = model.aggregated().cpu()

        if (y_idx + 1) % max(1, len(dataset.years) // 5) == 0:
            print(f"  Year {year}: final BCE={loss_history[-1]:.4f}")

    return BaselineResult(
        model_name="bilinear_latent_space",
        loss_history=loss_history,
        yearly_embeddings=embeddings,
    )


def compare_baselines(
    dataset: MultiplexTemporalDataset,
    full_model_result: TrainingResult,
    model_config: Optional[MultiplexGNNConfig] = None,
    seq_len: int = 5,
    num_epochs: int = 100,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
    include_bilinear: bool = True,
) -> pd.DataFrame:
    """Run all baselines and compare with the full model.

    Returns a summary DataFrame.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baselines = {}

    # Static GNN
    static_result = train_static_gnn(dataset, model_config, num_epochs, device=device)
    baselines["static_gnn"] = static_result

    # Degree AR
    out_dim = model_config.emb_dim if model_config else 32
    ar_result = train_degree_ar(dataset, out_dim=out_dim, seq_len=seq_len,
                                 num_epochs=num_epochs, device=device)
    baselines["degree_ar"] = ar_result

    # Mean baseline
    mean_result = mean_baseline(dataset, dataset.years[-1])
    baselines["mean_baseline"] = mean_result

    # Bilinear latent space (AME-lite) — most relevant baseline for PA reviewers
    if include_bilinear:
        bls_result = train_bilinear_latent_space(
            dataset, emb_dim=out_dim, num_epochs=200, device=device,
        )
        baselines["bilinear_latent_space"] = bls_result

    # Compare final losses
    rows = []
    rows.append({
        "model": "multiplex_temporal_gnn",
        "final_loss": full_model_result.loss_history[-1] if full_model_result.loss_history else np.nan,
        "n_epochs": len(full_model_result.loss_history),
    })
    for name, result in baselines.items():
        rows.append({
            "model": name,
            "final_loss": result.loss_history[-1] if result.loss_history else np.nan,
            "n_epochs": len(result.loss_history),
        })

    summary = pd.DataFrame(rows)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        summary.to_csv(save_path / "baseline_comparison.csv", index=False)
        print(f"Saved baseline comparison to {save_path / 'baseline_comparison.csv'}")

    print("\n--- Baseline Comparison ---")
    print(summary.to_string(index=False))

    return summary
