"""Phase 1 diagnostic ablation harness.

Five small, isolated experiments designed to identify which component of
the multiplex GNN training setup is preventing the model from learning
informative embeddings on the real Roar dataset (where link prediction
collapses to chance baseline despite three rounds of fixes).

The five experiments, in priority order:

  Exp 1 — STATIC GCN + INNER-PRODUCT BCE on a single year.
          Tests whether the encoder alone (no GRU, no temporal) can
          learn link-predictive embeddings. If this fails, the
          GCN+inner-product-BCE combo is the root problem.

  Exp 2 — STATIC GCN + INFONCE on a single year.
          Same encoder, contrastive loss instead of BCE. Modern
          standard for graph SSL. Tests whether the loss formulation
          is the issue.

  Exp 3 — GAT / GraphSAGE BACKBONE + INNER-PRODUCT BCE.
          Replace our hand-rolled SimpleGCNLayerSparse with
          torch_geometric's GAT or GraphSAGE. Tests whether our
          encoder implementation has a subtle bug.

  Exp 4 — FULL TEMPORAL MODEL with smooth + anti-collapse OFF.
          Run the standard train_model with lambda_smooth=0 and
          lambda_anti_collapse=0. Tests whether competing losses are
          preventing the encoder from learning.

  Exp 5 — DELIBERATE OVERFIT on a single year.
          Tiny model, 500 epochs, no regularization. If even maximal
          overfitting can't drive link AUC above 0.6 on training data,
          the architecture is fundamentally broken.

Each experiment returns a dict with: name, final_link_auc, final_mean_norm,
loss_history, link_loss_history, n_epochs_run, notes. The runner aggregates
results into a CSV and a markdown summary.

Hardcoded for clarity (per project coding preferences): each experiment is
its own function, sequential, seed=123 throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiplex_data import (
    MultiplexTemporalDataset,
    MultiplexSnapshot,
    discover_layers,
    build_global_node_index,
    build_multiplex_dataset,
)
from .multiplex_model import (
    MultiplexTemporalGNN,
    MultiplexGNNConfig,
    LayerEncoder,
    LayerAggregator,
    HAS_TORCH_GEOMETRIC,
)
from .training import (
    TrainingConfig,
    train_model,
    _compute_link_loss,
    _merge_edge_indices,
)

SEED = 123


# ---------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------

def compute_link_auc(
    emb: torch.Tensor,
    edge_index: torch.LongTensor,
    num_nodes: int,
    n_neg: int = 1000,
) -> float:
    """AUC of inner-product scores at distinguishing positives from random negatives.

    Sigmoid-free; uses raw scores. Wilcoxon-Mann-Whitney rank statistic.
    """
    if edge_index.numel() == 0:
        return 0.5
    scores = emb @ emb.T  # raw (no sigmoid)

    src, tgt = edge_index[0], edge_index[1]
    n_pos = min(len(src), n_neg)
    perm = torch.randperm(len(src))[:n_pos]
    pos_scores = scores[src[perm], tgt[perm]].detach().cpu().numpy()

    neg_src = torch.randint(0, num_nodes, (n_neg,))
    neg_tgt = torch.randint(0, num_nodes, (n_neg,))
    neg_scores = scores[neg_src, neg_tgt].detach().cpu().numpy()

    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    order = np.argsort(-all_scores)
    sorted_labels = all_labels[order]
    n_p = sorted_labels.sum()
    n_n = len(sorted_labels) - n_p
    if n_p == 0 or n_n == 0:
        return 0.5
    cum_neg = np.cumsum(1 - sorted_labels)
    return float(np.sum(sorted_labels * cum_neg) / (n_p * n_n))


def pick_target_year_snapshot(
    dataset: MultiplexTemporalDataset,
    year: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
):
    """Return (snapshot, merged_edge_index) for one target year.

    Defaults to the most recent year with all layers available.
    """
    if year is None:
        # Pick the most recent year with all layers available
        for y in reversed(dataset.years):
            snap = dataset.snapshots[y]
            if all(snap.layer_mask.values()):
                year = y
                break
        if year is None:
            year = dataset.years[-1]
    snap = dataset.snapshots[year]
    merged_ei = _merge_edge_indices(
        {k: v.to(device) for k, v in snap.edge_indices.items()},
        snap.layer_mask,
    )
    return year, snap, merged_ei


# ---------------------------------------------------------------
# Experiment 1: Static GCN + inner-product BCE on one year
# ---------------------------------------------------------------

class StaticEncoder(nn.Module):
    """Just the per-layer GCN encoders + attention aggregator. No GRU."""

    def __init__(self, layer_names: List[str], in_dim: int,
                 hidden_dim: int = 64, emb_dim: int = 32, use_pyg: bool = False):
        super().__init__()
        self.layer_names = layer_names
        self.layer_encoders = nn.ModuleDict({
            name: LayerEncoder(in_dim, hidden_dim, emb_dim, use_pyg=use_pyg)
            for name in layer_names
        })
        self.aggregator = LayerAggregator(len(layer_names))
        self.emb_dim = emb_dim

    def forward(self, node_features, edge_indices, edge_weights, layer_mask):
        layer_embs = []
        mask_list = []
        for name in self.layer_names:
            available = layer_mask.get(name, False)
            mask_list.append(available)
            if available and name in edge_indices and edge_indices[name].numel() > 0:
                emb = self.layer_encoders[name](
                    node_features, edge_indices[name], edge_weights.get(name),
                )
            else:
                emb = torch.zeros(
                    node_features.size(0), self.emb_dim,
                    device=node_features.device,
                )
            layer_embs.append(emb)
        return self.aggregator(layer_embs, mask_list)


def exp1_static_gcn_bce(
    dataset: MultiplexTemporalDataset,
    year: Optional[int] = None,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    emb_dim: int = 32,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> Dict:
    """Static GCN encoder + inner-product BCE link loss on a single year."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    year, snap, merged_ei = pick_target_year_snapshot(dataset, year, device)
    print(f"[Exp 1] Static GCN + BCE — year={year}, n_edges={merged_ei.size(1)}")

    model = StaticEncoder(
        layer_names=dataset.layer_names,
        in_dim=len(dataset.layer_names),
        hidden_dim=hidden_dim, emb_dim=emb_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    nf = snap.node_features.to(device)
    ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
    ew = {k: v.to(device) for k, v in snap.edge_weights.items()}

    loss_history: List[float] = []
    norm_history: List[float] = []

    for epoch in range(num_epochs):
        model.train()
        emb = model(nf, ei, ew, snap.layer_mask)
        loss = _compute_link_loss(emb, merged_ei, dataset.num_nodes, sample_size=500)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
        norm_history.append(float(emb.norm(dim=1).mean().detach().cpu()))
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{num_epochs} link_bce={loss.item():.4f} ||z||={norm_history[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        final_emb = model(nf, ei, ew, snap.layer_mask)
    auc = compute_link_auc(final_emb, merged_ei, dataset.num_nodes)
    return {
        "name": "exp1_static_gcn_bce",
        "year": year,
        "n_edges": int(merged_ei.size(1)),
        "final_link_loss": loss_history[-1],
        "final_link_auc": auc,
        "final_mean_norm": norm_history[-1],
        "loss_history": loss_history,
        "norm_history": norm_history,
        "n_epochs_run": num_epochs,
        "notes": "If link_auc > 0.6 here, the encoder works; the temporal pipeline is the issue.",
    }


# ---------------------------------------------------------------
# Experiment 2: Static GCN + InfoNCE
# ---------------------------------------------------------------

def _infonce_loss(
    emb: torch.Tensor,
    edge_index: torch.LongTensor,
    num_neg_per_pos: int = 10,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Sampled InfoNCE / NT-Xent style loss for graph link prediction.

    For each positive (i, j) edge: numerator = exp(z_i . z_j / T),
    denominator = numerator + sum over K random negative j' of exp(z_i . z_j' / T).
    Loss = -mean log(numerator / denominator).
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=emb.device, requires_grad=True)
    n_nodes, _ = emb.shape
    src = edge_index[0]
    tgt = edge_index[1]
    n_pos = src.size(0)

    # Positive scores
    pos_scores = (emb[src] * emb[tgt]).sum(dim=1) / temperature  # [n_pos]

    # Negative samples
    neg_tgt = torch.randint(0, n_nodes, (n_pos, num_neg_per_pos), device=emb.device)
    neg_scores = (emb[src].unsqueeze(1) * emb[neg_tgt]).sum(dim=2) / temperature  # [n_pos, K]

    # log-sum-exp denominator
    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [n_pos, 1+K]
    log_denominator = torch.logsumexp(all_scores, dim=1)
    log_numerator = pos_scores
    return -(log_numerator - log_denominator).mean()


def exp2_static_gcn_infonce(
    dataset: MultiplexTemporalDataset,
    year: Optional[int] = None,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    emb_dim: int = 32,
    num_neg_per_pos: int = 10,
    temperature: float = 0.5,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> Dict:
    """Static GCN encoder + InfoNCE contrastive loss on a single year."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    year, snap, merged_ei = pick_target_year_snapshot(dataset, year, device)
    print(f"[Exp 2] Static GCN + InfoNCE — year={year}, K={num_neg_per_pos}, T={temperature}")

    model = StaticEncoder(
        layer_names=dataset.layer_names,
        in_dim=len(dataset.layer_names),
        hidden_dim=hidden_dim, emb_dim=emb_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    nf = snap.node_features.to(device)
    ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
    ew = {k: v.to(device) for k, v in snap.edge_weights.items()}

    loss_history: List[float] = []
    norm_history: List[float] = []

    for epoch in range(num_epochs):
        model.train()
        emb = model(nf, ei, ew, snap.layer_mask)
        loss = _infonce_loss(emb, merged_ei, num_neg_per_pos, temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
        norm_history.append(float(emb.norm(dim=1).mean().detach().cpu()))
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{num_epochs} infonce={loss.item():.4f} ||z||={norm_history[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        final_emb = model(nf, ei, ew, snap.layer_mask)
    auc = compute_link_auc(final_emb, merged_ei, dataset.num_nodes)
    return {
        "name": "exp2_static_gcn_infonce",
        "year": year,
        "n_edges": int(merged_ei.size(1)),
        "final_link_loss": loss_history[-1],
        "final_link_auc": auc,
        "final_mean_norm": norm_history[-1],
        "loss_history": loss_history,
        "norm_history": norm_history,
        "n_epochs_run": num_epochs,
        "notes": "If exp2 > exp1 by a clear margin, the BCE-vs-InfoNCE loss is the issue.",
    }


# ---------------------------------------------------------------
# Experiment 3: GAT / GraphSAGE backbone (torch_geometric)
# ---------------------------------------------------------------

class StaticPyGEncoder(nn.Module):
    """Per-layer GAT-style encoders from torch_geometric. Skipped if PyG missing."""

    def __init__(self, layer_names: List[str], in_dim: int,
                 hidden_dim: int = 64, emb_dim: int = 32, use_gat: bool = True):
        super().__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise RuntimeError("torch_geometric not installed; skip exp 3.")
        from torch_geometric.nn import GATConv, SAGEConv
        Conv = GATConv if use_gat else SAGEConv
        self.layer_names = layer_names
        self.conv1 = nn.ModuleDict({
            name: Conv(in_dim, hidden_dim) for name in layer_names
        })
        self.conv2 = nn.ModuleDict({
            name: Conv(hidden_dim, emb_dim) for name in layer_names
        })
        self.aggregator = LayerAggregator(len(layer_names))
        self.emb_dim = emb_dim

    def forward(self, node_features, edge_indices, edge_weights, layer_mask):
        layer_embs = []
        mask_list = []
        for name in self.layer_names:
            available = layer_mask.get(name, False)
            mask_list.append(available)
            if available and name in edge_indices and edge_indices[name].numel() > 0:
                h = self.conv1[name](node_features, edge_indices[name])
                h = F.relu(h)
                h = self.conv2[name](h, edge_indices[name])
            else:
                h = torch.zeros(
                    node_features.size(0), self.emb_dim,
                    device=node_features.device,
                )
            layer_embs.append(h)
        return self.aggregator(layer_embs, mask_list)


def exp3_pyg_backbone_bce(
    dataset: MultiplexTemporalDataset,
    year: Optional[int] = None,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    emb_dim: int = 32,
    use_gat: bool = True,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> Dict:
    """GAT (or SAGE) encoder from torch_geometric + inner-product BCE."""
    if not HAS_TORCH_GEOMETRIC:
        return {
            "name": "exp3_pyg_backbone_bce",
            "skipped": True,
            "notes": "torch_geometric not installed; cannot run.",
        }
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    year, snap, merged_ei = pick_target_year_snapshot(dataset, year, device)
    backbone = "GAT" if use_gat else "SAGE"
    print(f"[Exp 3] PyG {backbone} + BCE — year={year}")

    model = StaticPyGEncoder(
        layer_names=dataset.layer_names,
        in_dim=len(dataset.layer_names),
        hidden_dim=hidden_dim, emb_dim=emb_dim, use_gat=use_gat,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    nf = snap.node_features.to(device)
    ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
    ew = {k: v.to(device) for k, v in snap.edge_weights.items()}

    loss_history: List[float] = []
    norm_history: List[float] = []

    for epoch in range(num_epochs):
        model.train()
        emb = model(nf, ei, ew, snap.layer_mask)
        loss = _compute_link_loss(emb, merged_ei, dataset.num_nodes, sample_size=500)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
        norm_history.append(float(emb.norm(dim=1).mean().detach().cpu()))
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{num_epochs} link_bce={loss.item():.4f} ||z||={norm_history[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        final_emb = model(nf, ei, ew, snap.layer_mask)
    auc = compute_link_auc(final_emb, merged_ei, dataset.num_nodes)
    return {
        "name": "exp3_pyg_backbone_bce",
        "backbone": backbone,
        "year": year,
        "n_edges": int(merged_ei.size(1)),
        "final_link_loss": loss_history[-1],
        "final_link_auc": auc,
        "final_mean_norm": norm_history[-1],
        "loss_history": loss_history,
        "norm_history": norm_history,
        "n_epochs_run": num_epochs,
        "notes": "If exp3 > exp1, our hand-rolled SimpleGCNLayerSparse has issues.",
    }


# ---------------------------------------------------------------
# Experiment 4: Full temporal model with smooth + anti-collapse OFF
# ---------------------------------------------------------------

def exp4_full_model_no_competing_losses(
    dataset: MultiplexTemporalDataset,
    num_epochs: int = 100,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> Dict:
    """Run the standard train_model with smooth + anti-collapse zeroed out."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("[Exp 4] Full temporal model with lambda_smooth=0, lambda_anti_collapse=0")
    train_cfg = TrainingConfig(
        num_epochs=num_epochs,
        lambda_link=1.0,
        lambda_smooth=0.0,
        lambda_anti_collapse=0.0,
        patience=20,
        print_every=25,
    )
    result = train_model(dataset, train_config=train_cfg, device=device)

    # Score link AUC on the final year
    final_year = dataset.years[-1]
    final_emb = result.yearly_embeddings[final_year].to(device)
    snap = dataset.snapshots[final_year]
    merged_ei = _merge_edge_indices(
        {k: v.to(device) for k, v in snap.edge_indices.items()},
        snap.layer_mask,
    )
    auc = compute_link_auc(final_emb, merged_ei, dataset.num_nodes)
    final_link_loss = result.epoch_details[-1]["link_loss"] if result.epoch_details else float("nan")
    final_norm = float(final_emb.norm(dim=1).mean().cpu())
    return {
        "name": "exp4_full_model_no_competing_losses",
        "final_link_loss": final_link_loss,
        "final_link_auc": auc,
        "final_mean_norm": final_norm,
        "loss_history": [d["total_loss"] for d in result.epoch_details],
        "n_epochs_run": len(result.epoch_details),
        "notes": "If exp4 link_auc > exp1, smooth+anti-collapse losses were preventing learning.",
    }


# ---------------------------------------------------------------
# Experiment 5: Single-year deliberate overfit
# ---------------------------------------------------------------

def exp5_overfit_one_year(
    dataset: MultiplexTemporalDataset,
    year: Optional[int] = None,
    num_epochs: int = 1000,
    learning_rate: float = 5e-3,
    hidden_dim: int = 32,
    emb_dim: int = 16,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> Dict:
    """Deliberate overfit. Tiny model, no regularization, many epochs.

    If even this can't get link AUC above 0.6 on the training data, the
    architecture is fundamentally unable to learn link structure at all.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    year, snap, merged_ei = pick_target_year_snapshot(dataset, year, device)
    print(f"[Exp 5] Overfit one year — year={year}, {num_epochs} epochs")

    model = StaticEncoder(
        layer_names=dataset.layer_names,
        in_dim=len(dataset.layer_names),
        hidden_dim=hidden_dim, emb_dim=emb_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    nf = snap.node_features.to(device)
    ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
    ew = {k: v.to(device) for k, v in snap.edge_weights.items()}

    loss_history: List[float] = []
    norm_history: List[float] = []
    auc_checkpoints: List[float] = []

    for epoch in range(num_epochs):
        model.train()
        emb = model(nf, ei, ew, snap.layer_mask)
        loss = _compute_link_loss(emb, merged_ei, dataset.num_nodes, sample_size=500)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
        norm_history.append(float(emb.norm(dim=1).mean().detach().cpu()))
        if (epoch + 1) % 100 == 0 or epoch == 0:
            with torch.no_grad():
                e_eval = model(nf, ei, ew, snap.layer_mask)
                a = compute_link_auc(e_eval, merged_ei, dataset.num_nodes)
                auc_checkpoints.append(a)
            print(f"  epoch {epoch+1}/{num_epochs} bce={loss.item():.4f} ||z||={norm_history[-1]:.4f} auc={a:.4f}")

    model.eval()
    with torch.no_grad():
        final_emb = model(nf, ei, ew, snap.layer_mask)
    auc = compute_link_auc(final_emb, merged_ei, dataset.num_nodes)
    return {
        "name": "exp5_overfit_one_year",
        "year": year,
        "n_edges": int(merged_ei.size(1)),
        "final_link_loss": loss_history[-1],
        "final_link_auc": auc,
        "final_mean_norm": norm_history[-1],
        "auc_checkpoints": auc_checkpoints,
        "loss_history": loss_history,
        "norm_history": norm_history,
        "n_epochs_run": num_epochs,
        "notes": "If even this can't reach AUC 0.6, the architecture cannot learn this task.",
    }


# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------

def run_all_diagnostics(
    dataset: MultiplexTemporalDataset,
    out_dir: str,
    skip_pyg: bool = False,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Run all five experiments sequentially. Save results + summary."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []

    print("\n" + "=" * 60)
    print("PHASE 1 DIAGNOSTIC ABLATION")
    print("=" * 60)

    print("\n--- Experiment 1: Static GCN + BCE ---")
    r1 = exp1_static_gcn_bce(dataset, device=device)
    results.append(r1)

    print("\n--- Experiment 2: Static GCN + InfoNCE ---")
    r2 = exp2_static_gcn_infonce(dataset, device=device)
    results.append(r2)

    if HAS_TORCH_GEOMETRIC and not skip_pyg:
        print("\n--- Experiment 3: PyG GAT backbone + BCE ---")
        r3 = exp3_pyg_backbone_bce(dataset, device=device)
        results.append(r3)
    else:
        print("\n--- Experiment 3: SKIPPED (torch_geometric unavailable) ---")
        results.append({"name": "exp3_pyg_backbone_bce", "skipped": True})

    print("\n--- Experiment 4: Full model, no competing losses ---")
    r4 = exp4_full_model_no_competing_losses(dataset, device=device)
    results.append(r4)

    print("\n--- Experiment 5: Deliberate overfit ---")
    r5 = exp5_overfit_one_year(dataset, device=device)
    results.append(r5)

    # Save full results
    torch.save(results, out_path / "diagnostic_results.pt")

    # Build summary CSV
    rows = []
    for r in results:
        if r.get("skipped"):
            rows.append({"name": r["name"], "skipped": True})
            continue
        rows.append({
            "name": r["name"],
            "skipped": False,
            "final_link_loss": r.get("final_link_loss"),
            "final_link_auc": r.get("final_link_auc"),
            "final_mean_norm": r.get("final_mean_norm"),
            "n_epochs_run": r.get("n_epochs_run"),
            "notes": r.get("notes", ""),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_path / "diagnostic_summary.csv", index=False)

    # Build markdown report
    lines = ["# Phase 1 Diagnostic Ablation — Summary\n"]
    lines.append(f"**Dataset:** {dataset.num_nodes} nodes, {len(dataset.years)} years, "
                 f"{len(dataset.layer_names)} layers")
    lines.append(f"**Layers:** {dataset.layer_names}\n")
    lines.append("| Experiment | link_auc | mean_norm | epochs | takeaway |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        if r.get("skipped"):
            lines.append(f"| {r['name']} | — | — | — | (skipped) |")
            continue
        auc = r.get("final_link_auc", float("nan"))
        norm = r.get("final_mean_norm", float("nan"))
        n = r.get("n_epochs_run", 0)
        verdict = "GOOD" if auc and auc > 0.7 else ("WEAK" if auc and auc > 0.55 else "CHANCE")
        lines.append(
            f"| {r['name']} | {auc:.3f} ({verdict}) | {norm:.3f} | {n} | {r.get('notes', '')[:80]} |"
        )
    lines.append("\n## Decision rule\n")
    lines.append("- **If exp1 (BCE) gives AUC > 0.7** → encoder works alone; the temporal pipeline is the problem.")
    lines.append("- **If exp1 fails AND exp2 (InfoNCE) gives AUC > 0.7** → the loss formulation is the issue. Switch to InfoNCE.")
    lines.append("- **If exp1 + exp2 both fail AND exp3 (PyG GAT) gives AUC > 0.7** → our hand-rolled GCN has a bug.")
    lines.append("- **If exp1-3 fail AND exp4 (no smooth/anti-collapse) gives AUC > 0.7** → competing losses were the issue.")
    lines.append("- **If even exp5 (overfit) cannot exceed AUC 0.6** → the architecture cannot learn this task; consider dropping the GNN entirely and using AME or latent space models as the embedding source for the wedge framework.")
    (out_path / "diagnostic_summary.md").write_text("\n".join(lines))

    print("\n" + "=" * 60)
    print(f"SAVED: {out_path / 'diagnostic_summary.csv'}")
    print(f"SAVED: {out_path / 'diagnostic_summary.md'}")
    print(f"SAVED: {out_path / 'diagnostic_results.pt'}")
    print("=" * 60)
    return summary
