"""Uncertainty quantification for the multiplex temporal GNN.

Three complementary methods, all keyed off seed 123 as the base:

1. Ensemble training (`train_ensemble`) — train N independent models with
   incremented seeds, return per-seed embeddings and a stacked tensor for
   downstream variance estimation. This is the recommended approach for
   final paper results.

2. Bootstrap counterfactual CIs (`bootstrap_counterfactual_cis`) — for a
   trained ensemble, compute mean ± percentile CIs on the dual-focal
   embedding-delta distribution across ensemble members. Reports CIs on
   USA delta, China delta, and the wedge.

3. MC dropout (`mc_dropout_predictions`) — adds dropout layers to the
   model and runs T forward passes with dropout active at inference, used
   when the user prefers a single-model uncertainty estimate.

All routines are sequential and hard-code seed = 123 as the base for
reproducibility (incremented by ensemble index).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiplex_data import MultiplexTemporalDataset, MultiplexSnapshot
from .multiplex_model import MultiplexGNNConfig, MultiplexTemporalGNN
from .training import TrainingConfig, train_model
from .counterfactual import (
    USA_CCODE,
    CHN_CCODE,
    compute_embeddedness_metrics,
)

BASE_SEED = 123


@dataclass
class EnsembleResult:
    """Trained ensemble: list of models + per-seed yearly embeddings."""
    seeds: List[int]
    models: List[MultiplexTemporalGNN]
    yearly_embeddings: List[Dict[int, torch.Tensor]]  # one dict per ensemble member
    layer_weights: List[Dict[str, float]]
    final_losses: List[float]


def train_ensemble(
    dataset: MultiplexTemporalDataset,
    n_members: int = 10,
    base_seed: int = BASE_SEED,
    model_config: Optional[MultiplexGNNConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> EnsembleResult:
    """Train an ensemble of N models with seeds = [base_seed, base_seed+1, ...].

    Sequential — no parallelism — so debugging stays straightforward.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_config is None:
        model_config = MultiplexGNNConfig(
            num_layers=len(dataset.layer_names),
            in_dim=len(dataset.layer_names),
            hidden_dim=64,
            emb_dim=32,
        )

    if train_config is None:
        train_config = TrainingConfig(
            num_epochs=100,
            patience=20,
            print_every=25,
        )

    seeds = [base_seed + i for i in range(n_members)]
    models: List[MultiplexTemporalGNN] = []
    yearly_embeddings: List[Dict[int, torch.Tensor]] = []
    layer_weights: List[Dict[str, float]] = []
    final_losses: List[float] = []

    for i, seed in enumerate(seeds):
        print(f"\n=== Ensemble member {i+1}/{n_members} (seed={seed}) ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        result = train_model(
            dataset=dataset,
            model_config=model_config,
            train_config=train_config,
            device=device,
        )
        models.append(result.model)
        yearly_embeddings.append(result.yearly_embeddings)
        layer_weights.append(result.layer_weights)
        final_losses.append(result.loss_history[-1])

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for i, seed in enumerate(seeds):
            torch.save(models[i].state_dict(), save_path / f"ensemble_seed{seed}_weights.pt")
            torch.save(yearly_embeddings[i], save_path / f"ensemble_seed{seed}_embeddings.pt")
        # Layer weights summary
        lw_rows = []
        for seed, lw in zip(seeds, layer_weights):
            row = {"seed": seed}
            row.update(lw)
            lw_rows.append(row)
        pd.DataFrame(lw_rows).to_csv(save_path / "ensemble_layer_weights.csv", index=False)
        print(f"Saved ensemble outputs to {save_path}")

    return EnsembleResult(
        seeds=seeds,
        models=models,
        yearly_embeddings=yearly_embeddings,
        layer_weights=layer_weights,
        final_losses=final_losses,
    )


# ---------------------------------------------------------------
# Embedding-level uncertainty across ensemble members
# ---------------------------------------------------------------

def stack_yearly_embeddings(
    ensemble: EnsembleResult,
    year: int,
) -> torch.Tensor:
    """Return a [n_members, num_nodes, emb_dim] tensor for a given year."""
    parts = []
    for ye in ensemble.yearly_embeddings:
        if year in ye:
            parts.append(ye[year])
    if not parts:
        raise ValueError(f"No ensemble member has embeddings for year {year}")
    return torch.stack(parts, dim=0)


def embeddedness_with_cis(
    ensemble: EnsembleResult,
    dataset: MultiplexTemporalDataset,
    year: int,
    ccode: int,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Embeddedness metrics for one (year, country) with percentile CIs across ensemble.

    Returns mean and (alpha/2, 1-alpha/2) percentile bounds for each metric.
    """
    if ccode not in dataset.ccode_to_idx:
        raise ValueError(f"ccode {ccode} not in dataset")
    focal_idx = dataset.ccode_to_idx[ccode]

    cosines: List[float] = []
    centroids: List[float] = []
    mp_dists: List[float] = []
    knns: List[float] = []

    for ye in ensemble.yearly_embeddings:
        if year not in ye:
            continue
        m = compute_embeddedness_metrics(ye[year], focal_idx, dataset.ccode_to_idx)
        cosines.append(m.mean_cosine_similarity)
        centroids.append(m.centroid_distance)
        mp_dists.append(m.major_power_distance)
        knns.append(m.knn_density)

    lo, hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    out = {
        "ccode": ccode,
        "year": year,
        "n_members": len(cosines),
        "mean_cosine_mean": float(np.mean(cosines)),
        "mean_cosine_lo": float(np.percentile(cosines, lo)),
        "mean_cosine_hi": float(np.percentile(cosines, hi)),
        "centroid_dist_mean": float(np.mean(centroids)),
        "centroid_dist_lo": float(np.percentile(centroids, lo)),
        "centroid_dist_hi": float(np.percentile(centroids, hi)),
        "mp_dist_mean": float(np.mean(mp_dists)),
        "mp_dist_lo": float(np.percentile(mp_dists, lo)),
        "mp_dist_hi": float(np.percentile(mp_dists, hi)),
        "knn_density_mean": float(np.mean(knns)),
        "knn_density_lo": float(np.percentile(knns, lo)),
        "knn_density_hi": float(np.percentile(knns, hi)),
    }
    return out


# ---------------------------------------------------------------
# Bootstrap counterfactual CIs across ensemble members
# ---------------------------------------------------------------

def _encode_history_for_member(
    model: MultiplexTemporalGNN,
    dataset: MultiplexTemporalDataset,
    seq_len: int,
    focal_year: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Encode the seq_len years ending at focal_year using a specific model."""
    years = [y for y in dataset.years if y <= focal_year]
    if len(years) < seq_len:
        raise ValueError(f"Need at least {seq_len} years, got {len(years)}")
    history: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for y in years[-seq_len:]:
            snap = dataset.snapshots[y]
            nf = snap.node_features.to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            history.append(model.encode_snapshot(nf, ei, ew, snap.layer_mask))
    return history


def _apply_partner_perturbation(
    snap: MultiplexSnapshot,
    layer_name: str,
    partner_idx: int,
    operation: str,
    n_add_edges: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.LongTensor], Dict[str, torch.FloatTensor]]:
    """Symmetric perturbation: remove n edges (or all if op=remove_all),
    add n new edges to currently-unconnected nodes (deterministic ordering).

    For symmetry between add and remove, both operate on the SAME number of
    edges (n_add_edges). The 'remove' op removes up to n_add_edges of the
    partner's existing ties (lowest-index neighbors first); 'add' adds up to
    n_add_edges new ties to unconnected nodes (lowest-index first). This
    makes the operations directly comparable in magnitude.

    The 'remove_all' op preserves the original disconnection behavior for
    backward compatibility with the wedge analysis.
    """
    modified_ei = {}
    modified_ew = {}
    for ln in snap.edge_indices:
        modified_ei[ln] = snap.edge_indices[ln].clone().to(device)
        modified_ew[ln] = snap.edge_weights[ln].clone().to(device)

    if layer_name not in modified_ei:
        return modified_ei, modified_ew

    ei = modified_ei[layer_name]
    ew = modified_ew[layer_name]

    if operation == "remove_all":
        if ei.numel() > 0:
            keep = ~((ei[0] == partner_idx) | (ei[1] == partner_idx))
            modified_ei[layer_name] = ei[:, keep]
            modified_ew[layer_name] = ew[keep]
        return modified_ei, modified_ew

    if operation == "remove":
        if ei.numel() == 0:
            return modified_ei, modified_ew
        # Find indices where the partner appears (either side)
        partner_mask = (ei[0] == partner_idx) | (ei[1] == partner_idx)
        partner_edge_indices = partner_mask.nonzero(as_tuple=False).flatten()
        if partner_edge_indices.numel() == 0:
            return modified_ei, modified_ew
        # Remove up to n_add_edges of them (lowest-index first for determinism)
        to_remove = partner_edge_indices[:n_add_edges]
        keep_mask = torch.ones(ei.size(1), dtype=torch.bool, device=device)
        keep_mask[to_remove] = False
        modified_ei[layer_name] = ei[:, keep_mask]
        modified_ew[layer_name] = ew[keep_mask]
        return modified_ei, modified_ew

    if operation == "add":
        existing = set()
        if ei.numel() > 0:
            src_mask = ei[0] == partner_idx
            existing.update(ei[1, src_mask].cpu().tolist())
            tgt_mask = ei[1] == partner_idx
            existing.update(ei[0, tgt_mask].cpu().tolist())
        num_nodes = snap.num_nodes
        unconnected = [n for n in range(num_nodes) if n != partner_idx and n not in existing]
        new_targets = unconnected[:n_add_edges]
        if not new_targets:
            return modified_ei, modified_ew
        src = [partner_idx] * len(new_targets) + new_targets
        tgt = new_targets + [partner_idx] * len(new_targets)
        new_e = torch.tensor([src, tgt], dtype=torch.long, device=device)
        new_w = torch.ones(len(src), device=device)
        if ei.numel() > 0:
            modified_ei[layer_name] = torch.cat([ei, new_e], dim=1)
            modified_ew[layer_name] = torch.cat([ew, new_w])
        else:
            modified_ei[layer_name] = new_e
            modified_ew[layer_name] = new_w
        return modified_ei, modified_ew

    raise ValueError(f"Unknown operation: {operation}")


def bootstrap_counterfactual_cis(
    ensemble: EnsembleResult,
    dataset: MultiplexTemporalDataset,
    partner_ccodes: List[int],
    layer_names: Optional[List[str]] = None,
    operations: Optional[List[str]] = None,
    n_add_edges: int = 5,
    seq_len: int = 5,
    focal_year: Optional[int] = None,
    alpha: float = 0.05,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """For each (partner, layer, operation), compute mean Δ and CIs across ensemble.

    Each ensemble member runs its own forward-pass counterfactual; CIs come
    from the percentile spread across members. This is a model-uncertainty
    bootstrap, not a data bootstrap.

    The add and remove operations are now symmetric: both modify exactly
    n_add_edges (default 5) of the partner's ties. Use operation='remove_all'
    to recover the older behavior.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if focal_year is None:
        focal_year = dataset.years[-1]
    if layer_names is None:
        layer_names = dataset.layer_names
    if operations is None:
        operations = ["add", "remove"]

    ccode_to_idx = dataset.ccode_to_idx
    if USA_CCODE not in ccode_to_idx or CHN_CCODE not in ccode_to_idx:
        raise ValueError("USA (2) and China (710) must be in the dataset")
    usa_idx = ccode_to_idx[USA_CCODE]
    chn_idx = ccode_to_idx[CHN_CCODE]

    # Pre-compute per-member baseline + history
    target_snap = dataset.snapshots[focal_year]
    member_history: List[List[torch.Tensor]] = []
    member_baseline_pred: List[torch.Tensor] = []

    for model in ensemble.models:
        model.to(device)
        history = _encode_history_for_member(model, dataset, seq_len, focal_year, device)
        member_history.append(history)
        with torch.no_grad():
            member_baseline_pred.append(model.forward_temporal(history[-seq_len:]))

    rows = []
    total = len(partner_ccodes) * len(layer_names) * len(operations)
    done = 0

    for partner_ccode in partner_ccodes:
        if partner_ccode not in ccode_to_idx:
            continue
        partner_idx = ccode_to_idx[partner_ccode]

        for layer_name in layer_names:
            if not target_snap.layer_mask.get(layer_name, False):
                continue

            for op in operations:
                # Per-member deltas
                usa_deltas = []
                chn_deltas = []
                wedges = []

                for m_idx, model in enumerate(ensemble.models):
                    mod_ei, mod_ew = _apply_partner_perturbation(
                        target_snap, layer_name, partner_idx, op, n_add_edges, device,
                    )
                    nf = target_snap.node_features.to(device)
                    with torch.no_grad():
                        mod_emb = model.encode_snapshot(
                            nf, mod_ei, mod_ew, target_snap.layer_mask,
                        )
                        history = member_history[m_idx]
                        mod_history = list(history[:-1]) + [mod_emb]
                        cf_pred = model.forward_temporal(mod_history[-seq_len:])

                        baseline_pred = member_baseline_pred[m_idx]
                        usa_b = compute_embeddedness_metrics(baseline_pred, usa_idx, ccode_to_idx)
                        chn_b = compute_embeddedness_metrics(baseline_pred, chn_idx, ccode_to_idx)
                        usa_c = compute_embeddedness_metrics(cf_pred, usa_idx, ccode_to_idx)
                        chn_c = compute_embeddedness_metrics(cf_pred, chn_idx, ccode_to_idx)

                    u_d = usa_c.mean_cosine_similarity - usa_b.mean_cosine_similarity
                    c_d = chn_c.mean_cosine_similarity - chn_b.mean_cosine_similarity
                    usa_deltas.append(u_d)
                    chn_deltas.append(c_d)
                    wedges.append(u_d - c_d)

                lo, hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
                rows.append({
                    "partner_ccode": partner_ccode,
                    "layer": layer_name,
                    "operation": op,
                    "n_add_edges": n_add_edges,
                    "n_members": len(usa_deltas),
                    "usa_delta_mean": float(np.mean(usa_deltas)),
                    "usa_delta_lo": float(np.percentile(usa_deltas, lo)),
                    "usa_delta_hi": float(np.percentile(usa_deltas, hi)),
                    "chn_delta_mean": float(np.mean(chn_deltas)),
                    "chn_delta_lo": float(np.percentile(chn_deltas, lo)),
                    "chn_delta_hi": float(np.percentile(chn_deltas, hi)),
                    "wedge_mean": float(np.mean(wedges)),
                    "wedge_lo": float(np.percentile(wedges, lo)),
                    "wedge_hi": float(np.percentile(wedges, hi)),
                    # Significance flag: CI excludes zero
                    "wedge_significant": bool(
                        np.percentile(wedges, lo) > 0 or np.percentile(wedges, hi) < 0
                    ),
                })
                done += 1
                if done % 50 == 0:
                    print(f"  Bootstrap CI: {done}/{total} (partner, layer, op) cells")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# MC dropout: alternative to ensemble for single-model uncertainty
# ---------------------------------------------------------------

class MCDropoutMultiplexGNN(nn.Module):
    """Wrapper around MultiplexTemporalGNN that adds dropout for MC uncertainty.

    Hard-coded p=0.1 dropout on per-layer embeddings before aggregation, and
    on the GRU hidden state. Activate at inference by calling model.train()
    on the dropout layers only (we do this via a context manager helper).
    """
    def __init__(self, base: MultiplexTemporalGNN, dropout_p: float = 0.1):
        super().__init__()
        self.base = base
        self.dropout_p = dropout_p
        self.layer_dropout = nn.Dropout(dropout_p)
        self.temporal_dropout = nn.Dropout(dropout_p)

    def encode_snapshot(self, node_features, edge_indices, edge_weights, layer_mask):
        # Run per-layer encoders, apply dropout, aggregate
        layer_embs = []
        mask_list = []
        for name in self.base.layer_names:
            available = layer_mask.get(name, False)
            mask_list.append(available)
            if available and name in edge_indices and edge_indices[name].numel() > 0:
                emb = self.base.layer_encoders[name](
                    node_features, edge_indices[name], edge_weights.get(name),
                )
                emb = self.layer_dropout(emb)
            else:
                emb = torch.zeros(
                    node_features.size(0), self.base.config.emb_dim,
                    device=node_features.device,
                )
            layer_embs.append(emb)
        return self.base.aggregator(layer_embs, mask_list)

    def forward_temporal(self, embedding_history):
        seq = torch.stack(embedding_history, dim=1)
        h, _ = self.base.temporal.gru(seq)
        h_last = self.temporal_dropout(h[:, -1, :])
        return self.base.temporal.head(h_last)


def mc_dropout_predictions(
    base_model: MultiplexTemporalGNN,
    dataset: MultiplexTemporalDataset,
    n_samples: int = 30,
    dropout_p: float = 0.1,
    seq_len: int = 5,
    focal_year: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a [n_samples, num_nodes, emb_dim] tensor of predictions.

    Activates dropout at inference time on a wrapper of the trained model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if focal_year is None:
        focal_year = dataset.years[-1]

    base_model = base_model.to(device)
    mc = MCDropoutMultiplexGNN(base_model, dropout_p=dropout_p).to(device)
    # Activate dropout layers only
    mc.layer_dropout.train()
    mc.temporal_dropout.train()
    # Keep GRU and GCN layers in eval mode
    mc.base.eval()

    years = [y for y in dataset.years if y <= focal_year]
    history_base: List[torch.Tensor] = []
    with torch.no_grad():
        for y in years[-seq_len:]:
            snap = dataset.snapshots[y]
            nf = snap.node_features.to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            history_base.append(mc.encode_snapshot(nf, ei, ew, snap.layer_mask))

    samples = []
    torch.manual_seed(BASE_SEED)
    for _ in range(n_samples):
        # Re-sample dropout masks per draw
        with torch.no_grad():
            history = []
            for y in years[-seq_len:]:
                snap = dataset.snapshots[y]
                nf = snap.node_features.to(device)
                ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
                ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
                history.append(mc.encode_snapshot(nf, ei, ew, snap.layer_mask))
            pred = mc.forward_temporal(history[-seq_len:])
            samples.append(pred.cpu())

    return torch.stack(samples, dim=0)
