"""Diagnostic v3 — sparse strategic-alignment layers + stratified-negatives AUC.

What changed from v2 (2026-05-07 plan):

  1. **Layer subsetting.** Only the sparse, strategically-meaningful layers
     are loaded: defensive alliances, offensive alliances, DCA, FTA,
     PTA-goods, customs unions, EIA. Trade and IGO (near-fully-connected)
     are explicitly excluded — those layers caused link prediction to be
     measurement-degenerate in v2 (every random "negative" was a true
     positive).

  2. **Stratified-negatives AUC.** When evaluating link AUC, negative
     pairs are sampled ONLY from the set of pairs that are NOT in any
     positive edge. Previously we sampled random pairs, which on dense
     layers produced AUC values stuck at 0.5000 regardless of model
     quality (the negative class was contaminated with positives).

  3. **Tie filtering.** Layer files are loaded in full-panel form
     (tie=0 for non-edges, tie=1 for edges). We filter to tie==1 before
     building edge_index tensors so the message-passing graph is sparse.

This module is self-contained — it loads layers directly from the
data/processed CSVs without going through multiplex_data.discover_layers.
That keeps multiplex_data.py untouched while we test the architecture
hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .heterogeneous_model import (
    HeterogeneousEncoder,
    HeterogeneousEncoderConfig,
)
from .multiplex_data import (
    MultiplexSnapshot,
    MultiplexTemporalDataset,
    compute_degree_features,
)
from .training_v2 import infonce_loss

SEED = 123

# Sparse strategic layers we want to use. File names are matched flexibly
# (looking for any file whose name contains one of these substrings).
STRATEGIC_LAYER_PATTERNS = {
    "defensive_alliances": ["alliances_defensive_offensive", "defensive_alliances"],
    "offensive_alliances": ["alliances_defensive_offensive", "offensive_alliances"],
    "dca":   ["dca"],
    "fta":   ["fta"],
    "pta":   ["pta"],
    "cu":    ["cu_undirected", "customs_union"],
    "eia":   ["eia_undirected"],
}


# ---------------------------------------------------------------
# Sparse layer loading
# ---------------------------------------------------------------

def _load_sparse_layer(path: Path) -> pd.DataFrame:
    """Read a layer CSV and filter to tie==1 rows (edges only)."""
    df = pd.read_csv(path)
    # Normalize column names like the existing loader
    rename = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "year":
            rename[c] = "year"
        elif cl in ("source_ccode", "ccode1", "src", "source"):
            rename[c] = "source_ccode"
        elif cl in ("target_ccode", "ccode2", "dst", "target"):
            rename[c] = "target_ccode"
        elif cl in ("tie", "value", "weight", "value_raw"):
            rename[c] = "tie"
    df = df.rename(columns=rename)
    required = ["year", "source_ccode", "target_ccode", "tie"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    df = df[required].copy()
    df["year"] = df["year"].astype(int)
    df["source_ccode"] = df["source_ccode"].astype(int)
    df["target_ccode"] = df["target_ccode"].astype(int)
    df["tie"] = df["tie"].astype(float)
    df = df[df["tie"] > 0].reset_index(drop=True)
    return df


def discover_sparse_layers(
    data_dir: Path,
    requested: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Find and load the sparse strategic layer CSVs in data_dir.

    `requested` lists the canonical layer names to look for (a subset of
    STRATEGIC_LAYER_PATTERNS keys). Defaults to all of them.
    """
    if requested is None:
        requested = list(STRATEGIC_LAYER_PATTERNS.keys())

    layers: Dict[str, pd.DataFrame] = {}
    files = list(data_dir.glob("*.csv"))
    print(f"  Scanning {len(files)} CSV files in {data_dir}")

    for layer_name in requested:
        patterns = STRATEGIC_LAYER_PATTERNS[layer_name]
        match = None
        for f in files:
            fname = f.name.lower()
            if any(p in fname for p in patterns) and "weighted" not in fname:
                match = f
                break
        if match is None:
            print(f"    SKIP {layer_name} (no matching CSV)")
            continue
        df = _load_sparse_layer(match)
        layers[layer_name] = df
        print(f"    {layer_name}: {match.name} -> {len(df):,} positive ties, "
              f"years {df['year'].min()}-{df['year'].max()}")
    return layers


def build_sparse_dataset(
    layer_dfs: Dict[str, pd.DataFrame],
    year_range: Optional[Tuple[int, int]] = None,
) -> MultiplexTemporalDataset:
    """Build a MultiplexTemporalDataset from sparse layer DataFrames.

    Self-contained — doesn't go through multiplex_data.build_multiplex_dataset
    so we can keep that file untouched.
    """
    # Universe of countries
    all_ccodes: set[int] = set()
    for df in layer_dfs.values():
        all_ccodes |= set(df["source_ccode"].unique())
        all_ccodes |= set(df["target_ccode"].unique())
    sorted_ccodes = sorted(all_ccodes)
    ccode_to_idx = {cc: i for i, cc in enumerate(sorted_ccodes)}
    idx_to_ccode = {i: cc for cc, i in ccode_to_idx.items()}
    num_nodes = len(ccode_to_idx)
    nodes_df = pd.DataFrame({"ccode": sorted_ccodes})

    # Year set
    all_years: set[int] = set()
    for df in layer_dfs.values():
        all_years |= set(df["year"].unique())
    if year_range is not None:
        years = sorted(y for y in range(year_range[0], year_range[1] + 1)
                       if y in all_years)
    else:
        years = sorted(all_years)

    layer_names = list(layer_dfs.keys())
    snapshots: Dict[int, MultiplexSnapshot] = {}
    for year in years:
        edge_indices: Dict[str, torch.LongTensor] = {}
        edge_weights: Dict[str, torch.FloatTensor] = {}
        layer_mask: Dict[str, bool] = {}
        for ln in layer_names:
            df = layer_dfs[ln]
            year_df = df[df["year"] == year]
            if len(year_df) == 0:
                edge_indices[ln] = torch.zeros(2, 0, dtype=torch.long)
                edge_weights[ln] = torch.zeros(0)
                layer_mask[ln] = False
                continue
            src = year_df["source_ccode"].map(ccode_to_idx)
            tgt = year_df["target_ccode"].map(ccode_to_idx)
            valid = src.notna() & tgt.notna()
            src = src[valid].astype(int).values
            tgt = tgt[valid].astype(int).values
            weights = year_df["tie"][valid.values].astype(np.float32).values
            if len(src) == 0:
                edge_indices[ln] = torch.zeros(2, 0, dtype=torch.long)
                edge_weights[ln] = torch.zeros(0)
                layer_mask[ln] = False
                continue
            edge_indices[ln] = torch.tensor(np.stack([src, tgt], axis=0),
                                             dtype=torch.long)
            edge_weights[ln] = torch.tensor(weights, dtype=torch.float32)
            layer_mask[ln] = True

        node_features = compute_degree_features(
            edge_indices, edge_weights, layer_mask, num_nodes, layer_names,
        )
        snapshots[year] = MultiplexSnapshot(
            year=year, num_nodes=num_nodes,
            edge_indices=edge_indices, edge_weights=edge_weights,
            layer_mask=layer_mask, node_features=node_features,
        )

    return MultiplexTemporalDataset(
        years=years, num_nodes=num_nodes, layer_names=layer_names,
        ccode_to_idx=ccode_to_idx, idx_to_ccode=idx_to_ccode,
        snapshots=snapshots, nodes_df=nodes_df,
    )


# ---------------------------------------------------------------
# Stratified-negatives AUC
# ---------------------------------------------------------------

def stratified_link_auc(
    emb: torch.Tensor,
    edge_index: torch.LongTensor,
    num_nodes: int,
    n_neg: int = 1000,
    seed: int = SEED,
) -> float:
    """AUC of inner-product scores at distinguishing positives from
    sampled negatives that are guaranteed to NOT be in the positive set.

    Builds an N×N boolean mask of positive pairs (symmetric), oversamples
    candidate negatives, then keeps only those whose mask entry is False.
    """
    if edge_index.numel() == 0:
        return 0.5
    g = torch.Generator().manual_seed(seed)

    # Build symmetric positive mask
    mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    mask[edge_index[0], edge_index[1]] = True
    mask[edge_index[1], edge_index[0]] = True
    # exclude self-loops from negatives
    diag_mask = torch.eye(num_nodes, dtype=torch.bool)

    # Sample candidates with replacement, filter
    over_n = max(n_neg * 8, 5000)
    src_cand = torch.randint(0, num_nodes, (over_n,), generator=g)
    tgt_cand = torch.randint(0, num_nodes, (over_n,), generator=g)
    keep = (src_cand != tgt_cand) & ~mask[src_cand, tgt_cand]
    neg_src = src_cand[keep][:n_neg]
    neg_tgt = tgt_cand[keep][:n_neg]
    if neg_src.numel() < 10:
        # Graph is essentially complete — AUC is undefined. Return 0.5.
        return 0.5

    # Score positives and negatives
    scores = emb @ emb.T
    src, tgt = edge_index[0], edge_index[1]
    n_pos = min(len(src), neg_src.numel())
    perm = torch.randperm(len(src), generator=g)[:n_pos]
    pos_scores = scores[src[perm], tgt[perm]].detach().cpu().numpy()
    neg_scores = scores[neg_src[:n_pos], neg_tgt[:n_pos]].detach().cpu().numpy()

    all_s = np.concatenate([pos_scores, neg_scores])
    all_l = np.concatenate([np.ones(n_pos), np.zeros(n_pos)])
    # Mann-Whitney AUC: P(score(positive) > score(negative)).
    # Sort ascending; for each positive at rank i, count negatives ranked
    # below it (they have lower scores, which is correct ordering).
    order = np.argsort(all_s)
    sorted_l = all_l[order]
    n_p = sorted_l.sum()
    n_n = len(sorted_l) - n_p
    if n_p == 0 or n_n == 0:
        return 0.5
    cum_neg = np.cumsum(1 - sorted_l)
    return float(np.sum(sorted_l * cum_neg) / (n_p * n_n))


def _merge_edge_indices(snapshot: MultiplexSnapshot, device: torch.device) -> torch.LongTensor:
    """Merge edges from all available layers into one edge_index for link loss."""
    parts = []
    for ln, available in snapshot.layer_mask.items():
        if available and ln in snapshot.edge_indices and snapshot.edge_indices[ln].numel() > 0:
            parts.append(snapshot.edge_indices[ln].to(device))
    if not parts:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    return torch.cat(parts, dim=1)


# ---------------------------------------------------------------
# v3 hypothesis test: R-GCN + sparse strategic layers + InfoNCE
# ---------------------------------------------------------------

@dataclass
class V3Result:
    name: str
    layer_set: List[str]
    final_link_auc: float
    final_mean_norm: float
    n_epochs_run: int
    loss_history: List[float]
    auc_history: List[float]
    notes: str


def run_v3_diagnostic(
    dataset: MultiplexTemporalDataset,
    node_features_by_year: Dict[int, torch.Tensor],
    raw_feat_dim: int,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    emb_dim: int = 32,
    identity_dim: int = 16,
    num_neg_per_pos: int = 10,
    temperature: float = 0.5,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> V3Result:
    """Train R-GCN + InfoNCE on the sparse strategic-alignment dataset,
    evaluate with stratified-negatives AUC."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    enc_cfg = HeterogeneousEncoderConfig(
        relation_names=list(dataset.layer_names),
        raw_feat_dim=raw_feat_dim,
        identity_dim=identity_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        dropout=0.2,
    )
    encoder = HeterogeneousEncoder(dataset.num_nodes, enc_cfg).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_years = list(dataset.years)
    print(f"[v3] training on {len(train_years)} years, "
          f"{dataset.num_nodes} nodes, layers={dataset.layer_names}")

    loss_history: List[float] = []
    auc_history: List[float] = []

    for epoch in range(num_epochs):
        encoder.train()
        epoch_loss = 0.0
        n_batches = 0
        for year in train_years:
            snap = dataset.snapshots[year]
            nf = node_features_by_year[year].to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            merged = _merge_edge_indices(snap, device)
            if merged.numel() == 0:
                continue
            emb = encoder(nf, ei, ew, snap.layer_mask)
            loss = infonce_loss(emb, merged, num_neg_per_pos, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        loss_history.append(avg)

        # Evaluate stratified-negatives AUC on the most recent year
        encoder.eval()
        with torch.no_grad():
            year = train_years[-1]
            snap = dataset.snapshots[year]
            nf = node_features_by_year[year].to(device)
            ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
            ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
            emb = encoder(nf, ei, ew, snap.layer_mask)
            merged = _merge_edge_indices(snap, device)
            auc = stratified_link_auc(emb, merged, dataset.num_nodes)
            mean_norm = float(emb.norm(dim=1).mean().cpu())
        auc_history.append(auc)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{num_epochs} infonce={avg:.4f} "
                  f"strat_auc={auc:.4f} ||z||={mean_norm:.4f}")

    verdict = (
        "GOOD — sparse layers + stratified AUC + R-GCN learns structure."
        if auc_history[-1] > 0.7
        else "WEAK — improvement but not decisive."
        if auc_history[-1] > 0.55
        else "FAIL — even with all fixes, link prediction at chance."
    )
    return V3Result(
        name="v3_rgcn_sparse_layers_strat_auc",
        layer_set=list(dataset.layer_names),
  