"""Synthetic multiplex SBM with planted wedge edges, used to validate that
the counterfactual procedure recovers known interventions.

Workflow:
1. Generate a 3-block, 3-layer multiplex SBM over T years with two designated
   "focal" nodes (analogues of USA and China) in different blocks.
2. Plant a known wedge edge: a partner node whose ties to one focal differ
   sharply in effect on system-level embedding compared to the other focal.
3. Train the multiplex GNN on the synthetic data.
4. Run the dual-focal counterfactual sweep.
5. Check whether the planted wedge edge appears in the top-K isolation /
   wedge rankings.

Recovery rate (planted edge in top-K) is the validation metric.

All randomness is seeded with seed 123.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .multiplex_data import (
    LAYER_NAMES,
    MultiplexSnapshot,
    MultiplexTemporalDataset,
    compute_degree_features,
)
from .multiplex_model import MultiplexGNNConfig, MultiplexTemporalGNN
from .training import TrainingConfig, train_model
from .counterfactual import USA_CCODE, CHN_CCODE
from .isolation_analysis import dual_focal_simulation, dual_results_to_dataframe

SEED = 123


@dataclass
class SyntheticMultiplex:
    """A synthetic multiplex network for validation experiments."""
    dataset: MultiplexTemporalDataset
    block_assignment: Dict[int, int]   # ccode -> block id
    usa_ccode: int                     # ccode of focal-1 (mapped to USA_CCODE = 2)
    chn_ccode: int                     # ccode of focal-2 (mapped to CHN_CCODE = 710)
    planted_partner: int               # ccode of the partner with planted wedge
    planted_layer: str                 # layer in which the wedge was planted
    planted_operation: str             # "remove" or "add"


def _block_edge_prob(block_a: int, block_b: int, intra: float, inter: float) -> float:
    """Edge probability between two blocks under SBM."""
    return intra if block_a == block_b else inter


def generate_synthetic_multiplex(
    num_nodes: int = 60,
    num_years: int = 30,
    intra_block: float = 0.30,
    inter_block: float = 0.05,
    layers: Optional[List[str]] = None,
    seed: int = SEED,
) -> SyntheticMultiplex:
    """Generate a 3-block, multi-layer multiplex with two focal nodes.

    The first node (idx=0) gets ccode = USA_CCODE = 2.
    The second node (idx=1) gets ccode = CHN_CCODE = 710.
    Remaining nodes get arbitrary ccodes 1000..1000+N-2.

    Block assignment:
      Block 0: nodes 0..(N/3-1), includes focal-1 (USA)
      Block 1: nodes N/3..(2N/3-1), includes focal-2 (China)
      Block 2: nodes 2N/3..N-1

    A partner node from block 2 is selected as the "planted wedge": its
    edges to focal-1 are dense in one layer and to focal-2 are dense in
    another, so removing those ties asymmetrically changes the focals'
    embeddedness.
    """
    if layers is None:
        # Use the canonical layer names but only the first 3 (alliances + IGO + trade analogue)
        layers = LAYER_NAMES[:3]

    rng = np.random.default_rng(seed)

    # Block assignment
    block_size = num_nodes // 3
    block_assignment: Dict[int, int] = {}
    for i in range(num_nodes):
        if i < block_size:
            block_assignment[i] = 0
        elif i < 2 * block_size:
            block_assignment[i] = 1
        else:
            block_assignment[i] = 2

    # ccode assignment
    idx_to_ccode: Dict[int, int] = {}
    ccode_to_idx: Dict[int, int] = {}
    idx_to_ccode[0] = USA_CCODE
    idx_to_ccode[1] = CHN_CCODE
    next_ccode = 1000
    for i in range(2, num_nodes):
        idx_to_ccode[i] = next_ccode
        next_ccode += 1
    for idx, cc in idx_to_ccode.items():
        ccode_to_idx[cc] = idx

    # Choose a planted-wedge partner from block 2
    block2_indices = [i for i in range(num_nodes) if block_assignment[i] == 2]
    planted_partner_idx = int(block2_indices[0])
    planted_partner_ccode = idx_to_ccode[planted_partner_idx]
    planted_layer = layers[0]  # plant in the first layer
    planted_operation = "remove"

    # Build per-year, per-layer adjacency
    snapshots: Dict[int, MultiplexSnapshot] = {}
    base_year = 2000

    for t in range(num_years):
        year = base_year + t
        edge_indices: Dict[str, torch.LongTensor] = {}
        edge_weights: Dict[str, torch.FloatTensor] = {}
        layer_mask: Dict[str, bool] = {}

        for l_idx, ln in enumerate(layers):
            # Slight per-layer probability variation
            intra = intra_block * (0.8 + 0.2 * (l_idx + 1) / len(layers))
            inter = inter_block * (0.8 + 0.2 * (l_idx + 1) / len(layers))

            edges_src: List[int] = []
            edges_tgt: List[int] = []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    p = _block_edge_prob(
                        block_assignment[i], block_assignment[j], intra, inter,
                    )
                    if rng.random() < p:
                        edges_src.append(i)
                        edges_tgt.append(j)
                        edges_src.append(j)
                        edges_tgt.append(i)

            # Plant the wedge: ensure the planted partner is densely connected
            # to the USA in the planted_layer (so removing it should isolate USA),
            # and connected to China in a different layer.
            if ln == planted_layer:
                # Connect planted_partner to focal-1 (USA, idx=0) heavily
                for _ in range(3):
                    edges_src.append(planted_partner_idx)
                    edges_tgt.append(0)
                    edges_src.append(0)
                    edges_tgt.append(planted_partner_idx)
            elif l_idx == 1:
                # Connect planted_partner to focal-2 (China, idx=1) in layer 1
                for _ in range(3):
                    edges_src.append(planted_partner_idx)
                    edges_tgt.append(1)
                    edges_src.append(1)
                    edges_tgt.append(planted_partner_idx)

            if edges_src:
                ei = torch.tensor(np.array([edges_src, edges_tgt]), dtype=torch.long)
                ew = torch.ones(len(edges_src), dtype=torch.float32)
            else:
                ei = torch.zeros(2, 0, dtype=torch.long)
                ew = torch.zeros(0, dtype=torch.float32)

            edge_indices[ln] = ei
            edge_weights[ln] = ew
            layer_mask[ln] = True

        node_features = compute_degree_features(
            edge_indices, edge_weights, layer_mask, num_nodes, layers,
        )

        snapshots[year] = MultiplexSnapshot(
            year=year,
            num_nodes=num_nodes,
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            layer_mask=layer_mask,
            node_features=node_features,
        )

    nodes_df = pd.DataFrame([
        {"ccode": idx_to_ccode[i], "idx": i, "block": block_assignment[i]}
        for i in range(num_nodes)
    ])

    dataset = MultiplexTemporalDataset(
        years=sorted(snapshots.keys()),
        num_nodes=num_nodes,
        layer_names=layers,
        ccode_to_idx=ccode_to_idx,
        idx_to_ccode=idx_to_ccode,
        snapshots=snapshots,
        nodes_df=nodes_df,
    )

    return SyntheticMultiplex(
        dataset=dataset,
        block_assignment=block_assignment,
        usa_ccode=USA_CCODE,
        chn_ccode=CHN_CCODE,
        planted_partner=planted_partner_ccode,
        planted_layer=planted_layer,
        planted_operation=planted_operation,
    )


def run_recovery_study(
    n_replicates: int = 5,
    base_seed: int = SEED,
    top_k: int = 10,
    num_nodes: int = 60,
    num_years: int = 30,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Run the full planted-wedge recovery study.

    For each replicate:
      1. Generate synthetic multiplex with seed = base_seed + r
      2. Train the multiplex GNN
      3. Run dual-focal counterfactual sweep
      4. Check whether the planted partner appears in the top-K rankings
         for: usa_isolating, china_isolating, and abs_wedge

    Returns a DataFrame: one row per replicate with recovery indicators
    and rank of the planted edge.
    """
    rows = []

    for r in range(n_replicates):
        seed = base_seed + r
        print(f"\n=== Recovery replicate {r+1}/{n_replicates} (seed={seed}) ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        synth = generate_synthetic_multiplex(
            num_nodes=num_nodes, num_years=num_years, seed=seed,
        )

        train_cfg = TrainingConfig(
            num_epochs=50, patience=10, print_every=25, seq_len=5,
        )
        model_cfg = MultiplexGNNConfig(
            num_layers=len(synth.dataset.layer_names),
            in_dim=len(synth.dataset.layer_names),
            hidden_dim=32,
            emb_dim=16,
            seq_len=5,
        )
        result = train_model(synth.dataset, model_cfg, train_cfg)

        # Dual-focal sweep over a manageable subset of partners
        partner_pool = [
            cc for cc in synth.dataset.ccode_to_idx.keys()
            if cc not in (synth.usa_ccode, synth.chn_ccode)
        ]

        dual_results = dual_focal_simulation(
            result.model,
            synth.dataset,
            partner_ccodes=partner_pool,
            seq_len=5,
        )
        df = dual_results_to_dataframe(dual_results)

        # Check rankings
        # USA isolating = most negative usa_delta_cosine
        usa_iso = df.sort_values("usa_delta_cosine").reset_index(drop=True)
        # China isolating = most negative chn_delta_cosine
        chn_iso = df.sort_values("chn_delta_cosine").reset_index(drop=True)
        # Wedge magnitude
        df_w = df.copy()
        df_w["abs_wedge"] = df_w["wedge_cosine"].abs()
        wedge = df_w.sort_values("abs_wedge", ascending=False).reset_index(drop=True)

        def find_rank(d: pd.DataFrame) -> int:
            mask = (
                (d["partner_ccode"] == synth.planted_partner)
                & (d["layer"] == synth.planted_layer)
                & (d["operation"] == synth.planted_operation)
            )
            if not mask.any():
                return -1
            return int(d.index[mask][0]) + 1  # 1-indexed rank

        usa_rank = find_rank(usa_iso)
        chn_rank = find_rank(chn_iso)
        wedge_rank = find_rank(wedge)

        rows.append({
            "replicate": r,
            "seed": seed,
            "planted_partner_ccode": synth.planted_partner,
            "planted_layer": synth.planted_layer,
            "planted_operation": synth.planted_operation,
            "usa_isolation_rank": usa_rank,
            "chn_isolation_rank": chn_rank,
            "wedge_rank": wedge_rank,
            f"recovered_in_top{top_k}_usa": usa_rank > 0 and usa_rank <= top_k,
            f"recovered_in_top{top_k}_chn": chn_rank > 0 and chn_rank <= top_k,
            f"recovered_in_top{top_k}_wedge": wedge_rank > 0 and wedge_rank <= top_k,
            "n_perturbations_total": len(df),
        })

    df_out = pd.DataFrame(rows)

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(save_path / "recovery_study.csv", index=False)

        # Summary
        summary = {
            f"top{top_k}_usa_recovery_rate": df_out[f"recovered_in_top{top_k}_usa"].mean(),
            f"top{top_k}_chn_recovery_rate": df_out[f"recovered_in_top{top_k}_chn"].mean(),
            f"top{top_k}_wedge_recovery_rate": df_out[f"recovered_in_top{top_k}_wedge"].mean(),
            "median_usa_rank": df_out["usa_isolation_rank"].median(),
            "median_chn_rank": df_out["chn_isolation_rank"].median(),
            "median_wedge_rank": df_out["wedge_rank"].median(),
            "n_replicates": n_replicates,
        }
        pd.DataFrame([summary]).to_csv(save_path / "recovery_summary.csv", index=False)
        print(f"Saved recovery study to {save_path}")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    return df_out
