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
    planted_partner_idx: Optional[int] = None,
    planted_layer_idx: int = 0,
    usa_extra_edges: int = 12,
    china_extra_edges: int = 0,
    null_scenario: bool = False,
) -> SyntheticMultiplex:
    """Generate a 3-block, multi-layer multiplex with two focal nodes.

    Node 0 = USA (ccode 2), node 1 = China (ccode 710), the rest get
    ccodes 1000..1000+N-2. Three SBM blocks of equal size; USA in block
    0, China in block 1, planted partner drawn from block 2.

    Planted-wedge structure (default):
      - planted_partner gets `usa_extra_edges` extra ties to USA in
        layer `planted_layer_idx` (default 12 — strong asymmetric signal).
      - planted_partner gets `china_extra_edges` extra ties to China in
        the next layer (default 0 — no China tie, so removal isolates
        USA cleanly).

    Set `null_scenario=True` to skip planting entirely — used as the
    calibration condition for recovery-rate baselines.

    Set `planted_partner_idx` to override the default (block_2[0]) and
    rotate the planted partner across replicates.
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
    if planted_partner_idx is None:
        planted_partner_idx = int(block2_indices[0])
    planted_partner_ccode = idx_to_ccode[planted_partner_idx]
    planted_layer = layers[planted_layer_idx]
    china_layer_idx = (planted_layer_idx + 1) % len(layers)
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

            # Plant the wedge unless null scenario is requested.
            # Default config: strong USA tie in planted_layer (12 extras),
            # zero China tie. Removing partner from planted_layer should
            # isolate USA cleanly while leaving China untouched.
            if not null_scenario:
                if l_idx == planted_layer_idx and usa_extra_edges > 0:
                    for _ in range(usa_extra_edges):
                        edges_src.append(planted_partner_idx)
                        edges_tgt.append(0)
                        edges_src.append(0)
                        edges_tgt.append(planted_partner_idx)
                elif l_idx == china_layer_idx and china_extra_edges > 0:
                    for _ in range(china_extra_edges):
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


def _run_one_replicate(
    seed: int,
    num_nodes: int,
    num_years: int,
    planted_partner_idx: Optional[int],
    usa_extra_edges: int,
    null_scenario: bool,
    top_k: int,
) -> Dict:
    """One replicate: generate synth data, train GNN, run sweep, score recovery."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    synth = generate_synthetic_multiplex(
        num_nodes=num_nodes,
        num_years=num_years,
        seed=seed,
        planted_partner_idx=planted_partner_idx,
        usa_extra_edges=usa_extra_edges,
        china_extra_edges=0,
        null_scenario=null_scenario,
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

    usa_iso = df.sort_values("usa_delta_cosine").reset_index(drop=True)
    chn_iso = df.sort_values("chn_delta_cosine").reset_index(drop=True)
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
        return int(d.index[mask][0]) + 1

    usa_rank = find_rank(usa_iso)
    chn_rank = find_rank(chn_iso)
    wedge_rank = find_rank(wedge)

    return {
        "seed": seed,
        "scenario": "null" if null_scenario else "planted",
        "usa_extra_edges": 0 if null_scenario else usa_extra_edges,
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
    }


def run_recovery_study(
    n_replicates: int = 5,
    base_seed: int = SEED,
    top_k: int = 10,
    num_nodes: int = 60,
    num_years: int = 30,
    usa_extra_edges: int = 12,
    rotate_partner: bool = True,
    include_null: bool = True,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Planted-wedge recovery study with two improvements over the v1:

      1. Strong asymmetric planting (`usa_extra_edges`, default 12) and
         no China side-tie, so removing the partner should isolate USA
         cleanly.
      2. Partner rotation (`rotate_partner=True`): replicate r plants the
         r-th node from block 2, giving an honest recovery rate over
         partners rather than seed noise on one partner.
      3. Optional null condition (`include_null=True`): for each planted
         replicate, also runs an unplanted SBM with the same seed. The
         null recovery rate is the chance baseline (~k/N).

    Returns one row per (replicate, scenario in {planted, null}).
    Total replicates run: n_replicates × (2 if include_null else 1).
    """
    rows = []

    # Determine block 2 partner pool deterministically from generator config
    block_size = num_nodes // 3
    block2_indices = list(range(2 * block_size, num_nodes))

    for r in range(n_replicates):
        seed = base_seed + r
        partner_idx = block2_indices[r % len(block2_indices)] if rotate_partner else None

        print(f"\n=== Replicate {r+1}/{n_replicates} (seed={seed}, partner_idx={partner_idx}) ===")

        # Planted scenario
        print("  -- planted scenario --")
        row_planted = _run_one_replicate(
            seed=seed, num_nodes=num_nodes, num_years=num_years,
            planted_partner_idx=partner_idx,
            usa_extra_edges=usa_extra_edges,
            null_scenario=False, top_k=top_k,
        )
        row_planted["replicate"] = r
        rows.append(row_planted)

        # Null scenario (same seed, no planting)
        if include_null:
            print("  -- null scenario --")
            row_null = _run_one_replicate(
                seed=seed, num_nodes=num_nodes, num_years=num_years,
                planted_partner_idx=partner_idx,
                usa_extra_edges=usa_extra_edges,
                null_scenario=True, top_k=top_k,
            )
            row_null["replicate"] = r
            rows.append(row_null)

    df_out = pd.DataFrame(rows)

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(save_path / "recovery_study.csv", index=False)

        # Summary broken out by scenario (planted vs. null) for the
        # 2x2 calibration table the paper needs.
        summary_rows = []
        for scenario in df_out["scenario"].unique():
            sub = df_out[df_out["scenario"] == scenario]
            summary_rows.append({
                "scenario": scenario,
                "n_replicates": len(sub),
                f"top{top_k}_usa_recovery_rate": sub[f"recovered_in_top{top_k}_usa"].mean(),
                f"top{top_k}_chn_recovery_rate": sub[f"recovered_in_top{top_k}_chn"].mean(),
                f"top{top_k}_wedge_recovery_rate": sub[f"recovered_in_top{top_k}_wedge"].mean(),
                "median_usa_rank": sub["usa_isolation_rank"].median(),
                "median_chn_rank": sub["chn_isolation_rank"].median(),
                "median_wedge_rank": sub["wedge_rank"].median(),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(save_path / "recovery_summary.csv", index=False)
        # Also export the legacy single-row summary for the planted scenario only
        # so downstream consumers that read recovery_summary.csv as one row keep working.
        if "planted" in df_out["scenario"].values:
            planted = df_out[df_out["scenario"] == "planted"]
            legacy = {
                f"top{top_k}_usa_recovery_rate": planted[f"recovered_in_top{top_k}_usa"].mean(),
                f"top{top_k}_chn_recovery_rate": planted[f"recovered_in_top{top_k}_chn"].mean(),
                f"top{top_k}_wedge_recovery_rate": planted[f"recovered_in_top{top_k}_wedge"].mean(),
                "median_usa_rank": planted["usa_isolation_rank"].median(),
                "median_chn_rank": planted["chn_isolation_rank"].median(),
                "median_wedge_rank": planted["wedge_rank"].median(),
                "n_replicates": len(planted),
            }
            pd.DataFrame([legacy]).to_csv(save_path / "recovery_summary_planted.csv", index=False)
        # Replace bare summary printout with the per-scenario breakdown
        summary = {f"_{scenario}": row for scenario, row in zip(summary_df["scenario"], summary_df.to_dict("records"))}
        print(f"Saved recovery study to {save_path}")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    return df_out
