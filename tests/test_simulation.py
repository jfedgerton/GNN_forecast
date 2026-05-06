"""Tests for the synthetic multiplex generator and recovery study."""
from __future__ import annotations

import numpy as np
import torch

from gnn_forecast.simulation import (
    generate_synthetic_multiplex,
    SyntheticMultiplex,
)
from gnn_forecast.counterfactual import USA_CCODE, CHN_CCODE


def test_synthetic_multiplex_basic_shape() -> None:
    np.random.seed(123)
    torch.manual_seed(123)
    synth = generate_synthetic_multiplex(num_nodes=30, num_years=10, seed=123)

    assert isinstance(synth, SyntheticMultiplex)
    assert synth.dataset.num_nodes == 30
    assert len(synth.dataset.years) == 10
    # USA and China codes are in the index
    assert USA_CCODE in synth.dataset.ccode_to_idx
    assert CHN_CCODE in synth.dataset.ccode_to_idx
    # Planted partner is not USA or China
    assert synth.planted_partner not in (USA_CCODE, CHN_CCODE)
    # Planted layer is one of the dataset's layers
    assert synth.planted_layer in synth.dataset.layer_names


def test_synthetic_multiplex_has_edges() -> None:
    synth = generate_synthetic_multiplex(num_nodes=30, num_years=5, seed=123)
    # Every year has at least one edge in every layer
    for year in synth.dataset.years:
        snap = synth.dataset.snapshots[year]
        for ln in synth.dataset.layer_names:
            assert snap.layer_mask.get(ln, False)
            assert snap.edge_indices[ln].size(1) > 0


def test_planted_edge_present_in_planted_layer() -> None:
    """The planted partner should be densely connected to USA in the planted layer."""
    synth = generate_synthetic_multiplex(num_nodes=30, num_years=5, seed=123)
    usa_idx = synth.dataset.ccode_to_idx[USA_CCODE]
    partner_idx = synth.dataset.ccode_to_idx[synth.planted_partner]

    # Check the first year's edges in the planted layer
    snap = synth.dataset.snapshots[synth.dataset.years[0]]
    ei = snap.edge_indices[synth.planted_layer]
    # Should have at least one edge between USA and planted partner
    pair_mask = (
        ((ei[0] == usa_idx) & (ei[1] == partner_idx))
        | ((ei[0] == partner_idx) & (ei[1] == usa_idx))
    )
    assert pair_mask.sum().item() > 0
