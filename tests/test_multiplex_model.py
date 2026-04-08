"""Tests for the multiplex temporal GNN model and pipeline components."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from gnn_forecast.multiplex_data import (
    build_global_node_index,
    build_multiplex_dataset,
    compute_degree_features,
)
from gnn_forecast.multiplex_model import (
    LayerAggregator,
    LayerEncoder,
    MultiplexGNNConfig,
    MultiplexTemporalGNN,
    SimpleGCNLayerSparse,
    TemporalGRU,
)
from gnn_forecast.counterfactual import compute_embeddedness_metrics
from gnn_forecast.preprocessing import residualize_layer, _detect_binary


# ---------------------------------------------------------------
# Fixtures: small synthetic multiplex data
# ---------------------------------------------------------------
def _make_synthetic_layers():
    """Create 3 small synthetic layers spanning 3 years."""
    layers = {}

    # Layer 1: binary alliance (3 years)
    layers["offensive_alliances"] = pd.DataFrame({
        "year": [2000, 2000, 2001, 2001, 2002, 2002],
        "source_ccode": [2, 710, 2, 200, 2, 710],
        "target_ccode": [710, 200, 200, 365, 365, 365],
        "tie": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })

    # Layer 2: continuous trade (2 years -- missing 2002)
    layers["trade"] = pd.DataFrame({
        "year": [2000, 2000, 2001, 2001],
        "source_ccode": [2, 710, 2, 710],
        "target_ccode": [710, 200, 200, 365],
        "tie": [100.5, 50.2, 120.3, 60.1],
    })

    # Layer 3: IGO count (3 years)
    layers["igo"] = pd.DataFrame({
        "year": [2000, 2000, 2001, 2002, 2002],
        "source_ccode": [2, 200, 2, 2, 710],
        "target_ccode": [200, 365, 365, 710, 365],
        "tie": [5.0, 3.0, 6.0, 7.0, 4.0],
    })

    return layers


def _make_synthetic_dataset():
    """Build a MultiplexTemporalDataset from synthetic data."""
    layers = _make_synthetic_layers()
    ccode_to_idx, idx_to_ccode, nodes_df = build_global_node_index(layers)
    dataset = build_multiplex_dataset(
        layer_dfs=layers,
        ccode_to_idx=ccode_to_idx,
        idx_to_ccode=idx_to_ccode,
        nodes_df=nodes_df,
        year_range=(2000, 2002),
    )
    return dataset


# ---------------------------------------------------------------
# Tests: data loading and tensor construction
# ---------------------------------------------------------------
def test_global_node_index():
    layers = _make_synthetic_layers()
    ccode_to_idx, idx_to_ccode, _ = build_global_node_index(layers)
    # Should have 4 unique ccodes: 2, 200, 365, 710
    assert len(ccode_to_idx) == 4
    assert set(ccode_to_idx.keys()) == {2, 200, 365, 710}
    # Sorted order
    assert ccode_to_idx[2] == 0
    assert ccode_to_idx[200] == 1
    assert ccode_to_idx[365] == 2
    assert ccode_to_idx[710] == 3


def test_multiplex_dataset_structure():
    dataset = _make_synthetic_dataset()
    assert dataset.num_nodes == 4
    assert dataset.years == [2000, 2001, 2002]
    assert len(dataset.snapshots) == 3


def test_layer_mask_missing_trade_2002():
    """Trade is missing in 2002; check that the mask reflects this."""
    dataset = _make_synthetic_dataset()
    snap_2002 = dataset.snapshots[2002]
    assert snap_2002.layer_mask["offensive_alliances"] is True
    assert snap_2002.layer_mask["igo"] is True
    assert snap_2002.layer_mask["trade"] is False


def test_edge_indices_shape():
    dataset = _make_synthetic_dataset()
    snap = dataset.snapshots[2000]
    for ln in dataset.layer_names:
        ei = snap.edge_indices[ln]
        assert ei.dim() == 2
        assert ei.size(0) == 2


def test_node_features_shape():
    dataset = _make_synthetic_dataset()
    snap = dataset.snapshots[2000]
    assert snap.node_features.shape == (4, len(dataset.layer_names))
    # Features should be in [0, 1] range
    assert snap.node_features.min() >= 0.0
    assert snap.node_features.max() <= 1.0


# ---------------------------------------------------------------
# Tests: GCN layer
# ---------------------------------------------------------------
def test_simple_gcn_layer():
    layer = SimpleGCNLayerSparse(3, 4)
    x = torch.randn(5, 3)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    out = layer(x, ei, num_nodes=5)
    assert out.shape == (5, 4)


def test_simple_gcn_empty_graph():
    layer = SimpleGCNLayerSparse(3, 4)
    x = torch.randn(5, 3)
    ei = torch.zeros(2, 0, dtype=torch.long)
    out = layer(x, ei, num_nodes=5)
    assert out.shape == (5, 4)


# ---------------------------------------------------------------
# Tests: layer encoder
# ---------------------------------------------------------------
def test_layer_encoder():
    enc = LayerEncoder(in_dim=3, hidden_dim=8, out_dim=4, use_pyg=False)
    x = torch.randn(5, 3)
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    out = enc(x, ei)
    assert out.shape == (5, 4)


# ---------------------------------------------------------------
# Tests: layer aggregation
# ---------------------------------------------------------------
def test_layer_aggregator():
    agg = LayerAggregator(num_layers=3)
    embs = [torch.randn(5, 4) for _ in range(3)]
    mask = [True, True, True]
    out = agg(embs, mask)
    assert out.shape == (5, 4)


def test_layer_aggregator_masked():
    agg = LayerAggregator(num_layers=3)
    embs = [torch.randn(5, 4) for _ in range(3)]
    mask = [True, False, True]
    out = agg(embs, mask)
    assert out.shape == (5, 4)
    # With middle layer masked, its contribution should be zero
    weights = agg.get_layer_weights(mask)
    assert weights[1].item() == 0.0  # masked layer gets zero weight


def test_layer_aggregator_all_masked():
    agg = LayerAggregator(num_layers=3)
    embs = [torch.randn(5, 4) for _ in range(3)]
    mask = [False, False, False]
    out = agg(embs, mask)
    assert out.shape == (5, 4)
    assert torch.allclose(out, torch.zeros(5, 4))


# ---------------------------------------------------------------
# Tests: temporal GRU
# ---------------------------------------------------------------
def test_temporal_gru():
    gru = TemporalGRU(emb_dim=4, hidden_dim=8)
    seq = torch.randn(5, 3, 4)  # 5 nodes, 3 time steps, 4 dims
    out = gru(seq)
    assert out.shape == (5, 4)


# ---------------------------------------------------------------
# Tests: full model
# ---------------------------------------------------------------
def test_full_model_encode_snapshot():
    config = MultiplexGNNConfig(
        num_layers=3,
        in_dim=3,
        hidden_dim=8,
        emb_dim=4,
    )
    layer_names = ["offensive_alliances", "trade", "igo"]
    model = MultiplexTemporalGNN(config, layer_names)

    x = torch.randn(5, 3)
    edge_indices = {
        "offensive_alliances": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "trade": torch.tensor([[0, 2], [2, 3]], dtype=torch.long),
        "igo": torch.tensor([[1, 3], [3, 4]], dtype=torch.long),
    }
    edge_weights = {
        "offensive_alliances": torch.ones(2),
        "trade": torch.tensor([100.0, 50.0]),
        "igo": torch.tensor([5.0, 3.0]),
    }
    layer_mask = {
        "offensive_alliances": True,
        "trade": True,
        "igo": True,
    }

    emb = model.encode_snapshot(x, edge_indices, edge_weights, layer_mask)
    assert emb.shape == (5, 4)


def test_full_model_temporal():
    config = MultiplexGNNConfig(
        num_layers=2,
        in_dim=2,
        hidden_dim=8,
        emb_dim=4,
        seq_len=3,
    )
    layer_names = ["offensive_alliances", "trade"]
    model = MultiplexTemporalGNN(config, layer_names)

    history = [torch.randn(5, 4) for _ in range(3)]
    pred = model.forward_temporal(history)
    assert pred.shape == (5, 4)


def test_layer_weights_inspection():
    config = MultiplexGNNConfig(num_layers=3, in_dim=3)
    layer_names = ["a", "b", "c"]
    model = MultiplexTemporalGNN(config, layer_names)
    weights = model.get_layer_weights()
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 1e-5


# ---------------------------------------------------------------
# Tests: embeddedness metrics
# ---------------------------------------------------------------
def test_embeddedness_metrics():
    emb = torch.randn(5, 4)
    ccode_to_idx = {2: 0, 200: 1, 365: 2, 710: 3, 220: 4}
    metrics = compute_embeddedness_metrics(emb, 0, ccode_to_idx, k_neighbors=3)
    assert metrics.ccode == 2
    assert isinstance(metrics.mean_cosine_similarity, float)
    assert isinstance(metrics.centroid_distance, float)
    assert isinstance(metrics.knn_density, float)


# ---------------------------------------------------------------
# Tests: preprocessing (residualization)
# ---------------------------------------------------------------
def test_detect_binary():
    binary_series = pd.Series([0, 1, 0, 1, 1, 0])
    assert _detect_binary(binary_series) is True

    continuous_series = pd.Series([0.5, 1.2, 3.4, 0.0, 2.1])
    assert _detect_binary(continuous_series) is False


def test_residualize_layer_binary():
    edges = pd.DataFrame({
        "year": [2000] * 6 + [2001] * 6,
        "source_ccode": [2, 2, 200, 200, 365, 365] * 2,
        "target_ccode": [200, 365, 2, 365, 2, 200] * 2,
        "tie": [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0],
    })
    result = residualize_layer(edges, "test_binary", is_binary=True)
    assert "tie_hat" in result.edges.columns
    assert "tie_resid" in result.edges.columns
    assert len(result.edges) == 12
    assert result.is_binary is True


def test_residualize_layer_continuous():
    edges = pd.DataFrame({
        "year": [2000] * 4 + [2001] * 4,
        "source_ccode": [2, 2, 200, 365] * 2,
        "target_ccode": [200, 365, 365, 2] * 2,
        "tie": [100.0, 50.0, 30.0, 80.0, 110.0, 55.0, 35.0, 85.0],
    })
    result = residualize_layer(edges, "test_continuous", is_binary=False)
    assert "tie_hat" in result.edges.columns
    assert "tie_resid" in result.edges.columns
    assert result.is_binary is False
