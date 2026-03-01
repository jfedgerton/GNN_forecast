import numpy as np
import pandas as pd
import torch

from gnn_forecast.interventions import simulate_edge_toggle, simulate_random_multi_edge_deletion
from gnn_forecast.network_construction import capital_distance_matrix, residualize_ties_against_distance


def test_residualization_shapes():
    nodes = pd.DataFrame(
        {
            "ccode": [1, 2, 3],
            "cap_lat": [0.0, 0.0, 1.0],
            "cap_lon": [0.0, 1.0, 0.0],
        }
    )
    dist = capital_distance_matrix(nodes)
    edges = pd.DataFrame(
        {
            "source_ccode": [1, 1, 2],
            "target_ccode": [2, 3, 3],
            "tie": [1.0, 2.0, 1.5],
        }
    )
    out = residualize_ties_against_distance(edges, dist)
    assert {"pred_tie", "tie_resid"}.issubset(out.columns)
    assert len(out) == len(edges)


def test_intervention_outputs_nonempty():
    adj = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    emb = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    node_to_idx = {2: 0, 710: 1, 840: 2}
    res = simulate_edge_toggle(adj, emb, node_to_idx, focal_ccode=2, partner_ccodes=[710, 840])
    assert len(res) >= 1


def test_random_multi_edge_deletion():
    """Test random multi-edge deletion with a 5-node fully-connected graph."""
    n = 5
    adj = np.ones((n, n), dtype=float)
    np.fill_diagonal(adj, 0.0)
    emb = torch.randn(n, 4)
    idx_to_ccode = {0: 2, 1: 710, 2: 840, 3: 100, 4: 200}

    results = simulate_random_multi_edge_deletion(
        adj=adj,
        emb=emb,
        focal_idx=0,
        focal_ccode=2,
        idx_to_ccode=idx_to_ccode,
        k_values=[2, 3],
        n_sims=10,
        rng_seed=42,
    )

    # 2 k-values * 10 sims = 20 results
    assert len(results) == 20
    # All results should have focal_ccode = 2
    assert all(r.focal_ccode == 2 for r in results)
    # k=2 results should have 2 deleted partners each
    k2_results = [r for r in results if r.k == 2]
    assert len(k2_results) == 10
    assert all(len(r.deleted_partner_ccodes) == 2 for r in k2_results)
    # k=3 results should have 3 deleted partners each
    k3_results = [r for r in results if r.k == 3]
    assert len(k3_results) == 10
    assert all(len(r.deleted_partner_ccodes) == 3 for r in k3_results)
    # Delta should equal after - baseline
    for r in results:
        assert abs(r.embeddedness_delta - (r.embeddedness_after - r.embeddedness_baseline)) < 1e-6


def test_random_deletion_skips_when_degree_too_low():
    """If k > degree, that k is skipped gracefully."""
    adj = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    emb = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    idx_to_ccode = {0: 2, 1: 710, 2: 840}

    # Focal node 0 has degree 1, so k=2..10 should all be skipped
    results = simulate_random_multi_edge_deletion(
        adj=adj,
        emb=emb,
        focal_idx=0,
        focal_ccode=2,
        idx_to_ccode=idx_to_ccode,
        k_values=range(2, 11),
        n_sims=10,
        rng_seed=42,
    )
    assert len(results) == 0
