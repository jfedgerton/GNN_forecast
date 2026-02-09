import numpy as np
import pandas as pd
import torch

from gnn_forecast.interventions import simulate_edge_toggle
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
