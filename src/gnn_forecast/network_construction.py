from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def capital_distance_matrix(nodes: pd.DataFrame) -> pd.DataFrame:
    s = nodes.set_index("ccode")
    c = s.index.to_numpy()
    rows = []
    for i in c:
        for j in c:
            if i == j:
                continue
            d = haversine_km(s.loc[i, "cap_lat"], s.loc[i, "cap_lon"], s.loc[j, "cap_lat"], s.loc[j, "cap_lon"])
            rows.append((i, j, d))
    return pd.DataFrame(rows, columns=["source_ccode", "target_ccode", "capital_distance_km"])


def residualize_ties_against_distance(edges: pd.DataFrame, distances: pd.DataFrame) -> pd.DataFrame:
    merged = edges.merge(distances, on=["source_ccode", "target_ccode"], how="left")
    x = merged["capital_distance_km"].to_numpy()
    y = merged["tie"].to_numpy()
    x_aug = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    merged["pred_tie"] = x_aug @ beta
    merged["tie_resid"] = merged["tie"] - merged["pred_tie"]
    return merged


@dataclass
class YearLayerBundle:
    observed_layers: Dict[str, pd.DataFrame]
    residual_layers: Dict[str, pd.DataFrame]


def make_observed_and_residual_layers(nodes: pd.DataFrame, layers: Dict[str, pd.DataFrame]) -> YearLayerBundle:
    distances = capital_distance_matrix(nodes)
    residual = {}
    for layer_name, edges in layers.items():
        residual[layer_name] = residualize_ties_against_distance(edges, distances)
    return YearLayerBundle(observed_layers=layers, residual_layers=residual)
