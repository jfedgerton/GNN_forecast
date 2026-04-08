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
    return 2 * r * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def capital_distance_matrix(nodes: pd.DataFrame) -> pd.DataFrame:
    required = {"ccode", "cap_lat", "cap_lon"}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"Nodes DataFrame missing required columns: {missing}")
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
    merged = merged.dropna(subset=["capital_distance_km"])
    if len(merged) == 0:
        raise ValueError("No edges remain after merging with distances — check that edge ccodes match distance ccodes")
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
