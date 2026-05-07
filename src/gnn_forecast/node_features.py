"""Country-year node feature loader.

Loads a CSV of country-year attributes (GDP, polity, CINC, population, etc.)
and turns them into per-year [num_nodes, num_features] tensors aligned to
the canonical ccode_to_idx mapping.

Expected CSV format:

    ccode,year,gdp,gdp_pc,population,polity2,cinc,milex,milper,...
    2,1945,...
    2,1946,...
    200,1945,...
    ...

Missing values are mean-imputed per feature column. All features are
standardized (z-score) across all (ccode, year) cells before being
returned. The mean and std used for standardization are stored on the
returned object so downstream code (e.g., per-year extraction at test
time) uses the same normalization as training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# Canonical feature names we expect from the R export. Extend as needed.
# Listed in ranked order of "must-have" → "nice-to-have" so a partial CSV
# still works (missing columns are silently dropped from the feature set).
DEFAULT_FEATURES = [
    "lp_gdp",         # log GDP
    "lp_gdppc",       # log GDP per capita
    "lp_pop",         # log population
    "polity2",        # Polity score [-10, 10]
    "cinc",           # CINC composite index
    "milex_log",      # log military expenditures
    "milper_log",     # log military personnel
    "energy_log",     # log primary energy consumption
    "lat",            # capital latitude
    "lon",            # capital longitude
]


@dataclass
class NodeFeatureSet:
    """Per-year node feature tensors aligned to a global node index."""
    feature_names: List[str]
    num_features: int
    # year -> [num_nodes, num_features] tensor (z-scored, NaN-imputed to 0 after z-score)
    by_year: Dict[int, torch.Tensor]
    # standardization stats (for any out-of-sample year you encounter later)
    feature_means: np.ndarray   # [num_features]
    feature_stds: np.ndarray    # [num_features]
    # raw attribute table that was loaded (post-merge)
    raw_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_node_features(
    path: Path,
    ccode_to_idx: Dict[int, int],
    years: List[int],
    feature_columns: Optional[List[str]] = None,
    fill_with_country_mean: bool = True,
) -> NodeFeatureSet:
    """Load country-year covariate CSV, build per-year feature tensors.

    Parameters
    ----------
    path : Path
        CSV with columns ccode, year, and one or more attribute columns.
    ccode_to_idx : dict
        Global node index from the multiplex dataset.
    years : list of int
        Years for which to build feature tensors. Years missing from the
        CSV will be filled with the closest-available-year values (forward
        fill within each ccode), then NaN-imputed with the global mean.
    feature_columns : list of str, optional
        Which feature columns to keep. Defaults to whichever of
        DEFAULT_FEATURES are present.
    fill_with_country_mean : bool
        If True (default), missing (ccode, year) cells are first imputed
        with that country's own mean across all available years before
        falling back to the global mean.
    """
    df = pd.read_csv(path)
    if "ccode" not in df.columns or "year" not in df.columns:
        raise ValueError(f"{path} must have 'ccode' and 'year' columns")
    df["ccode"] = df["ccode"].astype(int)
    df["year"] = df["year"].astype(int)

    # Decide which columns are feature columns
    if feature_columns is None:
        feature_columns = [c for c in DEFAULT_FEATURES if c in df.columns]
        if not feature_columns:
            # fall back to "everything except ccode and year"
            feature_columns = [c for c in df.columns if c not in ("ccode", "year")]
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError(f"No feature columns found in {path}")

    print(f"  Loading node features: {len(feature_columns)} columns × "
          f"{df['ccode'].nunique()} countries × "
          f"{df['year'].nunique()} years from {path.name}")
    print(f"  Features: {feature_columns}")

    df = df[["ccode", "year"] + feature_columns].copy()

    # Per-country forward-fill across years before global imputation
    if fill_with_country_mean:
        df = df.sort_values(["ccode", "year"])
        df[feature_columns] = (
            df.groupby("ccode")[feature_columns]
            .transform(lambda g: g.fillna(g.mean()))
        )

    # Compute global standardization stats (ignoring NaN) on the FULL
    # set of (ccode, year) rows, so that downstream years use the same
    # normalization.
    feat_array = df[feature_columns].to_numpy(dtype=np.float64)
    feat_means = np.nanmean(feat_array, axis=0)
    feat_stds = np.nanstd(feat_array, axis=0)
    feat_stds[feat_stds < 1e-8] = 1.0  # avoid divide-by-zero on constant features

    # Build a quick lookup: (ccode, year) -> standardized feature row
    df_z = df.copy()
    z_array = (feat_array - feat_means) / feat_stds
    z_array = np.nan_to_num(z_array, nan=0.0)  # any remaining NaN → 0 (the global mean post-z-score)
    for i, col in enumerate(feature_columns):
        df_z[col] = z_array[:, i]
    lookup: Dict[Tuple[int, int], np.ndarray] = {
        (int(r["ccode"]), int(r["year"])): r[feature_columns].to_numpy(dtype=np.float32)
        for _, r in df_z.iterrows()
    }
    # Also build a per-country fallback (mean across years) for years missing entirely
    per_country_mean: Dict[int, np.ndarray] = {}
    for cc in df_z["ccode"].unique():
        mat = df_z[df_z["ccode"] == cc][feature_columns].to_numpy(dtype=np.float32)
        per_country_mean[int(cc)] = np.nanmean(mat, axis=0) if mat.size else np.zeros(len(feature_columns))

    num_nodes = len(ccode_to_idx)
    num_features = len(feature_columns)
    by_year: Dict[int, torch.Tensor] = {}

    for year in years:
        mat = np.zeros((num_nodes, num_features), dtype=np.float32)
        for cc, idx in ccode_to_idx.items():
            row = lookup.get((cc, year))
            if row is None:
                row = per_country_mean.get(cc)
            if row is None:
                row = np.zeros(num_features, dtype=np.float32)
            mat[idx, :] = row
        by_year[year] = torch.from_numpy(mat)

    return NodeFeatureSet(
        feature_names=feature_columns,
        num_features=num_features,
        by_year=by_year,
        feature_means=feat_means,
        feature_stds=feat_stds,
        raw_df=df,
    )


def concat_with_degree_features(
    rich_features: torch.Tensor,
    degree_features: torch.Tensor,
) -> torch.Tensor:
    """Stack the rich attribute features alongside the degree-profile features.

    Returns shape [num_nodes, num_rich + num_layers].
    """
    return torch.cat([rich_features, degree_features], dim=1)
