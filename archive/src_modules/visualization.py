"""Embedding visualization and figure generation for manuscript.

Produces:
1. 2D PCA/UMAP projections of yearly embeddings with USA/China highlighted
2. Temporal trajectory plots showing embedding drift
3. Layer weight bar charts
4. Loss curves
5. Embeddedness time series for USA and China
6. Counterfactual effect size heatmaps by layer
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .counterfactual import compute_embeddedness_metrics, USA_CCODE, CHN_CCODE, MAJOR_POWER_CCODES


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Simple 2D PCA without sklearn dependency."""
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Take top 2 eigenvectors (sorted descending)
    idx = np.argsort(-eigvals)[:2]
    return X_centered @ eigvecs[:, idx]


def embedding_projections(
    yearly_embeddings: Dict[int, torch.Tensor],
    idx_to_ccode: Dict[int, int],
    years_to_plot: Optional[List[int]] = None,
    method: str = "pca",
) -> pd.DataFrame:
    """Project embeddings to 2D and return a long-form DataFrame.

    Columns: year, ccode, dim1, dim2, is_usa, is_china, is_major_power
    """
    if years_to_plot is None:
        years_to_plot = sorted(yearly_embeddings.keys())

    # Stack all year embeddings for a joint projection
    all_embs = []
    all_years = []
    all_node_ids = []
    for year in years_to_plot:
        if year not in yearly_embeddings:
            continue
        emb = yearly_embeddings[year]
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        n = emb.shape[0]
        all_embs.append(emb)
        all_years.extend([year] * n)
        all_node_ids.extend(range(n))

    if not all_embs:
        return pd.DataFrame()

    stacked = np.vstack(all_embs)

    if method == "pca":
        projected = _pca_2d(stacked)
    else:
        # Fallback to PCA if UMAP not available
        try:
            from umap import UMAP
            projected = UMAP(n_components=2, random_state=42).fit_transform(stacked)
        except ImportError:
            projected = _pca_2d(stacked)

    rows = []
    for i in range(len(all_years)):
        ccode = idx_to_ccode.get(all_node_ids[i], all_node_ids[i])
        rows.append({
            "year": all_years[i],
            "ccode": ccode,
            "node_idx": all_node_ids[i],
            "dim1": projected[i, 0],
            "dim2": projected[i, 1],
            "is_usa": ccode == USA_CCODE,
            "is_china": ccode == CHN_CCODE,
            "is_major_power": ccode in MAJOR_POWER_CCODES,
        })

    return pd.DataFrame(rows)


def embeddedness_time_series(
    yearly_embeddings: Dict[int, torch.Tensor],
    ccode_to_idx: Dict[int, int],
    focal_ccodes: Optional[List[int]] = None,
    k_neighbors: int = 10,
) -> pd.DataFrame:
    """Compute embeddedness metrics for focal states across all years.

    Returns long-form DataFrame: year, ccode, mean_cosine, centroid_dist,
    mp_dist, knn_density
    """
    if focal_ccodes is None:
        focal_ccodes = [USA_CCODE, CHN_CCODE]

    rows = []
    for year in sorted(yearly_embeddings.keys()):
        emb = yearly_embeddings[year]
        if isinstance(emb, torch.Tensor):
            emb = emb.detach()

        for ccode in focal_ccodes:
            if ccode not in ccode_to_idx:
                continue
            idx = ccode_to_idx[ccode]
            metrics = compute_embeddedness_metrics(emb, idx, ccode_to_idx, k_neighbors)
            rows.append({
                "year": year,
                "ccode": ccode,
                "country": MAJOR_POWER_CCODES.get(ccode, str(ccode)),
                "mean_cosine_similarity": metrics.mean_cosine_similarity,
                "centroid_distance": metrics.centroid_distance,
                "major_power_distance": metrics.major_power_distance,
                "knn_density": metrics.knn_density,
            })

    return pd.DataFrame(rows)


def layer_weight_table(
    raw_weights: Dict[str, float],
    weighted_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Combine layer weights from both model sets into a comparison table."""
    rows = []
    all_layers = set(raw_weights.keys())
    if weighted_weights:
        all_layers |= set(weighted_weights.keys())

    for layer in sorted(all_layers):
        row = {
            "layer": layer,
            "raw_weight": raw_weights.get(layer, np.nan),
        }
        if weighted_weights:
            row["weighted_weight"] = weighted_weights.get(layer, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def loss_curve_table(epoch_details: List[Dict]) -> pd.DataFrame:
    """Convert epoch details to a DataFrame suitable for plotting."""
    return pd.DataFrame(epoch_details)


def model_set_comparison(
    raw_embeddings: Dict[int, torch.Tensor],
    weighted_embeddings: Dict[int, torch.Tensor],
    ccode_to_idx: Dict[int, int],
) -> pd.DataFrame:
    """Compare embeddedness trajectories between Model Set 1 and Model Set 2.

    Returns a DataFrame with year, ccode, raw_cosine, weighted_cosine, delta.
    """
    focal_ccodes = [USA_CCODE, CHN_CCODE]
    common_years = sorted(set(raw_embeddings.keys()) & set(weighted_embeddings.keys()))

    rows = []
    for year in common_years:
        raw_emb = raw_embeddings[year].detach() if isinstance(raw_embeddings[year], torch.Tensor) else raw_embeddings[year]
        wt_emb = weighted_embeddings[year].detach() if isinstance(weighted_embeddings[year], torch.Tensor) else weighted_embeddings[year]

        for ccode in focal_ccodes:
            if ccode not in ccode_to_idx:
                continue
            idx = ccode_to_idx[ccode]
            raw_m = compute_embeddedness_metrics(raw_emb, idx, ccode_to_idx)
            wt_m = compute_embeddedness_metrics(wt_emb, idx, ccode_to_idx)
            rows.append({
                "year": year,
                "ccode": ccode,
                "country": MAJOR_POWER_CCODES.get(ccode, str(ccode)),
                "raw_mean_cosine": raw_m.mean_cosine_similarity,
                "weighted_mean_cosine": wt_m.mean_cosine_similarity,
                "delta_mean_cosine": wt_m.mean_cosine_similarity - raw_m.mean_cosine_similarity,
                "raw_centroid_dist": raw_m.centroid_distance,
                "weighted_centroid_dist": wt_m.centroid_distance,
                "delta_centroid_dist": wt_m.centroid_distance - raw_m.centroid_distance,
            })

    return pd.DataFrame(rows)


def generate_all_tables(
    raw_result,
    weighted_result,
    dataset,
    out_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    """Generate all manuscript-ready CSV tables from training results.

    Parameters
    ----------
    raw_result : TrainingResult from Model Set 1
    weighted_result : TrainingResult from Model Set 2 (or None)
    dataset : MultiplexTemporalDataset
    out_dir : directory to save tables

    Returns dict of table_name -> DataFrame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}

    # 1. Embedding projections (raw model)
    proj = embedding_projections(
        raw_result.yearly_embeddings,
        dataset.idx_to_ccode,
    )
    proj.to_csv(out_dir / "embedding_projections_raw.csv", index=False)
    tables["projections_raw"] = proj

    # 2. Embeddedness time series
    emb_ts = embeddedness_time_series(
        raw_result.yearly_embeddings,
        dataset.ccode_to_idx,
    )
    emb_ts.to_csv(out_dir / "embeddedness_timeseries_raw.csv", index=False)
    tables["embeddedness_raw"] = emb_ts

    # 3. Layer weights comparison
    wt_weights = weighted_result.layer_weights if weighted_result else None
    lwt = layer_weight_table(raw_result.layer_weights, wt_weights)
    lwt.to_csv(out_dir / "layer_weights.csv", index=False)
    tables["layer_weights"] = lwt

    # 4. Loss curves
    loss_raw = loss_curve_table(raw_result.epoch_details)
    loss_raw.to_csv(out_dir / "loss_curve_raw.csv", index=False)
    tables["loss_raw"] = loss_raw

    if weighted_result:
        loss_wt = loss_curve_table(weighted_result.epoch_details)
        loss_wt.to_csv(out_dir / "loss_curve_weighted.csv", index=False)
        tables["loss_weighted"] = loss_wt

    # 5. Model set comparison
    if weighted_result:
        comparison = model_set_comparison(
            raw_result.yearly_embeddings,
            weighted_result.yearly_embeddings,
            dataset.ccode_to_idx,
        )
        comparison.to_csv(out_dir / "model_set_comparison.csv", index=False)
        tables["model_set_comparison"] = comparison

        # Weighted embeddedness time series
        emb_ts_wt = embeddedness_time_series(
            weighted_result.yearly_embeddings,
            dataset.ccode_to_idx,
        )
        emb_ts_wt.to_csv(out_dir / "embeddedness_timeseries_weighted.csv", index=False)
        tables["embeddedness_weighted"] = emb_ts_wt

        # Weighted projections
        proj_wt = embedding_projections(
            weighted_result.yearly_embeddings,
            dataset.idx_to_ccode,
        )
        proj_wt.to_csv(out_dir / "embedding_projections_weighted.csv", index=False)
        tables["projections_weighted"] = proj_wt

    print(f"Generated {len(tables)} tables in {out_dir}")
    return tables
