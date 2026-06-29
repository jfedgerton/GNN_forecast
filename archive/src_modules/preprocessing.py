"""Tie-strength residualization: observed tie net of baseline opportunity structure.

For each layer, estimate expected tie intensity conditional on contiguity,
sender effects, receiver effects, and year effects. Tie strength is the
residual: s_{ij,t} = y_{ij,t} - yhat_{ij,t}.

Binary layers use logistic regression; continuous layers use ridge regression.
Both use categorical sender/receiver/year features as fixed-effect proxies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ResidualizedLayer:
    """Holds both observed and residualized edge data for one layer."""
    layer_name: str
    edges: pd.DataFrame  # columns: year, source_ccode, target_ccode, tie, tie_hat, tie_resid
    is_binary: bool
    model_info: Dict  # coefficients or diagnostics for inspection


def _detect_binary(series: pd.Series, threshold: float = 0.05) -> bool:
    """Heuristic: a layer is binary if >95% of values are 0 or 1."""
    vals = series.dropna()
    if len(vals) == 0:
        return True
    frac_binary = ((vals == 0) | (vals == 1)).mean()
    return bool(frac_binary > (1.0 - threshold))


def _build_fe_features(df: pd.DataFrame, contiguity_map: Optional[Dict] = None) -> np.ndarray:
    """Build fixed-effect design matrix from sender, receiver, year categories.

    Uses one-hot encoding for sender, receiver, year as proxies for FE.
    If contiguity_map is provided, adds a contiguity indicator column.
    Returns a dense numpy array (for sklearn compatibility).
    """
    # Use pandas categoricals for efficiency
    source_dummies = pd.get_dummies(df["source_ccode"].astype(str), prefix="s", dtype=float)
    target_dummies = pd.get_dummies(df["target_ccode"].astype(str), prefix="t", dtype=float)
    year_dummies = pd.get_dummies(df["year"].astype(str), prefix="y", dtype=float)

    parts = [source_dummies, target_dummies, year_dummies]

    if contiguity_map is not None:
        contig = df.apply(
            lambda r: contiguity_map.get((int(r["source_ccode"]), int(r["target_ccode"])), 0.0),
            axis=1,
        )
        parts.append(pd.DataFrame({"contiguity": contig}))

    X = pd.concat(parts, axis=1).values.astype(np.float64)
    return X


def residualize_layer(
    edges: pd.DataFrame,
    layer_name: str,
    is_binary: Optional[bool] = None,
    contiguity_map: Optional[Dict] = None,
    regularization: float = 1.0,
) -> ResidualizedLayer:
    """Residualize one layer's ties against sender + receiver + year FE + contiguity.

    Parameters
    ----------
    edges : DataFrame
        Must have columns: year, source_ccode, target_ccode, tie
    layer_name : str
        Name of the layer (for metadata).
    is_binary : bool or None
        If None, auto-detect from data.
    contiguity_map : dict or None
        Mapping (source_ccode, target_ccode) -> contiguity indicator (0/1 or float).
    regularization : float
        Regularization strength (alpha for Ridge, C=1/alpha for Logistic).

    Returns
    -------
    ResidualizedLayer with edges augmented by tie_hat and tie_resid columns.
    """
    df = edges.copy()
    df = df.dropna(subset=["tie"])

    if len(df) == 0:
        df["tie_hat"] = np.nan
        df["tie_resid"] = np.nan
        return ResidualizedLayer(
            layer_name=layer_name,
            edges=df,
            is_binary=True,
            model_info={"n_obs": 0, "method": "empty"},
        )

    if is_binary is None:
        is_binary = _detect_binary(df["tie"])

    y = df["tie"].values.astype(np.float64)
    X = _build_fe_features(df, contiguity_map)

    model_info: Dict = {
        "n_obs": len(y),
        "n_features": X.shape[1],
        "is_binary": is_binary,
        "layer_name": layer_name,
    }

    if is_binary:
        # Logistic regression for binary outcomes
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(
                C=1.0 / max(regularization, 1e-8),
                max_iter=500,
                solver="lbfgs",
            )
            clf.fit(X, y)
            y_hat = clf.predict_proba(X)[:, 1] if len(clf.classes_) > 1 else np.full(len(y), y.mean())
            model_info["method"] = "logistic_regression"
        except (ImportError, Exception) as exc:
            # Fallback: use OLS even for binary
            y_hat = _ols_predict(X, y, regularization)
            model_info["method"] = f"ols_fallback ({exc})"
    else:
        # Ridge regression for continuous outcomes
        try:
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=regularization)
            reg.fit(X, y)
            y_hat = reg.predict(X)
            model_info["method"] = "ridge_regression"
        except (ImportError, Exception):
            y_hat = _ols_predict(X, y, regularization)
            model_info["method"] = "ols_fallback"

    df["tie_hat"] = y_hat
    df["tie_resid"] = y - y_hat
    model_info["resid_mean"] = float(np.mean(y - y_hat))
    model_info["resid_std"] = float(np.std(y - y_hat))

    return ResidualizedLayer(
        layer_name=layer_name,
        edges=df,
        is_binary=is_binary,
        model_info=model_info,
    )


def _ols_predict(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Simple ridge-regularized OLS fallback."""
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    return X @ beta


def residualize_all_layers(
    layer_edges: Dict[str, pd.DataFrame],
    contiguity_map: Optional[Dict] = None,
    binary_layers: Optional[Dict[str, bool]] = None,
    regularization: float = 1.0,
) -> Dict[str, ResidualizedLayer]:
    """Residualize all layers and return a dict of ResidualizedLayer objects.

    Parameters
    ----------
    layer_edges : dict
        layer_name -> DataFrame with columns [year, source_ccode, target_ccode, tie]
    contiguity_map : dict or None
        (source_ccode, target_ccode) -> contiguity indicator
    binary_layers : dict or None
        layer_name -> bool indicating if the layer is binary. Auto-detect if None.
    regularization : float
        Regularization strength.

    Returns
    -------
    dict of layer_name -> ResidualizedLayer
    """
    results = {}
    for layer_name, edges in layer_edges.items():
        is_binary = binary_layers.get(layer_name) if binary_layers else None
        results[layer_name] = residualize_layer(
            edges=edges,
            layer_name=layer_name,
            is_binary=is_binary,
            contiguity_map=contiguity_map,
            regularization=regularization,
        )
        print(f"  Residualized '{layer_name}': {results[layer_name].model_info}")
    return results
