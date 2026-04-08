"""Load and convert multiplex temporal network data into PyTorch tensors.

Supports 5 layers with different temporal coverage. Missing layers in any year
are represented via explicit boolean masks. Produces per-year Data objects
compatible with PyTorch Geometric, plus a global node index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

# Layer names used throughout the pipeline
LAYER_NAMES = [
    "offensive_alliances",
    "defensive_alliances",
    "igo",
    "trade",
    "dca",
]

# Which layers are binary (used for residualization decisions)
BINARY_LAYERS = {
    "offensive_alliances": True,
    "defensive_alliances": True,
    "igo": False,  # count of shared memberships
    "trade": False,  # continuous trade volume
    "dca": True,  # binary DCA indicator
}


@dataclass
class MultiplexSnapshot:
    """One year of the multiplex graph, converted to tensors."""
    year: int
    num_nodes: int
    # Per-layer edge indices [2, num_edges_l] and weights [num_edges_l]
    edge_indices: Dict[str, torch.LongTensor]
    edge_weights: Dict[str, torch.FloatTensor]
    # Boolean mask: which layers are available this year
    layer_mask: Dict[str, bool]
    # Node feature matrix [num_nodes, num_features]
    node_features: torch.FloatTensor


@dataclass
class MultiplexTemporalDataset:
    """Full temporal multiplex dataset ready for GNN training."""
    years: List[int]
    num_nodes: int
    layer_names: List[str]
    ccode_to_idx: Dict[int, int]
    idx_to_ccode: Dict[int, int]
    snapshots: Dict[int, MultiplexSnapshot]
    # Metadata
    nodes_df: pd.DataFrame
    contiguity_map: Optional[Dict] = None


def load_layer_csv(path: Path) -> pd.DataFrame:
    """Load a single layer CSV file. Expects columns: year, source_ccode, target_ccode, tie."""
    df = pd.read_csv(path)
    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("year",):
            col_map[c] = "year"
        elif cl in ("source_ccode", "ccode1", "src", "source"):
            col_map[c] = "source_ccode"
        elif cl in ("target_ccode", "ccode2", "dst", "target"):
            col_map[c] = "target_ccode"
        elif cl in ("tie", "value", "weight", "value_raw"):
            col_map[c] = "tie"
    df = df.rename(columns=col_map)
    required = ["year", "source_ccode", "target_ccode", "tie"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}. Found: {list(df.columns)}")
    df = df[required].copy()
    df["year"] = df["year"].astype(int)
    df["source_ccode"] = df["source_ccode"].astype(int)
    df["target_ccode"] = df["target_ccode"].astype(int)
    df["tie"] = df["tie"].astype(float)
    return df


def discover_layers(data_dir: Path, layer_file_map: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
    """Discover and load layer CSV files from data directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing layer CSV files.
    layer_file_map : dict or None
        Explicit mapping of layer_name -> filename. If None, attempts to
        discover files using naming conventions.

    Returns
    -------
    dict of layer_name -> DataFrame
    """
    if layer_file_map is None:
        # Default file discovery patterns
        layer_file_map = {}
        candidates = {
            "offensive_alliances": [
                "layer_offensive_alliances.csv",
                "layer_alliances_offensive_undirected.csv",
                "offensive_alliances.csv",
            ],
            "defensive_alliances": [
                "layer_defensive_alliances.csv",
                "layer_alliances_defensive_undirected.csv",
                "defensive_alliances.csv",
            ],
            "igo": [
                "layer_igo_shared_undirected.csv",
                "layer_igo.csv",
                "igo.csv",
            ],
            "trade": [
                "layer_trade_undirected.csv",
                "layer_trade.csv",
                "trade.csv",
            ],
            "dca": [
                "layer_dca.csv",
                "layer_dcas.csv",
                "dca.csv",
                "dcas.csv",
            ],
        }
        # Also try the combined alliance file as fallback
        combined_alliance_candidates = [
            "layer_alliances_defensive_offensive_undirected.csv",
        ]

        for layer_name, filenames in candidates.items():
            for fn in filenames:
                if (data_dir / fn).exists():
                    layer_file_map[layer_name] = fn
                    break

        # If separate offensive/defensive not found, try combined alliance file
        if "offensive_alliances" not in layer_file_map and "defensive_alliances" not in layer_file_map:
            for fn in combined_alliance_candidates:
                if (data_dir / fn).exists():
                    layer_file_map["defensive_alliances"] = fn
                    print(f"  Using combined alliance file '{fn}' as defensive_alliances proxy")
                    break

    layers = {}
    for layer_name, filename in layer_file_map.items():
        path = data_dir / filename
        if path.exists():
            layers[layer_name] = load_layer_csv(path)
            print(f"  Loaded layer '{layer_name}' from {filename}: {len(layers[layer_name])} rows")
        else:
            print(f"  WARNING: Layer file not found: {path}")
    return layers


def discover_weighted_layers(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Discover pre-residualized (weighted) layer CSVs from R export."""
    candidates = {
        "offensive_alliances": [
            "layer_offensive_alliances_weighted.csv",
            "layer_alliances_offensive_undirected_weighted.csv",
        ],
        "defensive_alliances": [
            "layer_defensive_alliances_weighted.csv",
            "layer_alliances_defensive_undirected_weighted.csv",
        ],
        "igo": [
            "layer_igo_shared_undirected_weighted.csv",
            "layer_igo_weighted.csv",
        ],
        "trade": [
            "layer_trade_undirected_weighted.csv",
            "layer_trade_weighted.csv",
        ],
        "dca": [
            "layer_dca_weighted.csv",
            "layer_dcas_weighted.csv",
        ],
    }
    # Fallback: combined alliance weighted
    combined = ["layer_alliances_defensive_offensive_undirected_weighted.csv"]

    layers = {}
    for layer_name, filenames in candidates.items():
        for fn in filenames:
            if (data_dir / fn).exists():
                layers[layer_name] = load_layer_csv(data_dir / fn)
                print(f"  Loaded weighted layer '{layer_name}' from {fn}")
                break

    if "offensive_alliances" not in layers and "defensive_alliances" not in layers:
        for fn in combined:
            if (data_dir / fn).exists():
                layers["defensive_alliances"] = load_layer_csv(data_dir / fn)
                print(f"  Using combined weighted alliance file '{fn}' as defensive_alliances proxy")
                break

    return layers


def build_global_node_index(
    layer_dfs: Dict[str, pd.DataFrame],
    nodes_csv: Optional[Path] = None,
) -> Tuple[Dict[int, int], Dict[int, int], pd.DataFrame]:
    """Build a deterministic global node index from all layers and optional nodes file.

    Returns (ccode_to_idx, idx_to_ccode, nodes_df).
    """
    all_ccodes: Set[int] = set()
    for df in layer_dfs.values():
        all_ccodes |= set(df["source_ccode"].unique()) | set(df["target_ccode"].unique())

    nodes_df = None
    if nodes_csv is not None and nodes_csv.exists():
        nodes_df = pd.read_csv(nodes_csv)
        if "ccode" in nodes_df.columns:
            all_ccodes |= set(nodes_df["ccode"].astype(int).unique())

    sorted_ccodes = sorted(all_ccodes)
    ccode_to_idx = {cc: i for i, cc in enumerate(sorted_ccodes)}
    idx_to_ccode = {i: cc for cc, i in ccode_to_idx.items()}

    if nodes_df is None:
        nodes_df = pd.DataFrame({"ccode": sorted_ccodes})

    return ccode_to_idx, idx_to_ccode, nodes_df


def compute_degree_features(
    edge_indices: Dict[str, torch.LongTensor],
    edge_weights: Dict[str, torch.FloatTensor],
    layer_mask: Dict[str, bool],
    num_nodes: int,
    layer_names: List[str],
) -> torch.FloatTensor:
    """Compute cross-layer degree profile as node features.

    For each layer, compute the weighted degree of each node. The feature
    vector is [degree_layer1, degree_layer2, ..., degree_layerL] normalized.
    """
    features = torch.zeros(num_nodes, len(layer_names))
    for l_idx, layer_name in enumerate(layer_names):
        if not layer_mask.get(layer_name, False):
            continue
        if layer_name not in edge_indices:
            continue
        ei = edge_indices[layer_name]
        ew = edge_weights[layer_name]
        if ei.numel() == 0:
            continue
        # Weighted degree: sum of edge weights for each node
        deg = torch.zeros(num_nodes)
        deg.scatter_add_(0, ei[0], ew)
        deg.scatter_add_(0, ei[1], ew)
        features[:, l_idx] = deg

    # Normalize per-layer to [0, 1]
    maxvals = features.max(dim=0).values.clamp(min=1e-8)
    features = features / maxvals.unsqueeze(0)
    return features


def build_multiplex_dataset(
    layer_dfs: Dict[str, pd.DataFrame],
    ccode_to_idx: Dict[int, int],
    idx_to_ccode: Dict[int, int],
    nodes_df: pd.DataFrame,
    year_range: Optional[Tuple[int, int]] = None,
    layer_names: Optional[List[str]] = None,
) -> MultiplexTemporalDataset:
    """Convert layer DataFrames into a MultiplexTemporalDataset of tensor snapshots.

    Parameters
    ----------
    layer_dfs : dict
        layer_name -> DataFrame with columns [year, source_ccode, target_ccode, tie]
    ccode_to_idx : dict
        Country code to node index mapping.
    idx_to_ccode : dict
        Node index to country code mapping.
    nodes_df : DataFrame
        Node metadata.
    year_range : tuple or None
        (start_year, end_year) inclusive. If None, use all years found in data.
    layer_names : list or None
        Ordered list of layer names. If None, use LAYER_NAMES filtered by available data.
    """
    if layer_names is None:
        layer_names = [ln for ln in LAYER_NAMES if ln in layer_dfs]

    num_nodes = len(ccode_to_idx)

    # Determine year range
    all_years: Set[int] = set()
    for df in layer_dfs.values():
        all_years |= set(df["year"].unique())

    if year_range is not None:
        years = sorted(y for y in range(year_range[0], year_range[1] + 1))
    else:
        years = sorted(all_years)

    # Determine which years each layer covers
    layer_year_coverage: Dict[str, Set[int]] = {}
    for ln in layer_names:
        if ln in layer_dfs:
            layer_year_coverage[ln] = set(layer_dfs[ln]["year"].unique())
        else:
            layer_year_coverage[ln] = set()

    snapshots: Dict[int, MultiplexSnapshot] = {}
    for year in years:
        edge_indices: Dict[str, torch.LongTensor] = {}
        edge_weights: Dict[str, torch.FloatTensor] = {}
        layer_mask: Dict[str, bool] = {}

        for ln in layer_names:
            available = year in layer_year_coverage.get(ln, set())
            layer_mask[ln] = available

            if not available or ln not in layer_dfs:
                edge_indices[ln] = torch.zeros(2, 0, dtype=torch.long)
                edge_weights[ln] = torch.zeros(0)
                continue

            df = layer_dfs[ln]
            year_df = df[df["year"] == year]

            if len(year_df) == 0:
                edge_indices[ln] = torch.zeros(2, 0, dtype=torch.long)
                edge_weights[ln] = torch.zeros(0)
                layer_mask[ln] = False
                continue

            src = year_df["source_ccode"].map(ccode_to_idx)
            tgt = year_df["target_ccode"].map(ccode_to_idx)
            weights = year_df["tie"].values.astype(np.float32)

            # Drop any edges with unmapped nodes
            valid = src.notna() & tgt.notna()
            src = src[valid].astype(int).values
            tgt = tgt[valid].astype(int).values
            weights = weights[valid.values]

            if len(src) == 0:
                edge_indices[ln] = torch.zeros(2, 0, dtype=torch.long)
                edge_weights[ln] = torch.zeros(0)
                layer_mask[ln] = False
                continue

            ei = torch.tensor(np.stack([src, tgt], axis=0), dtype=torch.long)
            ew = torch.tensor(weights, dtype=torch.float32)
            edge_indices[ln] = ei
            edge_weights[ln] = ew

        # Compute degree-profile node features
        node_features = compute_degree_features(
            edge_indices, edge_weights, layer_mask, num_nodes, layer_names,
        )

        snapshots[year] = MultiplexSnapshot(
            year=year,
            num_nodes=num_nodes,
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            layer_mask=layer_mask,
            node_features=node_features,
        )

    return MultiplexTemporalDataset(
        years=years,
        num_nodes=num_nodes,
        layer_names=layer_names,
        ccode_to_idx=ccode_to_idx,
        idx_to_ccode=idx_to_ccode,
        snapshots=snapshots,
        nodes_df=nodes_df,
    )


def save_dataset(dataset: MultiplexTemporalDataset, out_dir: Path) -> None:
    """Save processed dataset to disk for inspection and reloading."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save node mapping
    mapping = pd.DataFrame([
        {"ccode": cc, "idx": idx} for cc, idx in dataset.ccode_to_idx.items()
    ])
    mapping.to_csv(out_dir / "node_mapping.csv", index=False)

    # Save layer availability per year
    rows = []
    for year in dataset.years:
        snap = dataset.snapshots[year]
        row = {"year": year}
        for ln in dataset.layer_names:
            row[f"{ln}_available"] = snap.layer_mask.get(ln, False)
            row[f"{ln}_edges"] = snap.edge_indices[ln].shape[1] if ln in snap.edge_indices else 0
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "layer_availability.csv", index=False)

    # Save tensor snapshots
    tensor_dir = out_dir / "tensors"
    tensor_dir.mkdir(exist_ok=True)
    for year, snap in dataset.snapshots.items():
        torch.save({
            "year": year,
            "node_features": snap.node_features,
            "edge_indices": snap.edge_indices,
            "edge_weights": snap.edge_weights,
            "layer_mask": snap.layer_mask,
        }, tensor_dir / f"snapshot_{year}.pt")

    print(f"Saved dataset: {len(dataset.years)} years, {dataset.num_nodes} nodes, "
          f"{len(dataset.layer_names)} layers to {out_dir}")
