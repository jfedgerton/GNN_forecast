from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


@dataclass
class RelationFileSpec:
    """Schema details for one relation edge-list file."""

    filename: str
    year_col: str = "year"
    source_col: str = "source_ccode"
    target_col: str = "target_ccode"
    tie_col: str = "tie"


@dataclass
class CanonicalDataConfig:
    """Configuration for loading canonical nodes/edges from processed CSVs."""

    data_dir: str = "data/processed"
    nodes_file: str = "nodes.csv"
    node_ccode_col: str = "ccode"
    relation_specs: Dict[str, RelationFileSpec] = field(
        default_factory=lambda: {
            "alliance": RelationFileSpec(filename="edges_alliance.csv"),
            "igo": RelationFileSpec(filename="edges_igo.csv"),
            "trade": RelationFileSpec(filename="edges_trade.csv"),
        }
    )


@dataclass
class CanonicalMultiplexData:
    """Canonical node map and yearly relation edge tables."""

    nodes: pd.DataFrame
    ccode_to_idx: Dict[int, int]
    idx_to_ccode: Dict[int, int]
    yearly_edges: Dict[int, Dict[str, pd.DataFrame]]


def build_ccode_index(nodes: pd.DataFrame, ccode_col: str = "ccode") -> Dict[int, int]:
    """Build deterministic integer index by sorted unique ccode."""
    ccodes = sorted({int(v) for v in nodes[ccode_col].tolist()})
    return {ccode: idx for idx, ccode in enumerate(ccodes)}


def _load_nodes(config: CanonicalDataConfig) -> pd.DataFrame:
    nodes_path = Path(config.data_dir) / config.nodes_file
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes file: {nodes_path}")

    nodes = pd.read_csv(nodes_path)
    if config.node_ccode_col not in nodes.columns:
        raise ValueError(f"Missing node ccode column '{config.node_ccode_col}' in {nodes_path}")

    nodes = nodes.copy()
    nodes[config.node_ccode_col] = nodes[config.node_ccode_col].astype(int)
    return nodes.sort_values(config.node_ccode_col).drop_duplicates(subset=[config.node_ccode_col]).reset_index(drop=True)


def _load_edges(config: CanonicalDataConfig) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for relation, spec in config.relation_specs.items():
        edge_path = Path(config.data_dir) / spec.filename
        if not edge_path.exists():
            raise FileNotFoundError(f"Missing relation file for '{relation}': {edge_path}")

        table = pd.read_csv(edge_path)
        required = [spec.year_col, spec.source_col, spec.target_col, spec.tie_col]
        missing = [c for c in required if c not in table.columns]
        if missing:
            raise ValueError(f"Missing columns in {edge_path}: {missing}")

        clean = table[[spec.year_col, spec.source_col, spec.target_col, spec.tie_col]].copy()
        clean.columns = ["year", "source_ccode", "target_ccode", "tie"]
        clean["year"] = clean["year"].astype(int)
        clean["source_ccode"] = clean["source_ccode"].astype(int)
        clean["target_ccode"] = clean["target_ccode"].astype(int)
        clean["tie"] = clean["tie"].astype(float)
        out[relation] = clean
    return out


def load_canonical_multiplex_data(
    config: CanonicalDataConfig,
    years: Iterable[int] | None = None,
) -> CanonicalMultiplexData:
    """Load yearly multiplex edge-lists with a single canonical node indexing."""
    nodes = _load_nodes(config)
    ccode_to_idx = build_ccode_index(nodes, ccode_col=config.node_ccode_col)
    idx_to_ccode = {v: k for k, v in ccode_to_idx.items()}
    edges_by_relation = _load_edges(config)

    valid_ccodes = set(ccode_to_idx.keys())
    yearly_edges: Dict[int, Dict[str, pd.DataFrame]] = {}

    all_years = sorted({int(y) for df in edges_by_relation.values() for y in df["year"].unique().tolist()})
    target_years = all_years if years is None else sorted({int(y) for y in years})

    for year in target_years:
        yearly_edges[year] = {}
        for relation, df in edges_by_relation.items():
            part = df.loc[df["year"] == year, ["source_ccode", "target_ccode", "tie"]].copy()
            unknown_ccodes = (set(part["source_ccode"]) | set(part["target_ccode"])) - valid_ccodes
            if unknown_ccodes:
                raise ValueError(
                    f"Relation '{relation}' year {year} has ccodes absent from nodes.csv: {sorted(unknown_ccodes)}"
                )
            part["source_idx"] = part["source_ccode"].map(ccode_to_idx)
            part["target_idx"] = part["target_ccode"].map(ccode_to_idx)
            yearly_edges[year][relation] = part.reset_index(drop=True)

    return CanonicalMultiplexData(
        nodes=nodes,
        ccode_to_idx=ccode_to_idx,
        idx_to_ccode=idx_to_ccode,
        yearly_edges=yearly_edges,
    )
