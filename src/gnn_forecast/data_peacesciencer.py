from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


@dataclass
class MultiplexYear:
    year: int
    nodes: pd.DataFrame
    layers: Dict[str, pd.DataFrame]


REQUIRED_LAYERS = {
    "alliances_defensive_offensive_undirected": "Defensive + offensive alliance ties",
    "igo_shared_undirected": "Shared IGO memberships",
    "trade_undirected": "Directed/undirected trade flow tie",
}


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def build_multiplex_from_peacesciencer(
    output_dir: str | Path,
    years: Iterable[int] = range(1945, 2026),
) -> Dict[int, MultiplexYear]:
    """
    Load yearly multiplex layers previously exported from peacesciencer.

    Expected files under output_dir:
      - nodes.csv (columns: ccode, state_name, cap_lat, cap_lon)
      - layer_alliances_defensive_offensive_undirected.csv
      - layer_igo_shared_undirected.csv
      - layer_trade_undirected.csv

    Each layer CSV format: year, source_ccode, target_ccode, tie

    See scripts/export_peacesciencer_layers.R for producing these tables.
    """
    output_dir = Path(output_dir)
    nodes = _read_csv_if_exists(output_dir / "nodes.csv")
    if nodes is None:
        raise FileNotFoundError(
            f"{output_dir}/nodes.csv not found. Run scripts/export_peacesciencer_layers.R first."
        )

    layer_tables: Dict[str, pd.DataFrame] = {}
    for layer_name in REQUIRED_LAYERS:
        path = output_dir / f"layer_{layer_name}.csv"
        table = _read_csv_if_exists(path)
        if table is None:
            raise FileNotFoundError(f"Missing layer export: {path}")
        layer_tables[layer_name] = table

    years_set = set(years)
    by_year: Dict[int, MultiplexYear] = {}
    for year in sorted(years_set):
        year_layers: Dict[str, pd.DataFrame] = {}
        for name, table in layer_tables.items():
            year_layers[name] = table.loc[table["year"] == year, ["source_ccode", "target_ccode", "tie"]].copy()
        by_year[year] = MultiplexYear(year=year, nodes=nodes.copy(), layers=year_layers)
    return by_year
