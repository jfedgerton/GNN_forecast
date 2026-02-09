from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .data_peacesciencer import MultiplexYear, build_multiplex_from_peacesciencer
from .network_construction import make_observed_and_residual_layers


@dataclass
class PipelineConfig:
    data_dir: str = "data/processed"
    years_start: int = 1945
    years_end: int = 2025


def run_research_pipeline(config: PipelineConfig) -> Dict[int, dict]:
    by_year: Dict[int, MultiplexYear] = build_multiplex_from_peacesciencer(
        output_dir=config.data_dir,
        years=range(config.years_start, config.years_end + 1),
    )

    out: Dict[int, dict] = {}
    for y, rec in by_year.items():
        bundle = make_observed_and_residual_layers(rec.nodes, rec.layers)
        out[y] = {
            "observed": bundle.observed_layers,
            "residual": bundle.residual_layers,
        }
    return out


def export_year_layers(layer_map: Dict[int, dict], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for year, contents in layer_map.items():
        for mode in ["observed", "residual"]:
            for lname, df in contents[mode].items():
                path = output_dir / f"{mode}_{lname}_{year}.csv"
                df.to_csv(path, index=False)
