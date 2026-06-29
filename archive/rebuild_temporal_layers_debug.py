#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
from typing import Dict, List, Set

import pandas as pd

# ============================================================
# 1) Parse CLI arguments (script runs top-to-bottom)
# ============================================================
arg_parser = argparse.ArgumentParser(
    description="Load temporal layer CSVs, print diagnostics, and build per-year objects for GNN inputs.")
arg_parser.add_argument("--data-dir", default="data/processed", help="Directory containing layer_*.csv files")
arg_parser.add_argument(
    "--layer-files",
    nargs="*",
    default=None,
    help="Optional explicit list of CSV files (relative to --data-dir). If omitted, all layer_*.csv are used.",
)
arg_parser.add_argument("--start-year", type=int, default=None, help="Optional start year for required range")
arg_parser.add_argument("--end-year", type=int, default=None, help="Optional end year for required range")
arg_parser.add_argument(
    "--require-complete-layers",
    action="store_true",
    help="If set, fail loudly when any required year is missing in any layer.",
)
arg_parser.add_argument(
    "--allow-empty-year",
    action="store_true",
    help="If set, do not fail when a year has no edges in a layer that is marked available.",
)
arg_parser.add_argument(
    "--summary-out-dir",
    default=None,
    help="Optional directory to export diagnostics CSVs.",
)
cli_args = arg_parser.parse_args()

# ============================================================
# 2) Resolve input files and define schema aliases
# ============================================================
data_dir = Path(cli_args.data_dir)
assert data_dir.exists(), f"Data directory does not exist: {data_dir}"

if cli_args.layer_files:
    layer_file_paths = [data_dir / relative_name for relative_name in cli_args.layer_files]
else:
    layer_file_paths = sorted(data_dir.glob("layer_*.csv"))

assert layer_file_paths, f"No layer CSV files found in: {data_dir}"

# Accept multiple common schemas and normalize to:
# year, source_ccode, target_ccode, value_raw, value_resid
column_aliases = {
    "year": ["year"],
    "source_ccode": ["source_ccode", "i_ccode", "src", "source"],
    "target_ccode": ["target_ccode", "j_ccode", "dst", "target"],
    "value_raw": ["value_raw", "tie", "value", "weight"],
    "value_resid": ["value_resid", "resid", "residual"],
}


# ============================================================
# 3) Helper functions (kept local for single-file readability)
# ============================================================
def find_column(df_columns: List[str], canonical_name: str) -> str | None:
    aliases = column_aliases[canonical_name]
    for alias in aliases:
        if alias in df_columns:
            return alias
    return None


def normalize_layer_dataframe(layer_name: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    df_columns = list(raw_df.columns)

    year_col = find_column(df_columns, "year")
    source_col = find_column(df_columns, "source_ccode")
    target_col = find_column(df_columns, "target_ccode")
    value_raw_col = find_column(df_columns, "value_raw")
    value_resid_col = find_column(df_columns, "value_resid")

    required_missing = [
        name for name, col in [
            ("year", year_col),
            ("source_ccode", source_col),
            ("target_ccode", target_col),
            ("value_raw", value_raw_col),
        ] if col is None
    ]
    assert not required_missing, (
        f"Layer '{layer_name}' missing required columns after alias matching: {required_missing}. "
        f"Columns found: {df_columns}"
    )

    normalized_df = pd.DataFrame({
        "year": raw_df[year_col],
        "source_ccode": raw_df[source_col],
        "target_ccode": raw_df[target_col],
        "value_raw": raw_df[value_raw_col],
        "value_resid": raw_df[value_resid_col] if value_resid_col is not None else pd.NA,
    }).copy()

    normalized_df["year"] = normalized_df["year"].astype(int)
    normalized_df["source_ccode"] = normalized_df["source_ccode"].astype(int)
    normalized_df["target_ccode"] = normalized_df["target_ccode"].astype(int)
    normalized_df["value_raw"] = pd.to_numeric(normalized_df["value_raw"], errors="coerce")
    normalized_df["value_resid"] = pd.to_numeric(normalized_df["value_resid"], errors="coerce")

    normalized_df = normalized_df.dropna(subset=["year", "source_ccode", "target_ccode", "value_raw"])
    return normalized_df


# ============================================================
# 4) Load all layer CSVs into dataframes
# ============================================================
layer_dataframes: Dict[str, pd.DataFrame] = {}
for csv_path in layer_file_paths:
    assert csv_path.exists(), f"Layer file does not exist: {csv_path}"

    layer_name = csv_path.stem
    raw_layer_df = pd.read_csv(csv_path)
    assert not raw_layer_df.empty, f"Layer '{layer_name}' is empty: {csv_path}"

    normalized_layer_df = normalize_layer_dataframe(layer_name, raw_layer_df)
    assert not normalized_layer_df.empty, f"Layer '{layer_name}' has no valid rows after normalization"

    layer_dataframes[layer_name] = normalized_layer_df

print(f"Loaded layer count = {len(layer_dataframes)}")
print(f"Loaded layers = {sorted(layer_dataframes.keys())}")

# ============================================================
# 5) Compute year coverage and required year list
# ============================================================
years_by_layer: Dict[str, List[int]] = {}
for layer_name, layer_df in layer_dataframes.items():
    year_list = sorted(layer_df["year"].unique().tolist())
    years_by_layer[layer_name] = year_list

all_years: List[int] = sorted({year for year_list in years_by_layer.values() for year in year_list})
assert all_years, "No years found across loaded layers"

if cli_args.start_year is not None or cli_args.end_year is not None:
    assert cli_args.start_year is not None and cli_args.end_year is not None, (
        "Provide both --start-year and --end-year when requesting a required year range"
    )
    assert cli_args.start_year <= cli_args.end_year, "start-year must be <= end-year"
    required_year_list = list(range(cli_args.start_year, cli_args.end_year + 1))
else:
    required_year_list = all_years

print("\n=== YEARS AVAILABLE PER LAYER ===")
for layer_name in sorted(years_by_layer):
    layer_year_list = years_by_layer[layer_name]
    missing_year_list = sorted(set(required_year_list) - set(layer_year_list))
    print(f"{layer_name}: min={min(layer_year_list)}, max={max(layer_year_list)}, count={len(layer_year_list)}")
    print(f"{layer_name} missing years = {missing_year_list}")

if cli_args.require_complete_layers:
    for layer_name, layer_year_list in years_by_layer.items():
        missing_year_list = sorted(set(required_year_list) - set(layer_year_list))
        assert not missing_year_list, (
            f"Layer '{layer_name}' missing required years: {missing_year_list}"
        )

# ============================================================
# 6) Build summary tables (years/layers/nodes/dyads)
# ============================================================
summary_years_rows = []
summary_nodes_rows = []
summary_dyads_rows = []

for layer_name, layer_df in layer_dataframes.items():
    available_year_set: Set[int] = set(layer_df["year"].unique().tolist())

    for year_t in required_year_list:
        year_is_available = year_t in available_year_set
        year_slice_df = layer_df[layer_df["year"] == year_t]

        node_set_t: Set[int] = set(year_slice_df["source_ccode"].tolist()) | set(year_slice_df["target_ccode"].tolist())
        dyad_count_t = int(len(year_slice_df))

        summary_years_rows.append({
            "layer": layer_name,
            "year": year_t,
            "is_available": int(year_is_available),
        })
        summary_nodes_rows.append({
            "layer": layer_name,
            "year": year_t,
            "node_count": len(node_set_t),
            "nodes": sorted(node_set_t),
        })
        summary_dyads_rows.append({
            "layer": layer_name,
            "year": year_t,
            "dyad_count": dyad_count_t,
        })

summary_years_df = pd.DataFrame(summary_years_rows)
summary_nodes_df = pd.DataFrame(summary_nodes_rows)
summary_dyads_df = pd.DataFrame(summary_dyads_rows)

print("\n=== NODES PRESENT PER YEAR & LAYER ===")
print(summary_nodes_df[["layer", "year", "node_count"]].to_string(index=False))

print("\n=== DYAD COUNTS PER YEAR & LAYER ===")
print(summary_dyads_df.to_string(index=False))

# ============================================================
# 7) Build per-year objects with explicit layer masks
# ============================================================
per_year_objects: Dict[int, Dict[str, object]] = {}

for year_t in required_year_list:
    layers_for_year: Dict[str, Dict[str, object]] = {}
    union_nodes_t: Set[int] = set()

    for layer_name, layer_df in layer_dataframes.items():
        year_slice_df = layer_df[layer_df["year"] == year_t].copy()
        layer_available_t = not year_slice_df.empty

        layer_node_set_t: Set[int] = set(year_slice_df["source_ccode"].tolist()) | set(year_slice_df["target_ccode"].tolist())
        union_nodes_t |= layer_node_set_t

        if layer_available_t and not cli_args.allow_empty_year:
            assert len(year_slice_df) > 0, f"Layer '{layer_name}' year {year_t} should not be empty"

        layers_for_year[layer_name] = {
            "is_available": int(layer_available_t),
            "node_set": sorted(layer_node_set_t),
            "edge_count": int(len(year_slice_df)),
            "edge_df": year_slice_df,
        }

    per_year_objects[year_t] = {
        "year": year_t,
        "nodes_t": sorted(union_nodes_t),
        "node_count_t": len(union_nodes_t),
        "layer_data": layers_for_year,
    }

# ============================================================
# 8) Loud checks for assumptions about year/layer availability
# ============================================================
for year_t, year_object in per_year_objects.items():
    assert year_object["node_count_t"] > 0, f"No nodes present in required year: {year_t}"

    for layer_name, layer_object in year_object["layer_data"].items():
        if cli_args.require_complete_layers:
            assert layer_object["is_available"] == 1, (
                f"Layer '{layer_name}' missing for required year {year_t} while --require-complete-layers is enabled"
            )

print("\n=== LAYER AVAILABILITY MASKS BY YEAR ===")
for year_t in required_year_list:
    layer_mask_t = {layer_name: per_year_objects[year_t]["layer_data"][layer_name]["is_available"] for layer_name in sorted(layer_dataframes)}
    print(f"year={year_t} mask={layer_mask_t} nodes={per_year_objects[year_t]['node_count_t']}")

# ============================================================
# 9) Optional diagnostics export
# ============================================================
if cli_args.summary_out_dir is not None:
    summary_out_dir = Path(cli_args.summary_out_dir)
    summary_out_dir.mkdir(parents=True, exist_ok=True)

    summary_years_path = summary_out_dir / "summary_years_by_layer.csv"
    summary_nodes_path = summary_out_dir / "summary_nodes_per_year_layer.csv"
    summary_dyads_path = summary_out_dir / "summary_dyads_per_year_layer.csv"

    summary_years_df.to_csv(summary_years_path, index=False)
    summary_nodes_df.to_csv(summary_nodes_path, index=False)
    summary_dyads_df.to_csv(summary_dyads_path, index=False)

    print("\n=== EXPORTED DIAGNOSTICS ===")
    print(summary_years_path)
    print(summary_nodes_path)
    print(summary_dyads_path)

print("\nFinished: temporal layer diagnostics and per-year object construction complete.")
