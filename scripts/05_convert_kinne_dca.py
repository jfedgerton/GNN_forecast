#!/usr/bin/env python3
"""Convert Kinne DCAD-v1.0-dyadic.csv into the standard layer CSV format,
filtering to COW system members in each year and validating the merge.

Input:  data/kinne_dca/DCAD-v1.0-dyadic.csv
        (cols: ccode1, abbrev1, ccode2, abbrev2, year, dcaGeneralV1, ...,
                dcaAnyV1, dcaAnyV2)
COW:    data/processed/cow_state_membership.csv
        (cols: ccode, year, in_cow, iso3)
Output: data/processed/layer_dca_undirected.csv
        (cols: year, source_ccode, target_ccode, tie)

Filtering:
  1. Keep dcaAnyV1 (high-confidence indicator per Kinne)
  2. Symmetrize (each undirected edge appears as both i→j and j→i)
  3. Drop self-loops
  4. **Restrict to dyads where BOTH ccodes are COW members in that year**

Merge validation prints:
  - rows before/after each filter
  - ccodes in Kinne but never in COW (orphans)
  - year coverage of the COW-filtered output
  - density of positive ties

Usage:
    python scripts/convert_kinne_dca.py
    python scripts/convert_kinne_dca.py --indicator dcaAnyV2  # use lower-conf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/kinne_dca/DCAD-v1.0-dyadic.csv")
    ap.add_argument("--cow-membership", default="data/processed/cow_state_membership.csv",
                    help="COW state-year membership CSV (build with export_cow_membership.R)")
    ap.add_argument("--output", default="data/processed/layer_dca_undirected.csv")
    ap.add_argument(
        "--indicator", default="dcaAnyV1",
        choices=["dcaAnyV1", "dcaAnyV2", "dcaGeneralV1", "dcaGeneralV2",
                 "dcaSectorV1", "dcaSectorV2"],
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    cow_path = Path(args.cow_membership)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Kinne DCA input not found: {in_path}")
    if not cow_path.exists():
        raise FileNotFoundError(
            f"COW membership CSV not found: {cow_path}. "
            f"Run scripts/export_cow_membership.R first."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load Kinne ----
    print(f"Loading {in_path}")
    df = pd.read_csv(in_path)
    print(f"  Kinne rows: {len(df):,}")
    print(f"  Kinne ccodes: {len(set(df['ccode1']) | set(df['ccode2'])):,} unique")
    print(f"  Kinne year range: {df['year'].min()}-{df['year'].max()}")
    if args.indicator not in df.columns:
        raise ValueError(f"Indicator column {args.indicator} not found")
    pos_before = (df[args.indicator] == 1).sum()
    print(f"  Kinne positive {args.indicator}: {pos_before:,}")

    # ---- Standardize ----
    df_std = pd.DataFrame({
        "year": df["year"].astype(int),
        "source_ccode": df["ccode1"].astype(int),
        "target_ccode": df["ccode2"].astype(int),
        "tie": df[args.indicator].fillna(0).astype(int),
    })
    df_rev = df_std.rename(
        columns={"source_ccode": "target_ccode", "target_ccode": "source_ccode"},
    )[["year", "source_ccode", "target_ccode", "tie"]]
    df_full = pd.concat([df_std, df_rev], ignore_index=True)
    df_full = df_full[df_full["source_ccode"] != df_full["target_ccode"]]
    df_full = df_full.drop_duplicates(subset=["year", "source_ccode", "target_ccode"])
    print(f"  Symmetrized + deduped: {len(df_full):,} rows")

    # ---- Load COW membership and filter ----
    cow = pd.read_csv(cow_path)
    cow = cow[["ccode", "year"]].drop_duplicates()
    cow_set = set(zip(cow["ccode"].astype(int), cow["year"].astype(int)))
    print(f"  COW membership panel: {len(cow_set):,} (ccode, year) cells")

    # Validate which Kinne ccodes never appear in COW (likely non-state entities or coding mismatches)
    kinne_ccodes = set(df_full["source_ccode"].unique()) | set(df_full["target_ccode"].unique())
    cow_ccodes = set(cow["ccode"].astype(int).unique())
    orphans = sorted(kinne_ccodes - cow_ccodes)
    if orphans:
        print(f"  WARNING: {len(orphans)} Kinne ccodes never in COW: {orphans[:20]}"
              + ("..." if len(orphans) > 20 else ""))

    # Apply the COW dyadic filter (both endpoints in COW in that year)
    src_ok = list(zip(df_full["source_ccode"], df_full["year"]))
    tgt_ok = list(zip(df_full["target_ccode"], df_full["year"]))
    src_in = pd.Series([p in cow_set for p in src_ok], index=df_full.index)
    tgt_in = pd.Series([p in cow_set for p in tgt_ok], index=df_full.index)
    keep = src_in & tgt_in
    n_before = len(df_full)
    df_filtered = df_full[keep].copy()
    print(f"  COW-dyadic filter kept {len(df_filtered):,} of {n_before:,} rows "
          f"({100 * len(df_filtered) / max(n_before, 1):.1f}%)")

    df_filtered = df_filtered.sort_values(["year", "source_ccode", "target_ccode"]).reset_index(drop=True)

    # ---- Final report ----
    pos_after = (df_filtered["tie"] == 1).sum()
    density = pos_after / max(len(df_filtered), 1)
    print(f"\nFinal output:")
    print(f"  rows: {len(df_filtered):,}")
    print(f"  positive ties: {pos_after:,} ({density:.4%} density)")
    print(f"  unique ccodes: {len(set(df_filtered['source_ccode']) | set(df_filtered['target_ccode'])):,}")
    print(f"  year range: {df_filtered['year'].min()}-{df_filtered['year'].max()}")

    df_filtered.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
