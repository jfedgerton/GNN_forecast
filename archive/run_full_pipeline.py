#!/usr/bin/env python3
"""End-to-end pipeline: data loading -> training -> forecasting -> counterfactual analysis.

Usage:
    python scripts/run_full_pipeline.py --data-dir data/processed --start-year 1950 --end-year 2014

This script runs both Model Set 1 (raw layers) and Model Set 2 (strength-weighted layers)
sequentially, producing separate outputs for each.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from gnn_forecast.multiplex_data import (
    build_global_node_index,
    build_multiplex_dataset,
    discover_layers,
    discover_weighted_layers,
    save_dataset,
    LAYER_NAMES,
)
from gnn_forecast.multiplex_model import MultiplexGNNConfig, MultiplexTemporalGNN
from gnn_forecast.training import TrainingConfig, train_model, forecast_embeddings
from gnn_forecast.counterfactual import (
    batch_counterfactual_analysis,
    results_to_dataframe,
    summarize_by_layer,
    top_interventions,
    USA_CCODE,
    CHN_CCODE,
)

# ============================================================
# CLI arguments
# ============================================================
ap = argparse.ArgumentParser(description="Multiplex temporal GNN pipeline")
ap.add_argument("--data-dir", default="data/processed", help="Directory with layer CSVs")
ap.add_argument("--start-year", type=int, default=1950, help="First year to include")
ap.add_argument("--end-year", type=int, default=2014, help="Last year to include")
ap.add_argument("--out-dir", default="outputs", help="Output directory")
ap.add_argument("--epochs", type=int, default=100, help="Training epochs")
ap.add_argument("--seq-len", type=int, default=5, help="Temporal window length")
ap.add_argument("--emb-dim", type=int, default=32, help="Embedding dimension")
ap.add_argument("--hidden-dim", type=int, default=64, help="GCN hidden dimension")
ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
ap.add_argument("--forecast-steps", type=int, default=25, help="Forecast horizon (years)")
ap.add_argument("--skip-weighted", action="store_true", help="Skip Model Set 2 (weighted layers)")
ap.add_argument("--skip-counterfactual", action="store_true", help="Skip counterfactual analysis")
ap.add_argument("--cf-max-partners", type=int, default=50, help="Max partners for counterfactual (0=all)")
args = ap.parse_args()

data_dir = Path(args.data_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Helper: run one full model set
# ============================================================
def run_model_set(
    label: str,
    layer_dfs: dict,
    ccode_to_idx: dict,
    idx_to_ccode: dict,
    nodes_df: pd.DataFrame,
    model_out_dir: Path,
):
    print(f"\n{'='*60}")
    print(f"MODEL SET: {label}")
    print(f"{'='*60}")

    # Build dataset
    print("\n--- Building multiplex dataset ---")
    dataset = build_multiplex_dataset(
        layer_dfs=layer_dfs,
        ccode_to_idx=ccode_to_idx,
        idx_to_ccode=idx_to_ccode,
        nodes_df=nodes_df,
        year_range=(args.start_year, args.end_year),
    )

    # Print layer availability summary
    avail_years = {}
    for ln in dataset.layer_names:
        avail_years[ln] = sum(
            1 for y in dataset.years if dataset.snapshots[y].layer_mask.get(ln, False)
        )
    print(f"  Layers and availability:")
    for ln, count in avail_years.items():
        print(f"    {ln}: {count}/{len(dataset.years)} years")

    # Save processed data
    data_out = model_out_dir / "processed_data"
    save_dataset(dataset, data_out)

    # Configure model
    active_layers = dataset.layer_names
    model_config = MultiplexGNNConfig(
        num_layers=len(active_layers),
        in_dim=len(active_layers),
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        temporal_hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
    )

    train_config = TrainingConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        seq_len=args.seq_len,
        save_dir=str(model_out_dir / "model"),
        print_every=max(1, args.epochs // 10),
    )

    # Train
    print("\n--- Training ---")
    result = train_model(
        dataset=dataset,
        model_config=model_config,
        train_config=train_config,
        device=device,
    )

    print(f"\n  Final loss: {result.loss_history[-1]:.6f}")
    print(f"  Layer weights: {result.layer_weights}")

    # Forecast
    print("\n--- Forecasting ---")
    forecasts = forecast_embeddings(
        model=result.model,
        dataset=dataset,
        n_steps=args.forecast_steps,
        seq_len=args.seq_len,
        device=device,
    )

    # Save forecasts
    forecast_dir = model_out_dir / "forecasts"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    torch.save(forecasts, forecast_dir / "forecast_embeddings.pt")

    # Export yearly embeddings as CSVs for inspection
    emb_dir = model_out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    for year, emb in result.yearly_embeddings.items():
        rows = []
        for idx in range(emb.size(0)):
            ccode = idx_to_ccode.get(idx, idx)
            row = {"ccode": ccode, "year": year}
            for d in range(emb.size(1)):
                row[f"dim_{d}"] = float(emb[idx, d])
            rows.append(row)
        pd.DataFrame(rows).to_csv(emb_dir / f"embeddings_{year}.csv", index=False)

    print(f"  Saved embeddings for {len(result.yearly_embeddings)} years")
    print(f"  Saved forecasts for {len(forecasts)} future years")

    # Counterfactual analysis
    if not args.skip_counterfactual:
        print("\n--- Counterfactual Analysis ---")
        cf_dir = model_out_dir / "counterfactual"
        cf_dir.mkdir(parents=True, exist_ok=True)

        # Select partners for counterfactual
        if args.cf_max_partners > 0:
            all_partners = sorted([
                cc for cc in ccode_to_idx.keys()
                if cc not in (USA_CCODE, CHN_CCODE)
            ])
            # Sample evenly if too many
            if len(all_partners) > args.cf_max_partners:
                step = len(all_partners) // args.cf_max_partners
                partners = all_partners[::step][:args.cf_max_partners]
            else:
                partners = all_partners
        else:
            partners = None  # all

        for focal_label, focal_ccode in [("usa", USA_CCODE), ("china", CHN_CCODE)]:
            if focal_ccode not in ccode_to_idx:
                print(f"  Skipping {focal_label}: ccode {focal_ccode} not in dataset")
                continue

            print(f"  Running counterfactual for {focal_label} (ccode={focal_ccode})...")
            cf_results = batch_counterfactual_analysis(
                model=result.model,
                dataset=dataset,
                focal_ccode=focal_ccode,
                partner_ccodes=partners,
                seq_len=args.seq_len,
                device=device,
            )

            if cf_results:
                cf_df = results_to_dataframe(cf_results)
                cf_df.to_csv(cf_dir / f"counterfactual_{focal_label}.csv", index=False)

                summary = summarize_by_layer(cf_df)
                summary.to_csv(cf_dir / f"summary_by_layer_{focal_label}.csv", index=False)

                top = top_interventions(cf_df, n=20)
                top.to_csv(cf_dir / f"top_interventions_{focal_label}.csv", index=False)

                print(f"    {len(cf_results)} simulations completed")
                print(f"    Top 5 interventions for {focal_label}:")
                for _, row in top.head(5).iterrows():
                    print(f"      {row['operation']} {row['layer']} with ccode={row['partner_ccode']}: "
                          f"delta_cosine={row['delta_mean_cosine']:.6f}")
            else:
                print(f"    No counterfactual results for {focal_label}")

    return result


# ============================================================
# Load data
# ============================================================
print("Loading layer data...")
print(f"Data directory: {data_dir}")

raw_layers = discover_layers(data_dir)
if not raw_layers:
    raise FileNotFoundError(f"No layer files found in {data_dir}")

# Build global node index from raw layers
nodes_csv = data_dir / "nodes.csv"
ccode_to_idx, idx_to_ccode, nodes_df = build_global_node_index(raw_layers, nodes_csv)
print(f"Global node index: {len(ccode_to_idx)} states")

# ============================================================
# Model Set 1: Raw layers
# ============================================================
raw_result = run_model_set(
    label="Raw Layers",
    layer_dfs=raw_layers,
    ccode_to_idx=ccode_to_idx,
    idx_to_ccode=idx_to_ccode,
    nodes_df=nodes_df,
    model_out_dir=out_dir / "model_set_1_raw",
)

# ============================================================
# Model Set 2: Strength-weighted layers
# ============================================================
if not args.skip_weighted:
    weighted_layers = discover_weighted_layers(data_dir)
    if weighted_layers:
        weighted_result = run_model_set(
            label="Strength-Weighted Layers",
            layer_dfs=weighted_layers,
            ccode_to_idx=ccode_to_idx,
            idx_to_ccode=idx_to_ccode,
            nodes_df=nodes_df,
            model_out_dir=out_dir / "model_set_2_weighted",
        )
    else:
        print("\nNo weighted layer files found. Skipping Model Set 2.")
        print("Run the R export script to generate weighted layers, or use")
        print("--skip-weighted to suppress this message.")

# ============================================================
# Done
# ============================================================
print(f"\n{'='*60}")
print("Pipeline complete.")
print(f"Outputs saved to: {out_dir}")
print(f"{'='*60}")
