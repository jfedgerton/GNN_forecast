#!/usr/bin/env python3
"""Full manuscript analysis pipeline.

Runs everything needed for a Political Analysis submission:
1. Data loading and preprocessing
2. Model Set 1 (raw) and Model Set 2 (weighted) training
3. Walk-forward backtesting
4. Multi-seed robustness evaluation
5. Baseline model comparisons
6. Counterfactual analysis for USA and China
7. All visualization tables and figures

Usage:
    python scripts/run_manuscript_analysis.py --data-dir data/processed --out-dir outputs/manuscript
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn_forecast.multiplex_data import (
    build_global_node_index,
    build_multiplex_dataset,
    discover_layers,
    discover_weighted_layers,
    save_dataset,
)
from gnn_forecast.multiplex_model import MultiplexGNNConfig
from gnn_forecast.training import TrainingConfig, train_model, forecast_embeddings
from gnn_forecast.counterfactual import (
    batch_counterfactual_analysis,
    results_to_dataframe,
    summarize_by_layer,
    top_interventions,
    USA_CCODE,
    CHN_CCODE,
)
from gnn_forecast.validation import walk_forward_backtest, multi_seed_evaluation
from gnn_forecast.visualization import generate_all_tables, embeddedness_time_series
from gnn_forecast.baselines import compare_baselines
from gnn_forecast.isolation_analysis import run_isolation_analysis

# ============================================================
# CLI arguments
# ============================================================
ap = argparse.ArgumentParser(description="Full manuscript analysis pipeline")
ap.add_argument("--data-dir", default="data/processed")
ap.add_argument("--out-dir", default="outputs/manuscript")
ap.add_argument("--start-year", type=int, default=1950)
ap.add_argument("--end-year", type=int, default=2014)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--seq-len", type=int, default=5)
ap.add_argument("--emb-dim", type=int, default=32)
ap.add_argument("--hidden-dim", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds")
ap.add_argument("--forecast-steps", type=int, default=25)
ap.add_argument("--backtest-splits", nargs="*", type=int, default=None,
                help="Years for walk-forward splits (default: auto)")
ap.add_argument("--cf-max-partners", type=int, default=0, help="Max partners for CF (0=all)")
ap.add_argument("--skip-weighted", action="store_true")
ap.add_argument("--skip-backtest", action="store_true")
ap.add_argument("--skip-baselines", action="store_true")
ap.add_argument("--skip-multiseed", action="store_true")
ap.add_argument("--skip-counterfactual", action="store_true")
ap.add_argument("--skip-isolation", action="store_true")
args = ap.parse_args()

data_dir = Path(args.data_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Load data
# ============================================================
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

raw_layers = discover_layers(data_dir)
weighted_layers = discover_weighted_layers(data_dir) if not args.skip_weighted else {}

nodes_csv = data_dir / "nodes.csv"
all_layer_dfs = {**raw_layers}
if weighted_layers:
    all_layer_dfs.update({f"wt_{k}": v for k, v in weighted_layers.items()})
ccode_to_idx, idx_to_ccode, nodes_df = build_global_node_index(raw_layers, nodes_csv)
print(f"Global node index: {len(ccode_to_idx)} states")

# Build raw dataset
raw_dataset = build_multiplex_dataset(
    raw_layers, ccode_to_idx, idx_to_ccode, nodes_df,
    year_range=(args.start_year, args.end_year),
)
save_dataset(raw_dataset, out_dir / "data_raw")

# Build weighted dataset
wt_dataset = None
if weighted_layers:
    wt_dataset = build_multiplex_dataset(
        weighted_layers, ccode_to_idx, idx_to_ccode, nodes_df,
        year_range=(args.start_year, args.end_year),
    )
    save_dataset(wt_dataset, out_dir / "data_weighted")

# Shared model config
model_config = MultiplexGNNConfig(
    num_layers=len(raw_dataset.layer_names),
    in_dim=len(raw_dataset.layer_names),
    hidden_dim=args.hidden_dim,
    emb_dim=args.emb_dim,
    temporal_hidden_dim=args.hidden_dim,
    seq_len=args.seq_len,
)

train_config = TrainingConfig(
    learning_rate=args.lr,
    num_epochs=args.epochs,
    seq_len=args.seq_len,
    print_every=max(1, args.epochs // 10),
)

# ============================================================
# 2. Train Model Set 1 (raw)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Training Model Set 1 (Raw Layers)")
print("=" * 60)

torch.manual_seed(42)
np.random.seed(42)
raw_train_config = TrainingConfig(
    **{**train_config.__dict__, "save_dir": str(out_dir / "model_raw")}
)
raw_result = train_model(raw_dataset, model_config, raw_train_config, device)
print(f"Raw model final loss: {raw_result.loss_history[-1]:.6f}")
print(f"Raw layer weights: {raw_result.layer_weights}")

# ============================================================
# 3. Train Model Set 2 (weighted)
# ============================================================
wt_result = None
if wt_dataset and not args.skip_weighted:
    print("\n" + "=" * 60)
    print("STEP 3: Training Model Set 2 (Strength-Weighted Layers)")
    print("=" * 60)

    wt_model_config = MultiplexGNNConfig(
        num_layers=len(wt_dataset.layer_names),
        in_dim=len(wt_dataset.layer_names),
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        temporal_hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
    )
    torch.manual_seed(42)
    np.random.seed(42)
    wt_train_config = TrainingConfig(
        **{**train_config.__dict__, "save_dir": str(out_dir / "model_weighted")}
    )
    wt_result = train_model(wt_dataset, wt_model_config, wt_train_config, device)
    print(f"Weighted model final loss: {wt_result.loss_history[-1]:.6f}")
    print(f"Weighted layer weights: {wt_result.layer_weights}")

# ============================================================
# 4. Walk-forward backtesting
# ============================================================
if not args.skip_backtest:
    print("\n" + "=" * 60)
    print("STEP 4: Walk-Forward Backtesting")
    print("=" * 60)

    if args.backtest_splits is None:
        # Default: split every 10 years starting from start_year + 20
        split_start = args.start_year + 20
        split_end = args.end_year - 5
        splits = list(range(split_start, split_end + 1, 10))
    else:
        splits = args.backtest_splits

    print(f"Split years: {splits}")
    backtest = walk_forward_backtest(
        dataset=raw_dataset,
        split_years=splits,
        model_config=model_config,
        train_config=TrainingConfig(
            learning_rate=args.lr,
            num_epochs=min(args.epochs, 100),
            seq_len=args.seq_len,
            print_every=25,
        ),
        test_horizon=5,
        seq_len=args.seq_len,
        device=device,
        save_dir=str(out_dir / "backtest"),
    )
    print(f"\nBacktest summary:")
    print(backtest.summary[["fold_id", "train_end", "test_year",
                            "test_emb_mse", "test_link_auc"]].to_string(index=False))

# ============================================================
# 5. Multi-seed robustness
# ============================================================
if not args.skip_multiseed:
    print("\n" + "=" * 60)
    print("STEP 5: Multi-Seed Robustness")
    print("=" * 60)

    seeds = list(range(args.n_seeds))
    seed_results = multi_seed_evaluation(
        dataset=raw_dataset,
        seeds=seeds,
        model_config=model_config,
        train_config=TrainingConfig(
            learning_rate=args.lr,
            num_epochs=min(args.epochs, 100),
            seq_len=args.seq_len,
            print_every=50,
        ),
        device=device,
        save_dir=str(out_dir / "multiseed"),
    )

# ============================================================
# 6. Baseline comparisons
# ============================================================
if not args.skip_baselines:
    print("\n" + "=" * 60)
    print("STEP 6: Baseline Comparisons")
    print("=" * 60)

    torch.manual_seed(42)
    baseline_summary = compare_baselines(
        dataset=raw_dataset,
        full_model_result=raw_result,
        model_config=model_config,
        seq_len=args.seq_len,
        num_epochs=min(args.epochs, 100),
        device=device,
        save_dir=str(out_dir / "baselines"),
    )

# ============================================================
# 7. Forecasting
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Forecasting")
print("=" * 60)

raw_forecasts = forecast_embeddings(
    raw_result.model, raw_dataset, args.forecast_steps, args.seq_len, device,
)
torch.save(raw_forecasts, out_dir / "forecast_raw.pt")
print(f"Raw forecasts: {list(raw_forecasts.keys())[:5]}...{list(raw_forecasts.keys())[-1]}")

if wt_result and wt_dataset:
    wt_forecasts = forecast_embeddings(
        wt_result.model, wt_dataset, args.forecast_steps, args.seq_len, device,
    )
    torch.save(wt_forecasts, out_dir / "forecast_weighted.pt")

# ============================================================
# 8. Counterfactual analysis
# ============================================================
if not args.skip_counterfactual:
    print("\n" + "=" * 60)
    print("STEP 8: Counterfactual Analysis")
    print("=" * 60)

    partners = None
    if args.cf_max_partners > 0:
        all_partners = sorted([cc for cc in ccode_to_idx.keys() if cc not in (USA_CCODE, CHN_CCODE)])
        if len(all_partners) > args.cf_max_partners:
            step = len(all_partners) // args.cf_max_partners
            partners = all_partners[::step][:args.cf_max_partners]
        else:
            partners = all_partners

    cf_dir = out_dir / "counterfactual"
    cf_dir.mkdir(parents=True, exist_ok=True)

    for focal_label, focal_ccode in [("usa", USA_CCODE), ("china", CHN_CCODE)]:
        if focal_ccode not in ccode_to_idx:
            print(f"  Skipping {focal_label}: not in dataset")
            continue

        # Raw model counterfactuals
        print(f"\n  {focal_label.upper()} counterfactuals (raw model)...")
        cf = batch_counterfactual_analysis(
            raw_result.model, raw_dataset, focal_ccode,
            partner_ccodes=partners, seq_len=args.seq_len, device=device,
        )
        if cf:
            df = results_to_dataframe(cf)
            df.to_csv(cf_dir / f"cf_{focal_label}_raw.csv", index=False)
            summarize_by_layer(df).to_csv(cf_dir / f"cf_{focal_label}_raw_by_layer.csv", index=False)
            top_interventions(df, 30).to_csv(cf_dir / f"cf_{focal_label}_raw_top30.csv", index=False)
            print(f"    {len(cf)} simulations, saved to {cf_dir}")

        # Weighted model counterfactuals
        if wt_result and wt_dataset:
            print(f"  {focal_label.upper()} counterfactuals (weighted model)...")
            cf_wt = batch_counterfactual_analysis(
                wt_result.model, wt_dataset, focal_ccode,
                partner_ccodes=partners, seq_len=args.seq_len, device=device,
            )
            if cf_wt:
                df_wt = results_to_dataframe(cf_wt)
                df_wt.to_csv(cf_dir / f"cf_{focal_label}_weighted.csv", index=False)
                summarize_by_layer(df_wt).to_csv(cf_dir / f"cf_{focal_label}_weighted_by_layer.csv", index=False)
                top_interventions(df_wt, 30).to_csv(cf_dir / f"cf_{focal_label}_weighted_top30.csv", index=False)

# ============================================================
# 9. Isolation analysis: paired USA vs China edge effects
# ============================================================
if not args.skip_isolation:
    print("\n" + "=" * 60)
    print("STEP 9: Isolation Analysis (USA vs China)")
    print("=" * 60)

    iso_partners = partners  # reuse partner list from counterfactual step
    if iso_partners is None and args.cf_max_partners > 0:
        all_partners = sorted([cc for cc in ccode_to_idx.keys() if cc not in (USA_CCODE, CHN_CCODE)])
        if len(all_partners) > args.cf_max_partners:
            step = len(all_partners) // args.cf_max_partners
            iso_partners = all_partners[::step][:args.cf_max_partners]

    # Example multi-edge scenarios (policy-relevant)
    multi_scenarios = {
        "India leaves US orbit (trade + alliance)": [
            (750, "trade", "remove"),
            (750, "defensive_alliances", "remove"),
        ],
        "Brazil joins China (trade + IGO)": [
            (140, "trade", "add"),
            (140, "igo", "add"),
        ],
        "US loses Turkey alliance": [
            (640, "defensive_alliances", "remove"),
            (640, "offensive_alliances", "remove"),
        ],
        "China gains South Korea trade": [
            (732, "trade", "add"),
        ],
    }

    # Raw model
    run_isolation_analysis(
        model=raw_result.model,
        dataset=raw_dataset,
        out_dir=out_dir / "isolation_raw",
        partner_ccodes=iso_partners,
        seq_len=args.seq_len,
        multi_edge_scenarios=multi_scenarios,
        device=device,
    )

    # Weighted model
    if wt_result and wt_dataset:
        run_isolation_analysis(
            model=wt_result.model,
            dataset=wt_dataset,
            out_dir=out_dir / "isolation_weighted",
            partner_ccodes=iso_partners,
            seq_len=args.seq_len,
            multi_edge_scenarios=multi_scenarios,
            device=device,
        )

# ============================================================
# 10. Generate all tables and figure data
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Generating Tables and Figure Data")
print("=" * 60)

tables = generate_all_tables(
    raw_result=raw_result,
    weighted_result=wt_result,
    dataset=raw_dataset,
    out_dir=out_dir / "tables",
)

# ============================================================
# Done
# ============================================================
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print(f"All outputs saved to: {out_dir}")
print("=" * 60)
print("\nOutput structure:")
print(f"  {out_dir}/model_raw/          - Model Set 1 weights and diagnostics")
if wt_result:
    print(f"  {out_dir}/model_weighted/     - Model Set 2 weights and diagnostics")
if not args.skip_backtest:
    print(f"  {out_dir}/backtest/           - Walk-forward backtesting results")
if not args.skip_multiseed:
    print(f"  {out_dir}/multiseed/          - Multi-seed robustness results")
if not args.skip_baselines:
    print(f"  {out_dir}/baselines/          - Baseline model comparisons")
if not args.skip_counterfactual:
    print(f"  {out_dir}/counterfactual/     - Counterfactual simulation results")
if not args.skip_isolation:
    print(f"  {out_dir}/isolation_raw/      - USA vs China isolation analysis + figures")
print(f"  {out_dir}/tables/             - All manuscript tables and figure data")
print(f"  {out_dir}/forecast_raw.pt     - Forecast embeddings")
