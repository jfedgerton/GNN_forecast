#!/usr/bin/env python3
"""Generate all paper figures from the output CSVs.

Reads from outputs/ subdirectories produced by the diagnostic, intervention,
SBM-validation, and forecast scripts. Writes PNGs to outputs/figures/.

Per user preferences (paper_outline.md):
  - x-axis tick text not angled (broken into two lines and aligned to plot)
  - facet_wrap text NOT moved to left side; angles allowed on facet labels
  - seed = 123

Figures generated (each saved with high DPI for paper inclusion):
  fig_diagnostic_training.png        Training-curve (loss + AUC over epochs)
  fig_planted_feature_recovery.png   Per-hop displacement, planted vs. null
  fig_planted_edge_recovery.png      Wedge-rank distribution, planted vs. null
  fig_regime_shock_cascade.png       Per-focal hop-decay bars (8 scenarios)
  fig_top_wedge_edges.png            Top-K |wedge| edges per focal-pair
  fig_layer_wedge_heatmap.png        Mean wedge by (layer, operation) per focal-pair
  fig_focal_scatter_USA_CHN.png      USA-vs-CHN delta scatter (all perturbations)
  fig_joint_interaction.png          Joint vs. additive bar (interaction term)
  fig_forecast_baseline.png          2017-2040 baseline trajectories, 4 focals
  fig_forecast_scenarios.png         Baseline vs. counterfactual per scenario

Usage:
    PYTHONPATH=src python scripts/make_paper_figures.py \\
        --outputs-dir outputs --figures-dir outputs/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 123
np.random.seed(SEED)

DPI = 200


# ---------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------

def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _two_line(label: str) -> str:
    """Break a long x-tick label into two lines on the first underscore
    or whitespace past index 6 (per user pref: no angled x-axis text)."""
    if len(label) <= 10:
        return label
    for i in range(6, len(label) - 4):
        if label[i] in (" ", "_", "-", "/"):
            return label[:i] + "\n" + label[i+1:]
    mid = len(label) // 2
    return label[:mid] + "\n" + label[mid:]


# ---------------------------------------------------------------
# Individual figure functions
# ---------------------------------------------------------------

def fig_diagnostic_training(outputs_dir: Path, figures_dir: Path) -> None:
    """Encoder loss + AUC over epochs (from diagnostic_v3_results.pt history)."""
    summary_path = outputs_dir / "diagnostic_v3" / "diagnostic_v3_summary.csv"
    if not summary_path.exists():
        print(f"  skip diagnostic_training: {summary_path} not found")
        return
    summary = pd.read_csv(summary_path).iloc[0]
    # Pull histories from results.pt
    import torch
    res_path = outputs_dir / "diagnostic_v3" / "diagnostic_v3_results.pt"
    if not res_path.exists():
        print(f"  skip diagnostic_training: {res_path} not found")
        return
    payload = torch.load(res_path, map_location="cpu", weights_only=False)
    res = payload["result"]
    epochs = list(range(1, len(res.loss_history) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, res.loss_history, color="C0")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("InfoNCE loss")
    ax1.set_title("Training loss")

    ax2.plot(epochs, res.auc_history, color="C1")
    ax2.axhline(summary["best_link_auc"], color="grey", ls="--", lw=0.8,
                label=f"best AUC = {summary['best_link_auc']:.3f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Stratified-negatives AUC")
    ax2.set_title(f"Validation AUC (best at epoch {int(summary['best_epoch'])})")
    ax2.legend(loc="lower right")
    ax2.set_ylim(0.4, 1.0)
    fig.suptitle("R-GCN diagnostic v3 training (6-layer multiplex)")
    _save(fig, figures_dir / "fig_diagnostic_training.png")


def fig_planted_feature_recovery(outputs_dir: Path, figures_dir: Path) -> None:
    """Per-hop displacement: planted vs null, mean over replicates."""
    path = outputs_dir / "regime_shock_simulation" / "recovery_summary.csv"
    if not path.exists():
        print(f"  skip planted_feature_recovery: {path} not found")
        return
    df = pd.read_csv(path)
    # Strip any leading/trailing whitespace from scenario column (defensive)
    df["scenario"] = df["scenario"].astype(str).str.strip()
    hops = ["hop_0", "hop_1", "hop_2", "hop_3plus"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    x = np.arange(len(hops))
    for i, scenario in enumerate(["planted", "null"]):
        sub = df[df["scenario"] == scenario]
        if sub.empty:
            print(f"  skip planted_feature_recovery: no rows for scenario={scenario!r} "
                  f"(have {df['scenario'].unique().tolist()})")
            plt.close(fig)
            return
        row = sub.iloc[0]
        vals = [row[f"mean_{h}_displacement"] for h in hops]
        ax.bar(x + (i - 0.5) * width, vals, width,
               label=f"{scenario} (n={int(row['n_replicates'])})",
               color="C2" if scenario == "planted" else "C3")
    ax.set_xticks(x)
    ax.set_xticklabels(["hop-0\n(focal)", "hop-1\n(neighbors)",
                        "hop-2\n(2nd ring)", "hop-3+\n(beyond)"])
    ax.set_ylabel("Mean L2 embedding displacement")
    ax.set_title("Planted-feature-shock SBM recovery: monotonic decay in planted, "
                 "zero in null")
    ax.legend(loc="upper right")
    _save(fig, figures_dir / "fig_planted_feature_recovery.png")


def fig_planted_edge_recovery(outputs_dir: Path, figures_dir: Path) -> None:
    """Wedge-rank distribution for planted vs null."""
    path = outputs_dir / "regime_shock_simulation" / "edge_recovery_per_replicate.csv"
    if not path.exists():
        print(f"  skip planted_edge_recovery: {path} not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for scenario, color in [("planted", "C2"), ("null", "C3")]:
        sub = df[df["scenario"] == scenario]
        ax.hist(sub["rank_centroid"], bins=20, alpha=0.6, label=scenario, color=color)
    ax.set_xlabel("Rank of planted edge in wedge-magnitude ordering")
    ax.set_ylabel("Replicate count")
    ax.set_title("Planted-edge SBM recovery: planted ranks at top, null is uniform")
    ax.legend(loc="upper right")
    _save(fig, figures_dir / "fig_planted_edge_recovery.png")


def fig_regime_shock_cascade(outputs_dir: Path, figures_dir: Path) -> None:
    """Per-focal hop-decay bars across the 8 scenarios."""
    path = outputs_dir / "regime_shock" / "regime_shock_summary.csv"
    if not path.exists():
        print(f"  skip regime_shock_cascade: {path} not found")
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    hops = ["hop_0", "hop_1", "hop_2", "hop_3plus"]
    hop_labels = ["hop-0", "hop-1", "hop-2", "hop-3+"]
    for ax, (_, row) in zip(axes.flatten(), df.iterrows()):
        vals = [row[f"{h}_displacement"] for h in hops]
        ax.bar(hop_labels, vals, color="C0")
        ax.set_title(_two_line(row["scenario"]), fontsize=10)
        ax.set_ylabel("displacement")
    fig.suptitle("Regime-shock cascades: per-hop embedding displacement, 4 focals × {Polity, CINC}")
    _save(fig, figures_dir / "fig_regime_shock_cascade.png")


def fig_top_wedge_edges(outputs_dir: Path, figures_dir: Path) -> None:
    """Top-K |wedge| edges per focal pair."""
    path = outputs_dir / "multi_focal_edge" / "top_k_wedges.csv"
    if not path.exists():
        print(f"  skip top_wedge_edges: {path} not found")
        return
    df = pd.read_csv(path)
    pairs = df.groupby(["focal_a_ccode", "focal_b_ccode"]).size().reset_index(name="n")
    n_pairs = min(4, len(pairs))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (_, p) in zip(axes.flatten(), pairs.head(n_pairs).iterrows()):
        sub = df[(df["focal_a_ccode"] == p["focal_a_ccode"])
                 & (df["focal_b_ccode"] == p["focal_b_ccode"])]
        sub = sub.head(15).copy()  # top-15 only
        sub["label"] = (sub["partner_ccode"].astype(str)
                        + "/" + sub["layer_name"]
                        + "/" + sub["operation"])
        sub = sub.iloc[::-1]  # so top is at top
        ax.barh(sub["label"], sub["wedge"],
                color=["C0" if w >= 0 else "C3" for w in sub["wedge"]])
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(f"{int(p['focal_a_ccode'])} vs {int(p['focal_b_ccode'])}",
                     fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Top-15 absolute wedge edges per focal pair (centroid metric)")
    _save(fig, figures_dir / "fig_top_wedge_edges.png")


def fig_layer_wedge_heatmap(outputs_dir: Path, figures_dir: Path) -> None:
    """Mean wedge by (layer, operation) per focal pair."""
    path = outputs_dir / "multi_focal_edge" / "wedges_centroid.csv"
    if not path.exists():
        print(f"  skip layer_wedge_heatmap: {path} not found")
        return
    df = pd.read_csv(path)
    df["pair"] = (df["focal_a_ccode"].astype(str) + "→"
                  + df["focal_b_ccode"].astype(str))
    df["layer_op"] = df["layer_name"] + " (" + df["operation"] + ")"
    piv = df.groupby(["pair", "layer_op"])["wedge"].mean().unstack()
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r",
                   vmin=-piv.abs().max().max(), vmax=piv.abs().max().max())
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([_two_line(c) for c in piv.columns], fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title("Mean wedge by (layer, operation) per focal pair")
    fig.colorbar(im, ax=ax, label="mean wedge (focal_a − focal_b)")
    _save(fig, figures_dir / "fig_layer_wedge_heatmap.png")


def fig_focal_scatter_usa_chn(outputs_dir: Path, figures_dir: Path) -> None:
    """USA Δ vs CHN Δ scatter for all (partner, layer, op) perturbations."""
    path = outputs_dir / "multi_focal_edge" / "sweep_long.csv"
    if not path.exists():
        print(f"  skip focal_scatter_usa_chn: {path} not found")
        return
    df = pd.read_csv(path)
    usa = df[df["focal_ccode"] == 2][["partner_ccode", "layer_name", "operation",
                                      "delta_centroid"]].rename(
        columns={"delta_centroid": "delta_USA"})
    chn = df[df["focal_ccode"] == 710][["partner_ccode", "layer_name", "operation",
                                        "delta_centroid"]].rename(
        columns={"delta_centroid": "delta_CHN"})
    paired = usa.merge(chn, on=["partner_ccode", "layer_name", "operation"])
    if paired.empty:
        print("  skip focal_scatter_usa_chn: no paired USA/CHN rows")
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(paired["delta_USA"], paired["delta_CHN"], alpha=0.5,
                    c=paired["operation"].map({"add": "C0", "remove": "C3"}),
                    s=12)
    lim = max(paired["delta_USA"].abs().max(), paired["delta_CHN"].abs().max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "--", color="grey", lw=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Δ USA centroid distance")
    ax.set_ylabel("Δ China centroid distance")
    ax.set_title("USA vs China centroid-distance shifts (each point = one edge perturbation)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    handles = [plt.scatter([], [], color="C0", label="add"),
               plt.scatter([], [], color="C3", label="remove")]
    ax.legend(handles=handles, loc="lower right")
    _save(fig, figures_dir / "fig_focal_scatter_USA_CHN.png")


def fig_joint_interaction(outputs_dir: Path, figures_dir: Path) -> None:
    """Joint vs additive bar with interaction-term annotation."""
    path = outputs_dir / "joint_intervention" / "joint_intervention_summary.csv"
    if not path.exists():
        print(f"  skip joint_interaction: {path} not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["delta_feature_only"], width, label="feature only", color="C0")
    ax.bar(x, df["delta_edge_only"], width, label="edge only", color="C1")
    ax.bar(x + width, df["delta_joint"], width, label="joint", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels([_two_line(s) for s in df["scenario"]], fontsize=8)
    ax.set_ylabel("Δ centroid distance")
    ax.set_title("Joint Polity+edge interventions: feature, edge, and joint deltas (per scenario)")
    ax.legend()
    # Annotate each scenario with the interaction term
    for i, row in df.iterrows():
        ax.annotate(f"int={row['interaction_term']:+.3f}",
                    xy=(i, row["delta_joint"]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, color="C2")
    _save(fig, figures_dir / "fig_joint_interaction.png")


def fig_forecast_baseline(outputs_dir: Path, figures_dir: Path) -> None:
    """2017-2040 baseline trajectories for the 4 focals."""
    path = outputs_dir / "forecast" / "baseline_trajectories.csv"
    if not path.exists():
        print(f"  skip forecast_baseline: {path} not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    last_obs_year = 2016
    for fc, sub in df.groupby("focal_name"):
        sub = sub.sort_values("year")
        ax.plot(sub["year"], sub["focal_centroid_dist"], label=fc, lw=1.5)
    ax.axvline(last_obs_year, color="grey", ls="--", lw=0.8,
               label=f"end of observed data ({last_obs_year})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Focal centroid distance")
    ax.set_title("Baseline embedding trajectories, 1948-2040 (GRU rollout post-2016)")
    ax.legend(loc="upper left")
    _save(fig, figures_dir / "fig_forecast_baseline.png")


def fig_layer_coverage(outputs_dir: Path, figures_dir: Path) -> None:
    """Layer-coverage matrix (years × layers, shaded where observed) for §3.

    Uses the COW-filtered layer CSVs to determine which (year, layer)
    cells are observed.
    """
    data_dir = outputs_dir.parent / "data" / "processed"
    layers = {
        "defensive_alliances": "layer_alliances_defensive_offensive_undirected.csv",
        "offensive_alliances": "layer_alliances_defensive_offensive_undirected.csv",
        "dca":                "layer_dca_undirected.csv",
        "fta":                "layer_fta_undirected.csv",
        "pta_services":       "layer_pta_services_undirected.csv",
        "cu":                 "layer_cu_undirected.csv",
    }
    coverage_rows = []
    for ln, fname in layers.items():
        path = data_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["year"])
        years = sorted(df["year"].unique())
        for y in years:
            coverage_rows.append({"layer": ln, "year": int(y)})
    if not coverage_rows:
        print("  skip layer_coverage: no layer CSVs found in data/processed")
        return
    cov = pd.DataFrame(coverage_rows)
    pivot = (
        cov.assign(observed=1)
        .pivot_table(index="layer", columns="year", values="observed", fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(13, 3.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Greens",
                   interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    cols = list(pivot.columns)
    decade_ticks = [i for i, y in enumerate(cols) if y % 10 == 0]
    ax.set_xticks(decade_ticks)
    ax.set_xticklabels([cols[i] for i in decade_ticks])
    ax.set_xlabel("Year")
    ax.set_title("Layer coverage matrix (green = observed)")
    _save(fig, figures_dir / "fig_layer_coverage.png")


def fig_recovery_distribution(outputs_dir: Path, figures_dir: Path) -> None:
    """Two-panel side-by-side wedge-rank distribution: planted vs null.

    Used in §5.2 to show that the planted-edge SBM produces a clean
    rank distribution at the top while the null is uniform.
    """
    path = outputs_dir / "regime_shock_simulation" / "edge_recovery_per_replicate.csv"
    if not path.exists():
        print(f"  skip recovery_distribution: {path} not found")
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    n_cands = int(df["n_candidates"].mean()) if "n_candidates" in df.columns else 174
    bins = np.arange(0, n_cands + 10, 10)
    for ax, scenario, color in [(axes[0], "planted", "C2"), (axes[1], "null", "C3")]:
        sub = df[df["scenario"] == scenario]
        ax.hist(sub["rank_centroid"], bins=bins, color=color, edgecolor="white")
        ax.axvline(10, color="grey", ls="--", lw=0.7,
                   label=f"top-10 cutoff ({(sub['rank_centroid'] <= 10).mean()*100:.0f}% of replicates)")
        ax.set_title(scenario)
        ax.set_xlabel("Wedge-magnitude rank of planted edge")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Replicate count")
    fig.suptitle("Planted-edge SBM recovery: planted clusters at the top, null is uniform")
    _save(fig, figures_dir / "fig_recovery_distribution.png")


def fig_edge_forecast_scenarios(outputs_dir: Path, figures_dir: Path) -> None:
    """Baseline vs each edge-counterfactual trajectory, faceted by focal."""
    base_path = outputs_dir / "forecast" / "baseline_trajectories.csv"
    cf_path = outputs_dir / "forecast" / "edge_scenario_trajectories.csv"
    if not (base_path.exists() and cf_path.exists()):
        print(f"  skip edge_forecast_scenarios: missing {base_path} or {cf_path}")
        return
    base = pd.read_csv(base_path)
    cf = pd.read_csv(cf_path)
    focals = sorted(cf["focal_name"].unique())
    n = len(focals)
    rows_n = (n + 1) // 2
    fig, axes = plt.subplots(rows_n, 2, figsize=(13, 3.5 * rows_n), sharex=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, fc in zip(axes, focals):
        b = base[base["focal_name"] == fc].sort_values("year")
        ax.plot(b["year"], b["focal_centroid_dist"], "k-", lw=2, label="baseline")
        for scen, sub in cf[cf["focal_name"] == fc].groupby("scenario"):
            sub = sub.sort_values("year")
            ax.plot(sub["year"], sub["focal_centroid_dist"], "--", lw=1,
                    label=scen, alpha=0.85)
        ax.axvline(2016, color="grey", ls=":", lw=0.6)
        ax.set_title(fc)
        ax.set_xlabel("Year"); ax.set_ylabel("Centroid distance")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Baseline vs edge-counterfactual trajectories (1948-2040)")
    _save(fig, figures_dir / "fig_edge_forecast_scenarios.png")


def fig_forecast_scenarios(outputs_dir: Path, figures_dir: Path) -> None:
    """Baseline vs each counterfactual trajectory, faceted by focal."""
    base_path = outputs_dir / "forecast" / "baseline_trajectories.csv"
    cf_path = outputs_dir / "forecast" / "scenario_trajectories.csv"
    if not (base_path.exists() and cf_path.exists()):
        print(f"  skip forecast_scenarios: missing {base_path} or {cf_path}")
        return
    base = pd.read_csv(base_path)
    cf = pd.read_csv(cf_path)
    focals = sorted(cf["focal_name"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, fc in zip(axes.flatten(), focals):
        b = base[base["focal_name"] == fc].sort_values("year")
        ax.plot(b["year"], b["focal_centroid_dist"], "k-", lw=2, label="baseline")
        for scen, sub in cf[cf["focal_name"] == fc].groupby("scenario"):
            sub = sub.sort_values("year")
            ax.plot(sub["year"], sub["focal_centroid_dist"], "--", lw=1,
                    label=scen, alpha=0.8)
        ax.axvline(2016, color="grey", ls=":", lw=0.6)
        ax.set_title(fc)
        ax.set_xlabel("Year"); ax.set_ylabel("Centroid distance")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Baseline vs counterfactual trajectories (2010-2040), 4 focals")
    _save(fig, figures_dir / "fig_forecast_scenarios.png")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def fig_horizon_decay(outputs_dir: Path, figures_dir: Path) -> None:
    """Figure 5.1 — walk-forward horizon decay (3-panel: MSE, AUC, centroid drift)."""
    summary_path = outputs_dir / "walk_forward" / "walk_forward_summary.csv"
    full_path = outputs_dir / "walk_forward" / "walk_forward_backtest.csv"
    if not summary_path.exists():
        print(f"  skip horizon_decay: {summary_path} not found")
        return
    summary = pd.read_csv(summary_path)
    full = pd.read_csv(full_path) if full_path.exists() else None

    # Expected columns in summary: horizon, mean_mse, mean_auc, mean_centroid_drift, n_splits
    horizons = sorted(summary["horizon"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    panels = [
        ("mean_mse",            "Mean embedding MSE",        "lower right"),
        ("mean_auc",            "Mean link-prediction AUC",  "upper right"),
        ("mean_centroid_drift", "Mean centroid drift",       "lower right"),
    ]
    for ax, (col, ylab, _) in zip(axes, panels):
        if col not in summary.columns:
            ax.text(0.5, 0.5, f"{col}\nnot in summary", ha="center", va="center")
            ax.set_axis_off(); continue
        means = summary.set_index("horizon").loc[horizons, col].values
        ax.plot(horizons, means, marker="o", linewidth=2, color="#1f4e79")
        # If we have the per-split file, overlay points + min/max envelope
        if full is not None and col.replace("mean_", "") in full.columns:
            raw_col = col.replace("mean_", "")
            for h in horizons:
                vals = full[full["horizon"] == h][raw_col].dropna().values
                if len(vals):
                    ax.scatter([h] * len(vals), vals, s=18, alpha=0.4,
                               color="#888888", zorder=1)
        ax.set_xlabel("Forecast horizon\n(years)")
        ax.set_ylabel(ylab)
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Walk-forward backtest: horizon decay")
    _save(fig, figures_dir / "fig_horizon_decay.png")


def fig_ame_vs_gnn(outputs_dir: Path, figures_dir: Path) -> None:
    """Figure 5.3 — AME-vs-GNN R² alignment by (layer, year)."""
    align_path = outputs_dir / "ame_baseline" / "ame_vs_gnn_alignment.csv"
    if not align_path.exists():
        print(f"  skip ame_vs_gnn: {align_path} not found")
        return
    df = pd.read_csv(align_path)
    if df.empty:
        print(f"  skip ame_vs_gnn: file is empty"); return

    # Layer order: alliance first (highest R²), then trade-cooperation, then sparse
    layer_order = ["defensive_alliances", "fta", "pta_services", "cu", "dca"]
    layers = [l for l in layer_order if l in df["layer"].unique()]
    years = sorted(df["year"].unique())
    n_layers, n_years = len(layers), len(years)

    fig, ax = plt.subplots(figsize=(11, 5))
    bar_w = 0.8 / max(n_years, 1)
    x = np.arange(n_layers)
    palette = ["#1f4e79", "#2e7d32", "#c95f00", "#7b1fa2"]
    for i, yr in enumerate(years):
        r2_vals = []
        for layer in layers:
            sub = df[(df["layer"] == layer) & (df["year"] == yr)]
            r2_vals.append(float(sub["ame_vs_gnn_r2_mean"].iloc[0]) if len(sub) else np.nan)
        ax.bar(x + i * bar_w - 0.4 + bar_w / 2, r2_vals, bar_w,
               label=str(yr), color=palette[i % len(palette)], edgecolor="white")

    # Per-layer mean as black horizontal tick
    for j, layer in enumerate(layers):
        layer_mean = df[df["layer"] == layer]["ame_vs_gnn_r2_mean"].mean()
        ax.hlines(layer_mean, x[j] - 0.4, x[j] + 0.4,
                  colors="black", linestyles="dashed", linewidth=1.5,
                  label="Layer mean" if j == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels([_two_line(l) for l in layers])
    ax.set_ylabel(r"$R^{2}$ (AME 2D vs GNN PCA-2D)")
    ax.set_xlabel("Layer")
    ax.set_ylim(0, max(0.7, df["ame_vs_gnn_r2_mean"].max() * 1.15))
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_title("AME bilinear latents vs R-GCN embeddings — per-cell alignment")
    ax.legend(title="Year", loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, figures_dir / "fig_ame_vs_gnn.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--figures-dir", default="outputs/figures")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures in {figures_dir}")

    figure_fns = [
        fig_layer_coverage,            # §3 layer-coverage matrix
        fig_diagnostic_training,
        fig_horizon_decay,             # NEW §5.1 walk-forward horizon decay
        fig_planted_edge_recovery,
        fig_recovery_distribution,     # §5.2 planted-vs-null distributions
        fig_top_wedge_edges,           # §5.2 top significant wedges (post-filter)
        fig_ame_vs_gnn,                # NEW §5.3 AME-vs-GNN R² alignment
        fig_layer_wedge_heatmap,
        fig_focal_scatter_usa_chn,
        fig_forecast_baseline,
        fig_edge_forecast_scenarios,   # §6 edge-counterfactual trajectories
        # Archived (paper #2): fig_planted_feature_recovery, fig_regime_shock_cascade,
        # fig_joint_interaction, fig_forecast_scenarios
    ]
    n_ok, n_fail = 0, 0
    import traceback
    for fn in figure_fns:
        try:
            fn(outputs_dir, figures_dir)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"  ERROR in {fn.__name__}: {e}")
            traceback.print_exc(limit=2)

    print(f"\nDone. {n_ok} figures generated, {n_fail} errored.")


if __name__ == "__main__":
    main()
