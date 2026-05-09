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
    hops = ["hop_0", "hop_1", "hop_2", "hop_3plus"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    x = np.arange(len(hops))
    for i, scenario in enumerate(["planted", "null"]):
        row = df[df["scenario"] == scenario].iloc[0]
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

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--figures-dir", default="outputs/figures")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures in {figures_dir}")

    fig_diagnostic_training(outputs_dir, figures_dir)
    fig_planted_feature_recovery(outputs_dir, figures_dir)
    fig_planted_edge_recovery(outputs_dir, figures_dir)
    fig_regime_shock_cascade(outputs_dir, figures_dir)
    fig_top_wedge_edges(outputs_dir, figures_dir)
    fig_layer_wedge_heatmap(outputs_dir, figures_dir)
    fig_focal_scatter_usa_chn(outputs_dir, figures_dir)
    fig_joint_interaction(outputs_dir, figures_dir)
    fig_forecast_baseline(outputs_dir, figures_dir)
    fig_forecast_scenarios(outputs_dir, figures_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
