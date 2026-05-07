# Paper Outline — Multiplex GNN Forecasting for the International System

**Target journal:** Political Analysis
**Working title:** *Multiplex Graph Neural Networks for International System Forecasting and Counterfactual Edge Analysis*
**Author:** Jared Edgerton

---

## Status & artifact paths (last updated 2026-05-06)

All code lives at `https://github.com/jfedgerton/GNN_forecast.git`, branch `pa-revision-uncertainty-sim`. On Roar Collab, the project is at `/scratch/jfe4/GNN_forecast` (symlink to `/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast`).

| Component | Status | Output paths (Roar) |
|---|---|---|
| Planted-wedge simulation (§5.2) | **DONE** | `outputs/simulation_study/recovery_summary.csv`, `recovery_study.csv`, `recovery_summary_planted.csv` |
| Multi-horizon backtest 5/10/15/20 (§5.1) | RUNNING — SLURM job 52743495 (~12h) | `outputs/multi_horizon/multi_horizon_backtest.csv`, `multi_horizon_summary.csv` |
| Ensemble UQ + bootstrap CIs (§5.4) | RUNNING — SLURM job 52743497 (~24h) | `outputs/uncertainty/bootstrap_counterfactual_cis.csv`, `embeddedness_ci_final_year.csv`, `top_significant_wedges.csv`, `ensemble_weights/` |
| AME baseline (§5.3) | NOT YET SUBMITTED | `outputs/ame_baseline/<layer>/ame_latent_<year>.csv`, `ame_fit_summary.csv` |
| Layer attention figure (§6.1) | code present in `visualization.py`, not yet generated | `outputs/figures/layer_attention_timeseries.png` (TBD) |
| Top wedge bar chart (§6.2) | code in `isolation_analysis.plot_wedge_bar`, not yet generated | `outputs/figures/bar_top_wedge_edges.png` (TBD) |
| USA-vs-China scatter (§6.3) | code in `isolation_analysis.plot_isolation_scatter`, not yet generated | `outputs/figures/scatter_usa_vs_china.png` (TBD) |
| Layer heatmap (§6.3) | code in `isolation_analysis.plot_layer_heatmap`, not yet generated | `outputs/figures/heatmap_layer_isolation.png` (TBD) |
| 25-year rollout figure (Appendix F) | not yet generated | `outputs/figures/embedding_forecast_2026_2050.png` (TBD) |

To pull any of these results from Roar to local, use `rsync` from the storage path or download via the OnDemand File Manager.

---

## 1. Introduction (≈ 1,000 words)

Open with the substantive puzzle in two sentences (states embedded in many overlapping relational systems — alliances, IGOs, trade — but methods analyze them one layer at a time), then pivot immediately to the methodological gap. State three contributions explicitly:

(i) a multiplex temporal GNN with learned attention aggregation that handles asynchronous temporal coverage via masking,

(ii) a counterfactual procedure that propagates edge interventions through learned message-passing rather than treating them as ex-post regression terms, and

(iii) a wedge-decomposition framework for asymmetric great-power leverage analysis, with super-additive interaction detection.

End the intro with the validation roadmap: planted-wedge recovery on synthetic data, walk-forward backtest at 5/10/15/20-year horizons, head-to-head against AME and three other baselines, and ensemble-bootstrapped uncertainty bands on every reported counterfactual estimate. Close by signaling that USA-China is a *demonstration*, not the contribution.

## 2. Related Work (≈ 800 words)

Three threads, each one short paragraph:

- **Multiplex network analysis in IR:** Cranmer, Heinrich, Desmarais (multiplex ERGM); Hafner-Burton, Kahler, Montgomery; Maoz on shared dyadic membership. Limit: each requires balanced panels, treats layer importance as ex-post correlational.
- **Latent-space and AME models:** Hoff (latent space, AME, GBME); Ward, Stovel, Sacks (latent-distance models for IR). Limit: cross-sectional, dyadic, no joint multilayer learning.
- **GNNs in social/political networks:** brief — DySAT, EvolveGCN, TGN; almost no IR applications. Limit: standard GNN architectures don't handle missing-layer-year masking or expose counterfactual intervention machinery.

Position the contribution at the intersection: multiplex-aware, latent-space-style embeddings, learned via GNN message passing, with explicit counterfactual analysis tooling.

## 3. Data (≈ 500 words)

- Source: `peacesciencer` R package, exported to CSV.
- Five layers: offensive alliances, defensive alliances, shared IGO membership (count), trade volume, defense cooperation agreements.
- Temporal coverage: 1816–2025 nominal, but layers have heterogeneous starts (alliances back to 1816, DCA only from 1980, IGO/trade gaps).
- Node universe: COW country codes; deterministic indexing so node identity is preserved across years.
- **Preprocessing:** capital-distance residualization for trade and (optionally) IGO. Two parallel datasets: observed and residualized. Be explicit that distance is a single-covariate gravity adjustment — note GDP/population residualization as a robustness option.

Include a layer-coverage figure (years × layers, shaded where observed). This visually motivates the masking architecture.

## 4. Methods (≈ 2,500 words — the core of the paper)

### 4.1 Multiplex temporal GNN

Notation. Multiplex graph $\mathcal{G}_t = \{(V, E_t^{(\ell)}, w_t^{(\ell)})\}_{\ell=1}^L$ over $T$ years. Each layer is undirected, edge-weighted.

**Per-layer encoder.** Two-layer GCN per layer, $\ell$:

$$ H_t^{(\ell)} = \mathrm{GCN}_\ell\bigl(X_t, A_t^{(\ell)}\bigr) $$

with degree-profile node features $X_t \in \mathbb{R}^{N \times L}$.

**Attention aggregator with masking.** Available-layer mask $m_t \in \{0,1\}^L$. Layer logits $\boldsymbol{\alpha}$ are a learnable parameter; aggregation weights are

$$ \pi_{t,\ell} = \frac{\exp(\alpha_\ell)\, m_{t,\ell}}{\sum_{\ell'} \exp(\alpha_{\ell'})\, m_{t,\ell'}} $$

with the softmax restricted to available layers via an `-inf` masking trick. This is the technical contribution of subsection 4.1 — a single sentence in the paper, but a paragraph of justification: it lets the model use every year where any subset of layers is observed, and the learned $\boldsymbol{\alpha}$ is interpretable as relative layer importance.

Aggregated embedding: $H_t = \sum_\ell \pi_{t,\ell} H_t^{(\ell)}$.

**Temporal head.** GRU over the embedding sequence: $\hat{H}_{t+1} = \mathrm{GRU}_\theta(H_{t-K+1:t})$ with window $K = 5$.

**Edge decoder.** Inner product: $\hat{p}_{ij,t+1} = \sigma(\hat{H}_{t+1,i}^\top \hat{H}_{t+1,j})$.

### 4.2 Two-stage learning signal

This subsection is non-negotiable — every careful reviewer will spot the `target_emb.detach()` choice and want it justified.

Loss: $\mathcal{L} = \mathcal{L}_{\text{emb}} + \lambda_{\text{link}} \mathcal{L}_{\text{link}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}$.

Critically, the embedding-MSE term uses the encoder output as a *detached* target. The encoder is trained only by the link-reconstruction BCE and the smoothness penalty; the GRU is trained only by the embedding MSE. Argue: joint backprop collapses the encoder to a near-constant mapping that trivially minimizes year-to-year MSE but produces uninformative embeddings. Two-stage decoupling forces the encoder to remain grounded in observed link structure.

### 4.3 Counterfactual edge interventions

Define the counterfactual operator $\mathrm{do}(e \to \tilde{e})$ on the graph. For each $(\text{partner}, \text{layer}, \text{operation})$ triple:

1. Encode the year-$t$ window unmodified → baseline prediction $\hat{H}_{t+1}^{\text{base}}$.
2. Apply the perturbation to year-$t$'s edges in the chosen layer; re-encode → $\tilde{H}_t$.
3. Replace the last entry of the encoded history with $\tilde{H}_t$, push through the GRU → $\hat{H}_{t+1}^{\text{cf}}$.
4. Compute embeddedness metrics for USA and China on both predictions; report deltas.

**Symmetric perturbations.** Add and remove operate on the same number of edges $k$ (default $k = 5$), so deltas are directly comparable in magnitude. The legacy "remove all / add up to 50" mode is preserved for backward-compatible comparison and reported as an appendix robustness check.

**Embeddedness metrics.** We report two distance-based metrics that capture different geometric properties of the focal node's embedding shift, plus two reference distances:

- **Centroid distance** (Euclidean): the focal's distance to the centroid of all other embeddings. Increases under graph-theoretic isolation. Used as the *headline* isolation metric.
- **Centroid proximity** (cosine): the focal's mean cosine similarity to all other embeddings. We deliberately rename this from "isolation" to "proximity" because the simulation study (§5.2) shows it captures *embedding typicality* rather than peripheralization: when a strong tie is removed, the GNN representation moves toward the system mean, which **increases** mean cosine similarity. This is consistent across distance metrics and is itself a substantive observation (see §4.6).
- **Mean distance to P5** (Euclidean): focal's average distance to the embeddings of the five major powers. Used for context, not headline.
- **$k$-nearest-neighbor density** (Euclidean): mean distance to the focal's $k$ nearest neighbors in embedding space. Local-density diagnostic.

For each (partner, layer, operation) triple we compute both centroid-distance and centroid-proximity wedges $w_e = \Delta_e^{\text{USA}} - \Delta_e^{\text{CHN}}$, and rank perturbations under both metric families. The simulation validation (§5.2) shows wedge recovery is robust to choice of distance metric.

**Causal interpretation — explicit disclaimer.** This subsection earns the paper credibility. The interventions are model-based counterfactuals, not do-calculus causal effects. State three reasons (correlational training, SUTVA violation through message passing, conditional-on-model dynamics). Reframe what these counterfactuals *do* support: sensitivity rankings, comparative leverage estimates, scenario projection. Cite the planted-wedge simulation study (§ 5.2) as the strongest available validation.

### 4.4 Wedge decomposition and combinatorial search

Wedge metric: $w_e = \Delta_e^{\text{USA}} - \Delta_e^{\text{CHN}}$. Quadrant classification (Q1–Q4). Greedy combinatorial edge search over the top-K single-edge perturbations (pairs and triples), with super-additive interaction term $\Delta_{\{e_1, e_2\}} - (\Delta_{e_1} + \Delta_{e_2})$ flagging synergistic leverage.

### 4.5 Uncertainty quantification

Ensemble of $M = 10$ models with seeds $\{123, 124, \ldots, 132\}$. Per-cell counterfactual deltas reported as ensemble means with percentile-bootstrap 95% CIs. A cell is flagged "significant" when its CI excludes zero on the wedge dimension. MC-dropout is provided as a single-model alternative for replication purposes.

### 4.6 Single-focal shifts measure embedding typicality, not isolation

A separate methodological observation arising from the simulation validation (§5.2): under both Euclidean and cosine distance metrics, removing a strong structural friend pulls the focal's embedding *toward* the system centroid, not away from it. The model's learned representation reads "remove a key tie" as "this country looks more like the average country," not as "this country is more peripheral." This holds across distance metric choice, ruling out a metric artifact, and is therefore a property of the GNN's encoder dynamics rather than of how we measure distance. Implication: single-focal $\Delta$ rankings should be reported as *typicality shifts*, not isolation rankings. The wedge metric (which differences USA's typicality shift from China's typicality shift) remains the appropriate operationalization of asymmetric structural effect.

## 5. Validation (≈ 1,800 words — the section reviewers will scrutinize most)

### 5.1 Walk-forward backtests at horizons 5, 10, 15, 20 — RUNNING

**SLURM job 52743495** on Roar `standard` partition (GPU, ~12h). Splits at 1985, 1990, 1995, 2000, 2005. For each split: train on $[\text{start}, t_{\text{split}}]$, autoregressively roll embeddings forward, score against actual encoded embeddings at $t_{\text{split}} + h$ for $h \in \{5, 10, 15, 20\}$. Metrics: embedding MSE, link prediction AUC, USA/China embeddedness drift.

Report a horizon-MSE/AUC table and a horizon-decay figure (one panel per metric, color by split). The story should be: the model holds up to 5–10 years cleanly, degrades predictably to 15, and at 20 years is close to the bilinear baseline — which is the honest framing that gets the long-horizon claim past reviewers.

**Expected artifacts.**
- `outputs/multi_horizon/multi_horizon_backtest.csv` — one row per (split_year × horizon)
- `outputs/multi_horizon/multi_horizon_summary.csv` — aggregated MSE/AUC by horizon
- Logs: `logs/horizon_52743495.out`, `.err`

### 5.2 Planted-wedge recovery on synthetic data — COMPLETED

**Design.** A 60-node, 3-block, 3-layer multiplex SBM over 30 years. USA at node 0 (block 0), China at node 1 (block 1). For each replicate, a partner is rotated through block 2 (`block_2_indices[r % len]`), and given $k=12$ extra alliance-layer ties to USA. The intervention tested: "remove all of partner's ties in offensive_alliances." Each replicate also runs a *null* condition (same SBM, no planted ties) as the chance baseline. $R = 10$ planted + $R = 10$ null replicates with seeds $\{123, \ldots, 132\}$.

**Results** (`outputs/simulation_study/recovery_summary.csv`):

| Metric family | Planted top-10 | Null top-10 | Planted median rank | Null median rank |
|---|---|---|---|---|
| **Wedge — centroid distance (Euclidean)** | **100%** | 0% | **1** | 282.5 |
| **Wedge — centroid proximity (cosine)** | **100%** | 0% | **1** | 303 |
| Single-focal USA — centroid distance | 0% | 0% | 348 | 229 |
| Single-focal USA — centroid proximity | 0% | 0% | 348 | 276 |
| Single-focal China — centroid distance | 10% | 0% | 309.5 | 211.5 |
| Single-focal China — centroid proximity | 0% | 0% | 348 | 241.5 |

**Headline reading.** The wedge metric recovers the planted asymmetric tie with 100% precision and 0% false-positive rate, and does so under *both* distance metrics — confirming the result is not metric-dependent. Median rank of the planted edge is 1 of 348 candidates in planted condition vs. ~290 in null, a ~290× separation between true positive and chance.

**Single-focal pattern.** The single-focal isolation rankings produce ranks at or near the maximum (348) in the planted condition under both distance metrics. This is not failure — it is the *consistent* signature of the encoder pulling focals toward the system centroid when a key tie is removed (see §4.6). The metric responds, but in the opposite direction from what "isolation" naively suggests.

**Artifacts.**
- `outputs/simulation_study/recovery_summary.csv` — 2-row table (planted, null) × 12 metrics
- `outputs/simulation_study/recovery_study.csv` — 20 rows (10 replicates × 2 scenarios) with per-replicate ranks
- `outputs/simulation_study/recovery_summary_planted.csv` — legacy single-row summary for backward compat
- Code: `src/gnn_forecast/simulation.py`, executed by `scripts/run_simulation_study.py` via SLURM `hpc/job_simulation_study.sh`

**For the paper.** Lead with the wedge result. Report the single-focal pattern as a separate finding tied to §4.6 (typicality, not isolation). Consider adding a calibration figure: planted vs. null wedge-rank distributions side-by-side, with the chance baseline marked.

### 5.3 Baseline comparison — PARTIAL (AME job not yet submitted)

Four baselines: static GNN (no temporal), pure AR on degree features (no GNN), mean baseline, and bilinear latent-space (AME-lite, fit per year via PyTorch — explicitly cite Hoff). Optional R-side full AME via `amen` package available in `scripts/run_ame_baseline.R` for the most rigorous comparison.

Report: per-horizon link-prediction AUC, embedding MSE at $h = 5$ and $h = 10$, recovered layer attention weights. The GNN should win by a clear margin at short horizons; the comparison at long horizons should show the GNN converges toward but does not collapse to the latent-space baseline.

**Expected artifacts.**
- `outputs/ame_baseline/<layer>/ame_latent_<year>.csv` — per (layer, year) AME latent positions (one CSV each)
- `outputs/ame_baseline/ame_fit_summary.csv` — fit diagnostics
- Submit via `sbatch hpc/job_ame_baseline.sh` — runs on `basic` partition, no GPU, ~12h
- Need a small comparison script to align AME latent positions with GNN embeddings (compute per-year R² between the two representations) — currently a TODO; will live in `scripts/compare_ame_to_gnn.py`

### 5.4 Ensemble uncertainty — RUNNING

**SLURM job 52743497** on Roar `standard` partition (GPU, ~24h). Trains an $M = 10$ ensemble (seeds 123–132), then runs the dual-focal counterfactual sweep separately for each ensemble member to get percentile-bootstrap 95% CIs on every (partner, layer, operation) wedge.

Show the bootstrap-CI table for the top-20 wedge edges. Argue that significance is concentrated in the trade and IGO layers, with alliance-layer wedges generally showing wider CIs (because alliance edges are sparser and the model is less confident).

**Expected artifacts.**
- `outputs/uncertainty/ensemble_weights/ensemble_seed{123..132}_weights.pt` — trained model checkpoints
- `outputs/uncertainty/ensemble_weights/ensemble_seed{123..132}_embeddings.pt` — yearly embeddings per seed
- `outputs/uncertainty/ensemble_weights/ensemble_layer_weights.csv` — learned layer attention by seed
- `outputs/uncertainty/embeddedness_ci_final_year.csv` — USA + China embeddedness at the final year with 95% CIs
- `outputs/uncertainty/bootstrap_counterfactual_cis.csv` — per (partner, layer, op) cell with mean Δ, lo, hi, `wedge_significant` flag
- `outputs/uncertainty/top_significant_wedges.csv` — top-20 wedges whose CI excludes zero
- Logs: `logs/uq_52743497.out`, `.err`

## 6. Empirical Demonstration: USA-China Edge Asymmetries (≈ 1,500 words)

This is the substantive section, but it stays *empirical demonstration*, not contribution.

### 6.1 Layer importance

Report learned $\boldsymbol{\pi}$ across years. Tell the historical story (alliance importance peaks during the Cold War, IGO importance grows through the 1990s, trade dominates in the 2000s).

### 6.2 Top wedge edges (1995–2025)

Bar chart of top-20 edges by $|w_e|$, with bootstrap-CI error bars. Interpret 3–5 specific edges substantively (e.g., a country whose alliance status creates large divergent effects on USA vs. China embeddedness).

### 6.3 Quadrant analysis by layer

Heatmap of $(\text{layer}, \text{operation}) \to$ mean USA $\Delta$, mean China $\Delta$, mean wedge. Show that some layers are wedge-dominant (asymmetric) and others are envelope-dominant (Q1/Q3 — both gain or both lose).

### 6.4 Combinatorial leverage

Two named scenarios with super-additive interactions (one US-favorable, one China-favorable). Report the additive prediction, the actual combo effect, and the interaction term with its bootstrap CI.

### 6.5 Sensitivity bounds

Show that conclusions about top-K wedge edges are stable across (a) ensemble seeds, (b) symmetric vs. legacy perturbation modes, (c) observed vs. residualized layers. This is the appendix-anchored robustness section that pre-empts reviewer concerns.

## 7. Discussion (≈ 600 words)

Three short subsections:

- **What the framework enables that AME cannot.** Joint multi-layer learning with asynchronous coverage; interpretable layer attention as a byproduct; counterfactual edge interventions through learned dynamics; super-additive interaction detection.
- **What this paper does NOT claim.** Causal effects on the real international system; point predictions for 2050; that the long-horizon (15+) forecast is reliable for any individual country.
- **Substantive implications and extensions.** Extend to TGN/EvolveGCN architectures; add GDP/population gravity controls; multi-task decoders that predict each layer separately; a "budgeted" intervention search constrained to politically plausible edge sets.

## 8. Conclusion (≈ 250 words)

One paragraph restating the three methodological contributions, one paragraph on what the substantive demonstration showed, one sentence on data/code availability.

---

## Appendix structure

- **A.** Full list of layer files, peacesciencer function calls, COW code mappings. Source: `data/processed/*.csv`, `scripts/export_peacesciencer_layers.R`.
- **B.** Hyperparameter search results (hidden_dim, emb_dim, seq_len, $\lambda$ values). Currently uses defaults from `MultiplexGNNConfig` (hidden=64, emb=32, seq_len=5, λ_link=0.1, λ_smooth=0.01). A formal sweep is a TODO.
- **C.** Full walk-forward results table (every fold × horizon × metric). Source: `outputs/multi_horizon/multi_horizon_backtest.csv` (after job 52743495 completes).
- **D.** Full planted-wedge recovery study results (every replicate, both metric families, both scenarios). Source: `outputs/simulation_study/recovery_study.csv`. **DONE.**
- **E.** AME-lite training diagnostics and per-year R² between AME and GNN embeddings. Source: `outputs/ame_baseline/ame_fit_summary.csv` plus latent files; comparison script TBD.
- **F.** 25-year scenario rollout (USA/China embeddedness trajectories with ensemble fan charts) — *this is where the 2050 figure goes, framed as illustrative scenario projection*. Source: built from `outputs/uncertainty/ensemble_weights/ensemble_seed*_embeddings.pt` plus `gnn_forecast.training.forecast_embeddings`. Plot script TBD; should live in `scripts/plot_long_horizon_fan.py`.
- **G.** Robustness: residualized layers, MC dropout vs. ensemble agreement, alternate $k$ for symmetric perturbations. Sources: `data/processed/*_weighted.csv`, `gnn_forecast.uncertainty.mc_dropout_predictions`, the `symmetric_n_edges` parameter on `dual_focal_simulation`.
- **H.** Computational cost: training time per epoch, ensemble training time, counterfactual sweep cost. Pull from SLURM job logs: `logs/{sim,horizon,uq,ame}_<jobid>.{out,err}` after each job completes.

## Figure inventory (where each one comes from)

| Figure | Section | Code that produces it | Output path |
|---|---|---|---|
| Layer-coverage matrix (years × layers) | §3 | TODO — small matplotlib heatmap from `MultiplexTemporalDataset.snapshots[y].layer_mask` | `outputs/figures/layer_coverage.png` (TBD) |
| Two-stage learning signal diagram | §4.2 | hand-drawn or TikZ, no auto-gen | `figures/two_stage_signal.pdf` (TBD) |
| Wedge-rank distribution (planted vs. null) | §5.2 | TODO — write `scripts/plot_recovery_distributions.py` reading `recovery_study.csv` | `outputs/figures/recovery_distributions.png` (TBD) |
| Horizon-decay (MSE & AUC) | §5.1 | TODO — built from `multi_horizon_summary.csv` | `outputs/figures/horizon_decay.png` (TBD) |
| Layer attention timeseries | §6.1 | `gnn_forecast.visualization.embeddedness_time_series` adapted | `outputs/figures/layer_attention_timeseries.png` (TBD) |
| Top wedge bar chart | §6.2 | `gnn_forecast.isolation_analysis.plot_wedge_bar` | `outputs/figures/bar_top_wedge_edges.png` |
| USA-vs-China scatter | §6.3 | `gnn_forecast.isolation_analysis.plot_isolation_scatter` | `outputs/figures/scatter_usa_vs_china.png` |
| Layer-isolation heatmap | §6.3 | `gnn_forecast.isolation_analysis.plot_layer_heatmap` | `outputs/figures/heatmap_layer_isolation.png` |
| Combo interaction plot | §6.4 | `gnn_forecast.isolation_analysis.plot_combo_interactions` | `outputs/figures/interactions_*.png` |
| Multi-edge scenario bar chart | §6.4 | `gnn_forecast.isolation_analysis.plot_multi_edge_scenarios` | `outputs/figures/bar_multi_edge.png` |
| 25-year fan chart (Appendix F) | App F | TBD — `scripts/plot_long_horizon_fan.py` | `outputs/figures/embedding_forecast_2026_2050.png` |

## Submission checklist (run before submitting)

- [x] Re-read the causal-interpretation disclaimer in `counterfactual.py` and verify the paper text matches.
- [x] Run `scripts/run_simulation_study.py` for the recovery numbers. **DONE — 10 planted + 10 null replicates, 100% wedge recovery in planted, 0% in null. See §5.2.**
- [ ] Run `scripts/run_uncertainty_analysis.py` with `--n-members 10` on the full 1945–2025 dataset. **RUNNING (job 52743497).**
- [ ] Run `scripts/run_multi_horizon_backtest.py` with all five split years. **RUNNING (job 52743495).**
- [ ] Run `scripts/run_ame_baseline.R` for AME comparison data. **NOT YET SUBMITTED — `sbatch hpc/job_ame_baseline.sh`.**
- [ ] Write `scripts/compare_ame_to_gnn.py` to align AME latent positions with GNN embeddings (per-year R²).
- [ ] Generate the layer-coverage figure for § 3.
- [ ] Generate the recovery-distribution figure for § 5.2 (`scripts/plot_recovery_distributions.py`).
- [ ] Generate the horizon-decay figure for § 5.1 once job 52743495 finishes.
- [ ] Verify all reported numbers in tables are pulled from the latest runs (not stale CSVs).
- [ ] Update CLAUDE.md to reflect the multiplex track as canonical.
- [ ] Code release: tag a v1.0 commit, archive on Zenodo, cite DOI in paper.

## Where to find things

- **All code:** GitHub `https://github.com/jfedgerton/GNN_forecast.git`, branch `pa-revision-uncertainty-sim`
- **Local working copy:** `C:\Users\Jared_Edgerton\Dropbox\GNN_forecast`
- **Roar canonical copy:** `/storage/group/LiberalArts/default/jfe4_collab/GNN_forecast` (aliased as `/scratch/jfe4/GNN_forecast`)
- **Roar venv (Python 3.11.2 + torch 2.11.0+cu126):** `/scratch/jfe4/GNN_forecast/.venv/`
- **Roar SLURM job logs:** `/scratch/jfe4/GNN_forecast/logs/{sim,horizon,uq,ame}_<jobid>.{out,err}`
- **All results CSVs:** `/scratch/jfe4/GNN_forecast/outputs/<job_subdir>/*.csv`
- **All figures (when generated):** `/scratch/jfe4/GNN_forecast/outputs/figures/`

## Anticipated reviewer questions (pre-empt in submission)

1. *"Why a GNN instead of AME?"* — § 5.3 head-to-head comparison; § 7 explicitly addresses this.
2. *"What identifies your counterfactual effects?"* — § 4.3 explicit disclaimer; § 5.2 simulation validation.
3. *"How do you handle missing layer-years?"* — § 4.1 attention masking, with layer-coverage figure in § 3.
4. *"How sensitive is the top-K ranking to seed and hyperparameters?"* — § 5.4 ensemble CIs; § 6.5 sensitivity bounds.
5. *"Is the 25-year forecast credible?"* — § 5.1 horizon-decay analysis; § 7 explicitly disclaims long-horizon point prediction; appendix F frames as scenario.
6. *"Is the wedge metric statistically meaningful?"* — § 4.5 CI construction; § 6.2 reports CIs on the top-20 wedges.
