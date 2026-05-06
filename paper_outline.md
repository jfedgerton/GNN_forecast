# Paper Outline — Multiplex GNN Forecasting for the International System

**Target journal:** Political Analysis
**Working title:** *Multiplex Graph Neural Networks for International System Forecasting and Counterfactual Edge Analysis*
**Author:** Jared Edgerton

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

**Embeddedness metrics.** Mean cosine similarity to all other states; centroid distance; mean distance to P5; $k$-nearest-neighbor density.

**Causal interpretation — explicit disclaimer.** This subsection earns the paper credibility. The interventions are model-based counterfactuals, not do-calculus causal effects. State three reasons (correlational training, SUTVA violation through message passing, conditional-on-model dynamics). Reframe what these counterfactuals *do* support: sensitivity rankings, comparative leverage estimates, scenario projection. Cite the planted-wedge simulation study (§ 5.2) as the strongest available validation.

### 4.4 Wedge decomposition and combinatorial search

Wedge metric: $w_e = \Delta_e^{\text{USA}} - \Delta_e^{\text{CHN}}$. Quadrant classification (Q1–Q4). Greedy combinatorial edge search over the top-K single-edge perturbations (pairs and triples), with super-additive interaction term $\Delta_{\{e_1, e_2\}} - (\Delta_{e_1} + \Delta_{e_2})$ flagging synergistic leverage.

### 4.5 Uncertainty quantification

Ensemble of $M = 10$ models with seeds $\{123, 124, \ldots, 132\}$. Per-cell counterfactual deltas reported as ensemble means with percentile-bootstrap 95% CIs. A cell is flagged "significant" when its CI excludes zero on the wedge dimension. MC-dropout is provided as a single-model alternative for replication purposes.

## 5. Validation (≈ 1,800 words — the section reviewers will scrutinize most)

### 5.1 Walk-forward backtests at horizons 5, 10, 15, 20

Splits at 1985, 1990, 1995, 2000, 2005. For each split: train on $[\text{start}, t_{\text{split}}]$, autoregressively roll embeddings forward, score against actual encoded embeddings at $t_{\text{split}} + h$ for $h \in \{5, 10, 15, 20\}$. Metrics: embedding MSE, link prediction AUC, USA/China embeddedness drift.

Report a horizon-MSE/AUC table and a horizon-decay figure (one panel per metric, color by split). The story should be: the model holds up to 5–10 years cleanly, degrades predictably to 15, and at 20 years is close to the bilinear baseline — which is the honest framing that gets the long-horizon claim past reviewers.

### 5.2 Planted-wedge recovery on synthetic data

Generate a 60-node, 3-block, 3-layer multiplex SBM over 30 years. Plant a known wedge edge (one partner densely connected to "USA" in one layer, to "China" in another). Train the GNN; run dual-focal counterfactual sweep; report the rank of the planted edge in the top-K isolation/wedge rankings, averaged over $R = 10$ replicates with seeds $\{123, \ldots, 132\}$.

This is the most reviewer-defensible methodological claim. State the recovery-rate-at-$K$ explicitly.

### 5.3 Baseline comparison

Four baselines: static GNN (no temporal), pure AR on degree features (no GNN), mean baseline, and bilinear latent-space (AME-lite, fit per year via PyTorch — explicitly cite Hoff). Optional R-side full AME via `amen` package available in `scripts/run_ame_baseline.R` for the most rigorous comparison.

Report: per-horizon link-prediction AUC, embedding MSE at $h = 5$ and $h = 10$, recovered layer attention weights. The GNN should win by a clear margin at short horizons; the comparison at long horizons should show the GNN converges toward but does not collapse to the latent-space baseline.

### 5.4 Ensemble uncertainty

Show the bootstrap-CI table for the top-20 wedge edges. Argue that significance is concentrated in the trade and IGO layers, with alliance-layer wedges generally showing wider CIs (because alliance edges are sparser and the model is less confident).

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

- **A.** Full list of layer files, peacesciencer function calls, COW code mappings.
- **B.** Hyperparameter search results (hidden_dim, emb_dim, seq_len, $\lambda$ values).
- **C.** Full walk-forward results table (every fold × horizon × metric).
- **D.** Full planted-wedge recovery study results (every replicate, every layer).
- **E.** AME-lite training diagnostics and per-year R² between AME and GNN embeddings.
- **F.** 25-year scenario rollout (USA/China embeddedness trajectories with ensemble fan charts) — *this is where the 2050 figure goes, framed as illustrative scenario projection*.
- **G.** Robustness: residualized layers, MC dropout vs. ensemble agreement, alternate $k$ for symmetric perturbations.
- **H.** Computational cost: training time per epoch, ensemble training time, counterfactual sweep cost.

## Submission checklist (run before submitting)

- [ ] Re-read the causal-interpretation disclaimer in `counterfactual.py` and verify the paper text matches.
- [ ] Run `scripts/run_simulation_study.py` with `--replicates 20` for the final recovery numbers.
- [ ] Run `scripts/run_uncertainty_analysis.py` with `--n-members 10` on the full 1945–2025 dataset.
- [ ] Run `scripts/run_multi_horizon_backtest.py` with all five split years.
- [ ] Run `scripts/run_ame_baseline.R` for AME comparison data.
- [ ] Generate the layer-coverage figure for § 3.
- [ ] Verify all reported numbers in tables are pulled from the latest runs (not stale CSVs).
- [ ] Update CLAUDE.md to reflect the multiplex track as canonical.
- [ ] Code release: tag a v1.0 commit, archive on Zenodo, cite DOI in paper.

## Anticipated reviewer questions (pre-empt in submission)

1. *"Why a GNN instead of AME?"* — § 5.3 head-to-head comparison; § 7 explicitly addresses this.
2. *"What identifies your counterfactual effects?"* — § 4.3 explicit disclaimer; § 5.2 simulation validation.
3. *"How do you handle missing layer-years?"* — § 4.1 attention masking, with layer-coverage figure in § 3.
4. *"How sensitive is the top-K ranking to seed and hyperparameters?"* — § 5.4 ensemble CIs; § 6.5 sensitivity bounds.
5. *"Is the 25-year forecast credible?"* — § 5.1 horizon-decay analysis; § 7 explicitly disclaims long-horizon point prediction; appendix F frames as scenario.
6. *"Is the wedge metric statistically meaningful?"* — § 4.5 CI construction; § 6.2 reports CIs on the top-20 wedges.
