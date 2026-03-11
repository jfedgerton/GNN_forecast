# GNN Forecast Repository Audit (Correctness, Temporal Integrity, and Research Validity)

## 1. Executive summary

- The repository implements **two partly overlapping pipelines**:
  1. a lightweight scaffold (`pipeline.py`, `run_research_pipeline.py`, `train_forecast_and_intervene.py`) and
  2. a fuller multiplex temporal GNN stack (`multiplex_data.py`, `multiplex_model.py`, `training.py`, `validation.py`, `run_full_pipeline.py`, `run_manuscript_analysis.py`).
- The core multiplex training loop is **conceptually coherent** for one-step temporal embedding prediction, but several implementation choices materially affect interpretability and validity.
- The most serious issue is an **incorrect AUC computation direction** in validation, which can invert model-quality conclusions.
- A second serious issue is **counterfactual semantics mismatch** (code modifies one bilateral edge, docs claim full partner disconnection), which can invalidate substantive interpretation of intervention outputs.
- A third major concern is that the lightweight forecasting/intervention script can fail at runtime due to an argument mismatch (fixed in this audit).
- Bottom line: the codebase is **promising but not paper-ready without fixes and explicit methodological qualification**.

## 2. Pipeline reconstruction

### True entry points observed

1. `scripts/run_research_pipeline.py`:
   - Loads peacesciencer exports via `run_research_pipeline`.
   - Computes observed + residualized layers year-by-year.
   - Exports `observed_*_{year}.csv` and `residual_*_{year}.csv`.

2. `scripts/run_full_pipeline.py`:
   - Discovers raw and weighted layer CSVs.
   - Builds multiplex dataset tensors.
   - Trains multiplex temporal GNN.
   - Forecasts future embeddings.
   - Runs counterfactual analysis.

3. `scripts/run_manuscript_analysis.py`:
   - Superset workflow: training, backtesting, multi-seed, baselines, forecasts, counterfactuals, isolation analysis, tables.

4. `scripts/train_forecast_and_intervene.py`:
   - Standalone scaffold using precomputed embedding history + adjacency to roll forward embeddings and run simple edge toggles.

### Dataflow map (operational)

- Raw inputs: `data/processed/layer_*.csv` + optional `nodes.csv`.
- Discovery: `discover_layers()` / `discover_weighted_layers()` in `multiplex_data.py`.
- Canonical indexing: `build_global_node_index()` establishes stable `ccode -> idx` mapping.
- Yearly tensor build: `build_multiplex_dataset()` creates per-year edge indices, edge weights, masks, and node features (normalized degree profiles).
- Model encode/predict:
  - Snapshot encoding: per-layer GCN + layer attention aggregation.
  - Temporal prediction: GRU over sequence of yearly embeddings.
- Training objective: embedding MSE + link reconstruction BCE + temporal smoothness.
- Forecasting: recursive rollout by feeding predicted embeddings back into temporal model.
- Evaluation: walk-forward backtest computes embedding MSE and “AUC”.
- Intervention/counterfactual:
  - `counterfactual.py`: one-edge add/remove perturbation on target snapshot.
  - `isolation_analysis.py`: dual-focal (USA/China) and multi-edge variants.

## 3. Critical issues

### C1. Link AUC calculation is directionally wrong (can invert conclusions)
- **Current behavior:** `_approximate_link_auc` sorts scores descending and computes a Mann-Whitney-like statistic using cumulative negatives before positives.
- **Why wrong:** That statistic as coded returns higher values when positives are ranked *lower*; effectively `1 - AUC` under common conventions.
- **Severity:** **Critical** (evaluation headline metric may be inverted).
- **Fix:** Replace with explicit pairwise ranking implementation or `sklearn.metrics.roc_auc_score` on sampled pos/neg scores.

### C2. Counterfactual documentation vs implementation mismatch
- **Current behavior:** `simulate_single_edge` modifies only the focal-partner bilateral edge(s) in one layer.
- **Why questionable/wrong:** `isolation_analysis` docs claim removing/adding **all partner edges** in a layer for system-wide perturbation; this does not match operation in core single-edge module and risks misinterpreting intervention claims.
- **Severity:** **Critical** for substantive claims.
- **Fix:** Either (a) change implementation to truly remove/add all partner incident edges when that design is intended, or (b) correct docs/output labels to “single bilateral edge perturbation.”

## 4. Major issues

### M1. Lightweight intervention script had runtime bug (fixed)
- **Current behavior (before fix):** `simulate_edge_toggle(..., partners=partners)` used an invalid keyword; function expects `partner_ccodes`.
- **Impact:** Script fails before generating USA/China outputs.
- **Fix applied:** Updated keyword to `partner_ccodes` in both calls.

### M2. Baseline/target space comparability is weak
- **Current behavior:** Main model predicts embeddings; “mean baseline” predicts averaged node features; losses compared directly in a table.
- **Why major:** This is not a like-for-like predictive target comparison and can mislead baseline claims.
- **Fix:** Ensure all baselines predict the same representation target (e.g., encoded embeddings from frozen encoder, or direct network-level metrics).

### M3. Negative sampling includes possible true edges/self-edges
- **Current behavior:** Link-loss negative samples are drawn uniformly at random, without filtering positives or diagonal.
- **Why major:** Noisy negatives weaken interpretability of link-regularizer and AUC proxy.
- **Fix:** sample from complement edge set and exclude `i==j`.

### M4. Reproducibility controls incomplete in primary scripts
- **Current behavior:** multi-seed module sets seeds, but main pipelines do not set deterministic seeds globally by default.
- **Why major:** reported numbers may drift across runs.
- **Fix:** add top-level seed argument and set `torch`, `numpy`, and deterministic backend options.

## 5. Moderate issues

- Per-year feature normalization (`compute_degree_features`) rescales by each year’s max degree. This is not leakage but may distort inter-year comparability; document rationale or use train-period scaler for forecast tasks requiring comparable magnitudes.
- Combined alliance file fallback maps into only `defensive_alliances`, potentially re-labeling semantics.
- Counterfactual “add” may duplicate pre-existing edges (no dedup check).
- Weighted-layer discovery and raw-layer naming can create silent layer-set differences between model sets; should be logged and asserted.

## 6. Minor issues

- Multiple legacy/parallel modules (`data_layer.py`, `data_peacesciencer.py`, simple scaffold scripts) increase ambiguity on canonical path.
- Some comments overstate “full model forward counterfactual” despite using one modified snapshot and one-step prediction.
- Optional imports and fallbacks are fine but should be listed in reproducibility docs.

## 7. File-by-file notes (key modules)

- `src/gnn_forecast/multiplex_data.py`:
  - Purpose: layer discovery, indexing, yearly tensor construction.
  - Mostly correct; good temporal masking.
  - Risks: fallback semantics and per-year scaling comparability.

- `src/gnn_forecast/multiplex_model.py`:
  - Purpose: layer GCN encoders + attention + temporal GRU.
  - Architecturally coherent for embedding forecasting.
  - Acceptable; no direct leakage observed.

- `src/gnn_forecast/training.py`:
  - Purpose: windowed training and forecasting rollout.
  - Temporal ordering is correct for one-step supervised windows.
  - Needs stronger negative sampling and clearer train/val split support.

- `src/gnn_forecast/validation.py`:
  - Purpose: walk-forward evaluation.
  - Major flaw: AUC inversion risk.

- `src/gnn_forecast/counterfactual.py`:
  - Purpose: edge intervention effects on embeddedness metrics.
  - Core mechanics are computationally valid, but interpretation depends on intervention semantics and edge dedup handling.

- `src/gnn_forecast/isolation_analysis.py`:
  - Purpose: paired USA/China intervention effects and rankings.
  - Useful tooling, but narrative of “full partner disconnection” must match implemented perturbation exactly.

- `src/gnn_forecast/baselines.py`:
  - Purpose: static and non-GNN comparisons.
  - Baseline target comparability needs tightening.

- `scripts/run_full_pipeline.py` and `scripts/run_manuscript_analysis.py`:
  - Purpose: orchestrate end-to-end research runs.
  - Good for reproducible execution once seeds and metric bug are fixed.

- `scripts/train_forecast_and_intervene.py`:
  - Purpose: standalone rollout/intervention on precomputed embeddings.
  - Bug fixed in this audit (keyword mismatch).

## 8. Recommended validation checks (“trust but verify”)

1. Year-layer edge counts and availability masks (`layer_availability.csv`) across all years.
2. Node continuity: verify fixed `ccode -> idx` and missing-node behavior by year.
3. Temporal boundaries: inspect each training window `(t-seq_len+1 ... t) -> t+1`.
4. No leakage checks:
   - feature construction uses only year `t` edges,
   - any scaling fitted only on allowed periods when doing strict forecasting evaluation.
5. AUC sanity test: synthetic perfect-separation case should yield near 1.0.
6. One-step vs rollout: compare backtest with teacher-forced one-step and recursive rollout errors.
7. Intervention sanity:
   - confirm exactly what edge set is changed,
   - verify no duplicate edges added,
   - inspect top USA/China interventions manually.
8. Seed sensitivity: run `multi_seed_evaluation` with >=5 seeds and report dispersion.

## 9. Minimal repair plan

1. **Fix AUC implementation** in `validation.py` and add a unit test that catches inversion.
2. **Resolve intervention semantics** (single-edge vs all-partner-edge) in code and docs.
3. **Harden negative sampling** for link loss/AUC (exclude positives + self edges).
4. **Add reproducibility controls** (global seed argument in all runner scripts).
5. **Unify baseline target space** to avoid misleading benchmark comparisons.
6. Keep architecture/workflow unchanged otherwise.

## 10. Final verdict

- **Is implementation basically correct?** Partially: core temporal GNN training/rollout logic is broadly coherent.
- **Reliable parts:** dataset construction with explicit year masks, model encode/temporal modules, walk-forward structure concept.
- **Not trustworthy yet:** evaluation AUC values, intervention interpretation, and some baseline comparison claims.
- **Must fix before paper use:** AUC bug, intervention semantics consistency, reproducibility defaults, and clearer benchmark validity.
