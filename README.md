# GNN Forecast: Multiplex International-System Forecasting (1945–2050)

This repository now includes a full research scaffold for your idea:

- Build a **multiplex network** from peacesciencer layers:
  1. Defensive/offensive alliances
  2. Shared IGO membership
  3. Trade
- Create two layer sets:
  - **Observed layers**
  - **Residualized layers** where `tie_resid = observed_tie - predicted_tie` and predicted tie comes from `tie ~ capital_distance`.
- Fit several GNN encoders (GCN, GraphSAGE, GAT) to produce embeddings.
- Fit an autoregressive neural model over embeddings and forecast forward.
- Simulate edge additions/removals targeting USA and China to assess which tie interventions increase/decrease embeddedness.

## Repository Structure

- `scripts/03_export_peacesciencer_layers.R`: Exports node and layer CSVs from peacesciencer.
- `scripts/run_research_pipeline.py`: Builds observed/residual layers by year and exports model inputs.
- `scripts/train_forecast_and_intervene.py`: Forecast scaffold (2026–2050) + intervention analysis.
- `src/gnn_forecast/data_peacesciencer.py`: Loader for exported peacesciencer files.
- `src/gnn_forecast/network_construction.py`: Capital-distance residualization utilities.
- `src/gnn_forecast/models.py`: GCN/GraphSAGE/GAT encoders.
- `src/gnn_forecast/forecast.py`: AR embedding forecaster and edge decoder.
- `src/gnn_forecast/interventions.py`: Edge knockout/addition simulation helpers.

## Quick Start

## 1) Export layers from peacesciencer (R)

```bash
Rscript scripts/03_export_peacesciencer_layers.R
```

This writes:

- `data/processed/nodes.csv`
- `data/processed/layer_alliances_defensive_offensive.csv`
- `data/processed/layer_igo_shared.csv`
- `data/processed/layer_trade.csv`

> Note: peacesciencer function names vary by version. The script tries multiple candidate function names and fails with a clear error if your installed version differs.

## 2) Build observed + residualized layer datasets

```bash
PYTHONPATH=src python scripts/run_research_pipeline.py \
  --data-dir data/processed \
  --start-year 1945 \
  --end-year 2025 \
  --out-dir data/model_inputs
```

## 3) Fit GNN(s), forecast embeddings, run interventions

Use your own training code for GNN encoders to generate a tensor:

- `embedding_history.pt` shape `[num_nodes, seq_len, emb_dim]`

Then run:

```bash
PYTHONPATH=src python scripts/train_forecast_and_intervene.py \
  --embedding-file embedding_history.pt \
  --adj-file adjacency_2025.npy \
  --nodes-file data/processed/nodes.csv \
  --out-dir outputs
```

Outputs:

- `outputs/embedding_forecast_2026_2050.pt`
- `outputs/interventions_usa.csv`
- `outputs/interventions_china.csv`

## Notes on a 2026–2050 horizon

A 25-year forecast is possible but should be framed as **scenario projection**, not point prediction certainty. For credibility:

- Evaluate shorter horizons too (1, 3, 5, 10 years).
- Use walk-forward backtests.
- Report uncertainty bands across model classes and random seeds.
- Treat intervention outputs as directional/relative rankings.

## Next additions you may want

- Temporal graph baselines (TGN, EvolveGCN, DySAT).
- Multi-task decoder that predicts each layer separately.
- Uncertainty quantification (ensembles, MC dropout).
- Constraint-aware intervention search (budgeted edge edits).
