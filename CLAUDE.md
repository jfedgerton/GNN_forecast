# CLAUDE.md

## Project Overview

GNN Forecast is a research scaffold for **multiplex international-system forecasting (1945–2050)** using Graph Neural Networks. It builds multiplex networks from international relations data (alliances, IGO membership, trade) and forecasts country embeddings and ties out to 2050, with edge-intervention analysis for the USA and China.

## Tech Stack

- **Python 3.10+** — main language (PyTorch, torch_geometric, pandas, numpy)
- **R** — data export via the `peacesciencer` package

## Repository Structure

```
src/gnn_forecast/       # Core Python package
scripts/                # Executable pipeline scripts (R + Python)
tests/                  # pytest test suite
tests/fixtures/         # Tiny processed data for tests
data/processed/         # Peacesciencer CSV exports (gitignored)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `data_peacesciencer.py` | Load peacesciencer exports into `MultiplexYear` objects |
| `data_layer.py` | Canonical multiplex data loader with deterministic node indexing |
| `network_construction.py` | Capital-distance residualization of edge weights |
| `models.py` | GNN encoders (GCN, GraphSAGE, GAT) |
| `forecast.py` | GRU-based autoregressive embedding forecaster |
| `interventions.py` | Edge-toggle intervention simulation |
| `pipeline.py` | Research pipeline orchestration |

## Setup

```bash
pip install -e .
# Optional: pip install torch_geometric
```

## Running Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

## Pipeline

1. **Export data (R):** `Rscript scripts/export_peacesciencer_layers.R`
2. **Build layers (Python):** `PYTHONPATH=src python scripts/run_research_pipeline.py --data-dir data/processed --start-year 1945 --end-year 2025 --out-dir data/model_inputs`
3. **Train & forecast (Python):** `PYTHONPATH=src python scripts/train_forecast_and_intervene.py --embedding-file embedding_history.pt --adj-file adjacency_2025.npy --nodes-file data/processed/nodes.csv --out-dir outputs`

## Code Conventions

- **Type hints** throughout; use `from __future__ import annotations`
- **Dataclasses** for configuration and result objects
- **snake_case** for functions/variables, **PascalCase** for classes
- Modular single-responsibility functions
- Explicit error messages with `FileNotFoundError` / `ValueError`
- Imports organized alphabetically; standard library, then third-party, then local

## Important Notes

- Always set `PYTHONPATH=src` when running scripts or tests outside of `pip install -e .`
- The 25-year forecast (2026–2050) is a scenario projection, not a point prediction
- No CI/CD is configured yet
