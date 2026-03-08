"""Multilayer network forecasting and intervention tooling."""

from .data_peacesciencer import build_multiplex_from_peacesciencer
from .pipeline import run_research_pipeline
from .multiplex_data import (
    MultiplexSnapshot,
    MultiplexTemporalDataset,
    build_multiplex_dataset,
    discover_layers,
)
from .multiplex_model import MultiplexTemporalGNN, MultiplexGNNConfig
from .training import train_model, forecast_embeddings, TrainingConfig
from .counterfactual import (
    batch_counterfactual_analysis,
    results_to_dataframe,
    compute_embeddedness_metrics,
)

__all__ = [
    "build_multiplex_from_peacesciencer",
    "run_research_pipeline",
    "MultiplexSnapshot",
    "MultiplexTemporalDataset",
    "build_multiplex_dataset",
    "discover_layers",
    "MultiplexTemporalGNN",
    "MultiplexGNNConfig",
    "train_model",
    "forecast_embeddings",
    "TrainingConfig",
    "batch_counterfactual_analysis",
    "results_to_dataframe",
    "compute_embeddedness_metrics",
]
