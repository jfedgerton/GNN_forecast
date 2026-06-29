"""Tests for the uncertainty quantification module."""
from __future__ import annotations

import numpy as np
import torch

from gnn_forecast.simulation import generate_synthetic_multiplex
from gnn_forecast.uncertainty import (
    train_ensemble,
    embeddedness_with_cis,
    bootstrap_counterfactual_cis,
    BASE_SEED,
)
from gnn_forecast.training import TrainingConfig
from gnn_forecast.multiplex_model import MultiplexGNNConfig
from gnn_forecast.counterfactual import USA_CCODE, CHN_CCODE


def _tiny_dataset():
    return generate_synthetic_multiplex(num_nodes=15, num_years=10, seed=123).dataset


def test_train_ensemble_returns_correct_member_count() -> None:
    np.random.seed(123)
    torch.manual_seed(123)
    dataset = _tiny_dataset()
    cfg = MultiplexGNNConfig(
        num_layers=len(dataset.layer_names),
        in_dim=len(dataset.layer_names),
        hidden_dim=8, emb_dim=4, seq_len=3,
    )
    train_cfg = TrainingConfig(num_epochs=3, patience=2, print_every=10, seq_len=3)
    ens = train_ensemble(
        dataset=dataset, n_members=2, model_config=cfg, train_config=train_cfg,
    )
    assert len(ens.models) == 2
    assert len(ens.yearly_embeddings) == 2
    assert ens.seeds == [BASE_SEED, BASE_SEED + 1]


def test_embeddedness_with_cis_returns_bounds() -> None:
    np.random.seed(123)
    torch.manual_seed(123)
    dataset = _tiny_dataset()
    cfg = MultiplexGNNConfig(
        num_layers=len(dataset.layer_names),
        in_dim=len(dataset.layer_names),
        hidden_dim=8, emb_dim=4, seq_len=3,
    )
    train_cfg = TrainingConfig(num_epochs=3, patience=2, print_every=10, seq_len=3)
    ens = train_ensemble(
        dataset=dataset, n_members=3, model_config=cfg, train_config=train_cfg,
    )
    final_year = dataset.years[-1]
    ci = embeddedness_with_cis(ens, dataset, final_year, USA_CCODE)
    assert ci["mean_cosine_lo"] <= ci["mean_cosine_mean"] <= ci["mean_cosine_hi"]
    assert ci["n_members"] == 3
