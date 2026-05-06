"""Walk-forward backtesting and out-of-sample validation.

Implements temporal cross-validation: train on [start, split_year], test on
(split_year, end]. Reports embedding prediction MSE, link prediction AUC,
and counterfactual stability across folds.

The horizon-by-horizon evaluator (`multi_horizon_backtest`) reports
performance separately at the 5-, 10-, 15-, and 20-year forecast horizons,
which replaces the original 25-year point-prediction framing per Political
Analysis revision recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .multiplex_data import MultiplexTemporalDataset, build_multiplex_dataset
from .multiplex_model import MultiplexGNNConfig, MultiplexTemporalGNN
from .training import TrainingConfig, TrainingResult, train_model
from .counterfactual import compute_embeddedness_metrics, USA_CCODE, CHN_CCODE


@dataclass
class FoldResult:
    """Metrics from one backtesting fold."""
    fold_id: int
    train_years: List[int]
    test_years: List[int]
    train_loss_final: float
    # Per-test-year embedding prediction MSE
    test_emb_mse: Dict[int, float]
    # Per-test-year link prediction AUC (approximate)
    test_link_auc: Dict[int, float]
    # Layer weights learned during training
    layer_weights: Dict[str, float]
    # Embeddedness of USA and China in test years
    usa_embeddedness: Dict[int, float]
    china_embeddedness: Dict[int, float]


@dataclass
class BacktestResult:
    """Aggregate results from walk-forward backtesting."""
    folds: List[FoldResult]
    summary: pd.DataFrame


def _approximate_link_auc(
    pred_emb: torch.Tensor,
    true_edge_index: torch.LongTensor,
    num_nodes: int,
    n_neg: int = 1000,
) -> float:
    """Approximate link prediction AUC using inner product scores.

    Samples positive edges from the true graph and negative edges randomly,
    then computes AUC from the score distributions.
    """
    if true_edge_index.numel() == 0:
        return 0.5

    scores = pred_emb @ pred_emb.T

    # Positive edge scores
    src, tgt = true_edge_index[0], true_edge_index[1]
    n_pos = min(len(src), n_neg)
    perm = torch.randperm(len(src))[:n_pos]
    pos_scores = scores[src[perm], tgt[perm]].cpu().numpy()

    # Negative edge scores
    neg_src = torch.randint(0, num_nodes, (n_neg,))
    neg_tgt = torch.randint(0, num_nodes, (n_neg,))
    neg_scores = scores[neg_src, neg_tgt].cpu().numpy()

    # Simple AUC: fraction of times a positive score > a negative score
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    # Sort by score descending, count concordance
    order = np.argsort(-all_scores)
    sorted_labels = all_labels[order]
    n_pos_total = sorted_labels.sum()
    n_neg_total = len(sorted_labels) - n_pos_total

    if n_pos_total == 0 or n_neg_total == 0:
        return 0.5

    # Wilcoxon-Mann-Whitney statistic
    cum_neg = np.cumsum(1 - sorted_labels)
    auc = np.sum(sorted_labels * cum_neg) / (n_pos_total * n_neg_total)
    return float(auc)


def _merge_all_edges(snapshot, device) -> torch.LongTensor:
    """Merge edge indices from all available layers in a snapshot."""
    parts = []
    for ln, mask in snapshot.layer_mask.items():
        if mask and ln in snapshot.edge_indices and snapshot.edge_indices[ln].numel() > 0:
            parts.append(snapshot.edge_indices[ln].to(device))
    if not parts:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    return torch.cat(parts, dim=1)


def walk_forward_backtest(
    dataset: MultiplexTemporalDataset,
    split_years: List[int],
    model_config: Optional[MultiplexGNNConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    test_horizon: int = 5,
    seq_len: int = 5,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> BacktestResult:
    """Run walk-forward backtesting.

    For each split_year:
      - Train on [dataset.years[0], split_year]
      - Test on (split_year, split_year + test_horizon]
      - Report embedding MSE, link AUC, embeddedness metrics

    Parameters
    ----------
    dataset : MultiplexTemporalDataset
    split_years : list of years to use as training cutoffs
    model_config : MultiplexGNNConfig (defaults auto-configured)
    train_config : TrainingConfig (defaults auto-configured)
    test_horizon : number of years to test after split
    seq_len : temporal window for the model
    device : torch device
    save_dir : optional directory to save per-fold outputs
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_years = dataset.years

    if model_config is None:
        model_config = MultiplexGNNConfig(
            num_layers=len(dataset.layer_names),
            in_dim=len(dataset.layer_names),
            hidden_dim=64,
            emb_dim=32,
            temporal_hidden_dim=64,
            seq_len=seq_len,
        )

    if train_config is None:
        train_config = TrainingConfig(
            num_epochs=100,
            seq_len=seq_len,
            patience=20,
            print_every=25,
        )

    folds: List[FoldResult] = []

    for fold_id, split_year in enumerate(split_years):
        print(f"\n--- Fold {fold_id + 1}/{len(split_years)}: "
              f"train <= {split_year}, test > {split_year} ---")

        train_years = [y for y in all_years if y <= split_year]
        test_years = [y for y in all_years if split_year < y <= split_year + test_horizon]

        if len(train_years) < seq_len + 1:
            print(f"  Skipping fold: only {len(train_years)} training years (need {seq_len + 1})")
            continue
        if len(test_years) == 0:
            print(f"  Skipping fold: no test years available")
            continue

        # Build train subset
        train_dataset = _subset_dataset(dataset, train_years)

        # Configure save directory for this fold
        fold_save = None
        if save_dir is not None:
            fold_save = str(Path(save_dir) / f"fold_{fold_id}")

        fold_train_config = TrainingConfig(
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            num_epochs=train_config.num_epochs,
            seq_len=train_config.seq_len,
            lambda_link=train_config.lambda_link,
            lambda_smooth=train_config.lambda_smooth,
            link_sample_size=train_config.link_sample_size,
            patience=train_config.patience,
            min_delta=train_config.min_delta,
            print_every=train_config.print_every,
            save_dir=fold_save,
        )

        # Train
        result = train_model(
            dataset=train_dataset,
            model_config=model_config,
            train_config=fold_train_config,
            device=device,
        )

        model = result.model
        model.eval()

        # Evaluate on test years
        # First, build the history from the last seq_len training years
        history: List[torch.Tensor] = []
        with torch.no_grad():
            for y in train_years[-seq_len:]:
                snap = dataset.snapshots[y]
                nf = snap.node_features.to(device)
                ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
                ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
                emb = model.encode_snapshot(nf, ei, ew, snap.layer_mask)
                history.append(emb)

        test_emb_mse: Dict[int, float] = {}
        test_link_auc: Dict[int, float] = {}
        usa_embeddedness: Dict[int, float] = {}
        china_embeddedness: Dict[int, float] = {}

        with torch.no_grad():
            for test_year in test_years:
                # Predict next embedding
                pred_emb = model.forward_temporal(history[-seq_len:])

                # Actual embedding from the test year
                test_snap = dataset.snapshots[test_year]
                nf = test_snap.node_features.to(device)
                ei = {k: v.to(device) for k, v in test_snap.edge_indices.items()}
                ew = {k: v.to(device) for k, v in test_snap.edge_weights.items()}
                actual_emb = model.encode_snapshot(nf, ei, ew, test_snap.layer_mask)

                # Embedding MSE
                mse = F.mse_loss(pred_emb, actual_emb).item()
                test_emb_mse[test_year] = mse

                # Link prediction AUC
                merged_ei = _merge_all_edges(test_snap, device)
                auc = _approximate_link_auc(pred_emb, merged_ei, dataset.num_nodes)
                test_link_auc[test_year] = auc

                # Embeddedness for USA and China
                if USA_CCODE in dataset.ccode_to_idx:
                    usa_idx = dataset.ccode_to_idx[USA_CCODE]
                    usa_metrics = compute_embeddedness_metrics(
                        pred_emb, usa_idx, dataset.ccode_to_idx,
                    )
                    usa_embeddedness[test_year] = usa_metrics.mean_cosine_similarity

                if CHN_CCODE in dataset.ccode_to_idx:
                    chn_idx = dataset.ccode_to_idx[CHN_CCODE]
                    chn_metrics = compute_embeddedness_metrics(
                        pred_emb, chn_idx, dataset.ccode_to_idx,
                    )
                    china_embeddedness[test_year] = chn_metrics.mean_cosine_similarity

                # Slide window forward
                history.append(pred_emb)

        avg_mse = np.mean(list(test_emb_mse.values()))
        avg_auc = np.mean(list(test_link_auc.values()))
        print(f"  Test avg MSE: {avg_mse:.6f}, avg AUC: {avg_auc:.4f}")
        print(f"  Layer weights: {result.layer_weights}")

        folds.append(FoldResult(
            fold_id=fold_id,
            train_years=train_years,
            test_years=test_years,
            train_loss_final=result.loss_history[-1],
            test_emb_mse=test_emb_mse,
            test_link_auc=test_link_auc,
            layer_weights=result.layer_weights,
            usa_embeddedness=usa_embeddedness,
            china_embeddedness=china_embeddedness,
        ))

    # Build summary
    summary_rows = []
    for fold in folds:
        for test_year in fold.test_years:
            summary_rows.append({
                "fold_id": fold.fold_id,
                "train_end": fold.train_years[-1],
                "test_year": test_year,
                "train_loss": fold.train_loss_final,
                "test_emb_mse": fold.test_emb_mse.get(test_year, np.nan),
                "test_link_auc": fold.test_link_auc.get(test_year, np.nan),
                "usa_embeddedness": fold.usa_embeddedness.get(test_year, np.nan),
                "china_embeddedness": fold.china_embeddedness.get(test_year, np.nan),
                **{f"weight_{k}": v for k, v in fold.layer_weights.items()},
            })

    summary = pd.DataFrame(summary_rows)

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        summary.to_csv(save_path / "backtest_summary.csv", index=False)
        print(f"\nSaved backtest summary to {save_path / 'backtest_summary.csv'}")

    return BacktestResult(folds=folds, summary=summary)


# ---------------------------------------------------------------
# Multi-horizon backtest: evaluate at 5, 10, 15, 20 year horizons
# ---------------------------------------------------------------
HORIZONS = (5, 10, 15, 20)


def multi_horizon_backtest(
    dataset: MultiplexTemporalDataset,
    split_years: List[int],
    horizons: Tuple[int, ...] = HORIZONS,
    model_config: Optional[MultiplexGNNConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    seq_len: int = 5,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Walk-forward backtest with explicit horizon decomposition.

    For each split_year, trains on [start, split_year] and rolls the
    autoregressive predictor forward h years for each h in horizons.
    Reports MSE and link AUC at each (split_year, horizon) cell.

    This replaces the older single-horizon design. The 5/10/15/20 grid
    is what PA expects from a forecasting paper claiming long-horizon
    relevance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_years = dataset.years
    max_horizon = max(horizons)

    if model_config is None:
        model_config = MultiplexGNNConfig(
            num_layers=len(dataset.layer_names),
            in_dim=len(dataset.layer_names),
            hidden_dim=64,
            emb_dim=32,
            seq_len=seq_len,
        )
    if train_config is None:
        train_config = TrainingConfig(
            num_epochs=100, seq_len=seq_len, patience=20, print_every=25,
        )

    rows = []

    for fold_id, split_year in enumerate(split_years):
        train_years = [y for y in all_years if y <= split_year]
        if len(train_years) < seq_len + 1:
            print(f"Skipping split {split_year}: too few train years")
            continue

        print(f"\n--- Fold {fold_id+1}: train ≤ {split_year}, horizons {horizons} ---")

        train_dataset = _subset_dataset(dataset, train_years)
        result = train_model(train_dataset, model_config, train_config, device=device)
        model = result.model
        model.eval()

        # Encode the trailing window from train years
        history: List[torch.Tensor] = []
        with torch.no_grad():
            for y in train_years[-seq_len:]:
                snap = dataset.snapshots[y]
                nf = snap.node_features.to(device)
                ei = {k: v.to(device) for k, v in snap.edge_indices.items()}
                ew = {k: v.to(device) for k, v in snap.edge_weights.items()}
                history.append(model.encode_snapshot(nf, ei, ew, snap.layer_mask))

        # Roll forward year by year and record metrics at horizon checkpoints
        horizon_set = set(horizons)
        usa_idx = dataset.ccode_to_idx.get(USA_CCODE)
        chn_idx = dataset.ccode_to_idx.get(CHN_CCODE)

        with torch.no_grad():
            for h in range(1, max_horizon + 1):
                pred_emb = model.forward_temporal(history[-seq_len:])
                target_year = split_year + h

                # Need the actual snapshot to score against
                if target_year not in dataset.snapshots:
                    history.append(pred_emb)
                    continue

                test_snap = dataset.snapshots[target_year]
                nf = test_snap.node_features.to(device)
                ei = {k: v.to(device) for k, v in test_snap.edge_indices.items()}
                ew = {k: v.to(device) for k, v in test_snap.edge_weights.items()}
                actual_emb = model.encode_snapshot(nf, ei, ew, test_snap.layer_mask)

                if h in horizon_set:
                    mse = F.mse_loss(pred_emb, actual_emb).item()
                    merged_ei = _merge_all_edges(test_snap, device)
                    auc = _approximate_link_auc(pred_emb, merged_ei, dataset.num_nodes)

                    row = {
                        "fold_id": fold_id,
                        "split_year": split_year,
                        "horizon": h,
                        "target_year": target_year,
                        "test_emb_mse": mse,
                        "test_link_auc": auc,
                    }
                    if usa_idx is not None:
                        usa_m = compute_embeddedness_metrics(
                            pred_emb, usa_idx, dataset.ccode_to_idx,
                        )
                        row["usa_mean_cosine_pred"] = usa_m.mean_cosine_similarity
                        row["usa_centroid_dist_pred"] = usa_m.centroid_distance
                    if chn_idx is not None:
                        chn_m = compute_embeddedness_metrics(
                            pred_emb, chn_idx, dataset.ccode_to_idx,
                        )
                        row["chn_mean_cosine_pred"] = chn_m.mean_cosine_similarity
                        row["chn_centroid_dist_pred"] = chn_m.centroid_distance
                    rows.append(row)
                    print(f"  h={h:>2} (yr {target_year}): MSE={mse:.6f}, AUC={auc:.4f}")

                # Slide the window using the prediction (autoregressive rollout)
                history.append(pred_emb)

    df = pd.DataFrame(rows)

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path / "multi_horizon_backtest.csv", index=False)

        # Per-horizon summary
        summary = df.groupby("horizon").agg(
            mean_mse=("test_emb_mse", "mean"),
            std_mse=("test_emb_mse", "std"),
            mean_auc=("test_link_auc", "mean"),
            std_auc=("test_link_auc", "std"),
            n_folds=("fold_id", "nunique"),
        ).reset_index()
        summary.to_csv(save_path / "multi_horizon_summary.csv", index=False)
        print("\n--- Horizon summary ---")
        print(summary.to_string(index=False))

    return df


def _subset_dataset(
    full_dataset: MultiplexTemporalDataset,
    years: List[int],
) -> MultiplexTemporalDataset:
    """Create a dataset subset containing only the specified years."""
    return MultiplexTemporalDataset(
        years=years,
        num_nodes=full_dataset.num_nodes,
        layer_names=full_dataset.layer_names,
        ccode_to_idx=full_dataset.ccode_to_idx,
        idx_to_ccode=full_dataset.idx_to_ccode,
        snapshots={y: full_dataset.snapshots[y] for y in years},
        nodes_df=full_dataset.nodes_df,
        contiguity_map=full_dataset.contiguity_map,
    )


def multi_seed_evaluation(
    dataset: MultiplexTemporalDataset,
    seeds: List[int],
    model_config: Optional[MultiplexGNNConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Train the model with multiple random seeds and report variance.

    Returns a DataFrame with per-seed metrics: final loss, layer weights,
    USA/China embeddedness at the final year.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_config is None:
        model_config = MultiplexGNNConfig(
            num_layers=len(dataset.layer_names),
            in_dim=len(dataset.layer_names),
        )

    rows = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        result = train_model(
            dataset=dataset,
            model_config=model_config,
            train_config=train_config,
            device=device,
        )

        row = {
            "seed": seed,
            "final_loss": result.loss_history[-1],
            "n_epochs": len(result.loss_history),
        }
        for k, v in result.layer_weights.items():
            row[f"weight_{k}"] = v

        # Embeddedness at final year
        final_year = dataset.years[-1]
        if final_year in result.yearly_embeddings:
            emb = result.yearly_embeddings[final_year]
            if USA_CCODE in dataset.ccode_to_idx:
                usa_m = compute_embeddedness_metrics(
                    emb, dataset.ccode_to_idx[USA_CCODE], dataset.ccode_to_idx,
                )
                row["usa_mean_cosine"] = usa_m.mean_cosine_similarity
                row["usa_centroid_dist"] = usa_m.centroid_distance

            if CHN_CCODE in dataset.ccode_to_idx:
                chn_m = compute_embeddedness_metrics(
                    emb, dataset.ccode_to_idx[CHN_CCODE], dataset.ccode_to_idx,
                )
                row["china_mean_cosine"] = chn_m.mean_cosine_similarity
                row["china_centroid_dist"] = chn_m.centroid_distance

        rows.append(row)

    df = pd.DataFrame(rows)

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path / "multi_seed_results.csv", index=False)

    # Print summary
    print(f"\n--- Multi-seed summary ({len(seeds)} seeds) ---")
    for col in df.columns:
        if col == "seed":
            continue
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"  {col}: mean={vals.mean():.6f}, std={vals.std():.6f}")

    return df
