"""Tests for the isolation analysis module: dual-focal simulation,
multi-edge scenarios, greedy combinatorial search, and utility functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from gnn_forecast.multiplex_data import (
    build_global_node_index,
    build_multiplex_dataset,
)
from gnn_forecast.multiplex_model import MultiplexGNNConfig, MultiplexTemporalGNN
from gnn_forecast.isolation_analysis import (
    dual_focal_simulation,
    multi_edge_simulation,
    greedy_combo_search,
    dual_results_to_dataframe,
    multi_edge_results_to_dataframe,
    classify_quadrants,
    quadrant_summary,
    top_usa_isolating,
    top_china_isolating,
    top_wedge_edges,
    top_pro_usa,
    top_pro_china,
    layer_isolation_heatmap,
    run_isolation_analysis,
)
from gnn_forecast.counterfactual import USA_CCODE, CHN_CCODE


# ---------------------------------------------------------------
# Fixtures: small synthetic data with USA (2) and China (710)
# ---------------------------------------------------------------
def _make_isolation_layers():
    """Synthetic layers containing USA (ccode 2) and China (ccode 710)."""
    layers = {}

    # 5 nodes: USA=2, UK=200, France=220, Germany=365, China=710
    ccodes = [2, 200, 220, 365, 710]

    # Offensive alliances: 3 years
    rows = []
    for year in [2000, 2001, 2002]:
        for s, t in [(2, 200), (2, 220), (200, 365), (710, 365)]:
            rows.append({"year": year, "source_ccode": s, "target_ccode": t, "tie": 1.0})
    layers["offensive_alliances"] = pd.DataFrame(rows)

    # Trade: 3 years (denser connections)
    rows = []
    for year in [2000, 2001, 2002]:
        for s, t, w in [
            (2, 200, 100.0), (2, 220, 80.0), (2, 365, 60.0),
            (710, 200, 50.0), (710, 365, 90.0), (200, 220, 40.0),
        ]:
            rows.append({"year": year, "source_ccode": s, "target_ccode": t, "tie": w})
    layers["trade"] = pd.DataFrame(rows)

    # IGO: 3 years
    rows = []
    for year in [2000, 2001, 2002]:
        for s, t, w in [(2, 200, 5.0), (2, 365, 3.0), (710, 220, 4.0)]:
            rows.append({"year": year, "source_ccode": s, "target_ccode": t, "tie": w})
    layers["igo"] = pd.DataFrame(rows)

    return layers


def _make_isolation_dataset():
    """Build dataset from synthetic isolation data."""
    layers = _make_isolation_layers()
    ccode_to_idx, idx_to_ccode, nodes_df = build_global_node_index(layers)
    dataset = build_multiplex_dataset(
        layer_dfs=layers,
        ccode_to_idx=ccode_to_idx,
        idx_to_ccode=idx_to_ccode,
        nodes_df=nodes_df,
        year_range=(2000, 2002),
    )
    return dataset


def _make_isolation_model(dataset):
    """Build a small model matching the synthetic dataset."""
    config = MultiplexGNNConfig(
        num_layers=len(dataset.layer_names),
        in_dim=len(dataset.layer_names),  # degree features = num layers
        hidden_dim=8,
        emb_dim=4,
        seq_len=3,
    )
    model = MultiplexTemporalGNN(config, dataset.layer_names)
    return model


# ---------------------------------------------------------------
# Tests: dual-focal simulation
# ---------------------------------------------------------------
def test_dual_focal_returns_results():
    """Dual-focal simulation should return a non-empty list of results."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    results = dual_focal_simulation(
        model, dataset, seq_len=3, focal_year=2002,
    )
    assert len(results) > 0


def test_dual_focal_covers_both_operations():
    """Should test both 'add' and 'remove' for each partner-layer pair."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    results = dual_focal_simulation(
        model, dataset, seq_len=3, focal_year=2002,
    )
    operations = {r.operation for r in results}
    assert "add" in operations
    assert "remove" in operations


def test_dual_focal_excludes_usa_and_china():
    """Partner ccodes should never be USA (2) or China (710)."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    results = dual_focal_simulation(
        model, dataset, seq_len=3, focal_year=2002,
    )
    partner_ccodes = {r.partner_ccode for r in results}
    assert USA_CCODE not in partner_ccodes
    assert CHN_CCODE not in partner_ccodes


def test_dual_focal_specific_partners():
    """Should respect the partner_ccodes filter."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    results = dual_focal_simulation(
        model, dataset, partner_ccodes=[200], seq_len=3,
    )
    partner_ccodes = {r.partner_ccode for r in results}
    assert partner_ccodes == {200}


def test_dual_focal_result_fields():
    """Each result should have the expected metric fields."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    results = dual_focal_simulation(model, dataset, seq_len=3)
    r = results[0]

    assert isinstance(r.usa_delta_cosine, float)
    assert isinstance(r.chn_delta_cosine, float)
    assert isinstance(r.wedge_cosine, float)
    assert isinstance(r.usa_delta_centroid, float)
    assert isinstance(r.usa_delta_mp_dist, float)
    assert isinstance(r.usa_delta_knn, float)
    # Wedge should be the difference
    assert abs(r.wedge_cosine - (r.usa_delta_cosine - r.chn_delta_cosine)) < 1e-6


# ---------------------------------------------------------------
# Tests: DataFrame conversion and classification
# ---------------------------------------------------------------
def test_dual_results_to_dataframe():
    """DataFrame should have the expected columns."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    results = dual_focal_simulation(model, dataset, seq_len=3)
    df = dual_results_to_dataframe(results)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(results)
    expected_cols = [
        "partner_ccode", "layer", "operation",
        "usa_delta_cosine", "chn_delta_cosine", "wedge_cosine",
    ]
    for col in expected_cols:
        assert col in df.columns


def test_classify_quadrants():
    """Quadrant classification should cover the 4 cases."""
    df = pd.DataFrame({
        "partner_ccode": [200, 200, 365, 365],
        "layer": ["trade"] * 4,
        "operation": ["remove"] * 4,
        "usa_delta_cosine": [0.1, -0.1, -0.1, 0.1],
        "chn_delta_cosine": [0.1, 0.1, -0.1, -0.1],
    })
    result = classify_quadrants(df)
    assert "quadrant" in result.columns
    quads = set(result["quadrant"])
    assert "Q1: Both embedded" in quads
    assert "Q2: USA isolated, China embedded" in quads
    assert "Q3: Both isolated" in quads
    assert "Q4: USA embedded, China isolated" in quads


def test_quadrant_summary():
    """Summary should aggregate by quadrant, layer, operation."""
    df = pd.DataFrame({
        "partner_ccode": [200, 220, 365],
        "layer": ["trade", "trade", "igo"],
        "operation": ["remove", "remove", "add"],
        "usa_delta_cosine": [-0.1, -0.2, 0.1],
        "chn_delta_cosine": [0.1, 0.05, -0.1],
        "wedge_cosine": [-0.2, -0.25, 0.2],
        "abs_wedge": [0.2, 0.25, 0.2],
    })
    df = classify_quadrants(df)
    summary = quadrant_summary(df)
    assert "quadrant" in summary.columns
    assert "n" in summary.columns
    assert len(summary) > 0


# ---------------------------------------------------------------
# Tests: ranking functions
# ---------------------------------------------------------------
def test_top_ranking_functions():
    """Top-N functions should return the right number of rows."""
    n = 3
    df = pd.DataFrame({
        "partner_ccode": list(range(10)),
        "layer": ["trade"] * 10,
        "operation": ["remove"] * 10,
        "usa_delta_cosine": np.random.randn(10),
        "chn_delta_cosine": np.random.randn(10),
        "wedge_cosine": np.random.randn(10),
        "abs_wedge": np.abs(np.random.randn(10)),
    })

    assert len(top_usa_isolating(df, n)) == n
    assert len(top_china_isolating(df, n)) == n
    assert len(top_wedge_edges(df, n)) == n
    assert len(top_pro_usa(df, n)) == n
    assert len(top_pro_china(df, n)) == n


def test_top_usa_isolating_order():
    """Most isolating = most negative usa_delta_cosine."""
    df = pd.DataFrame({
        "partner_ccode": [1, 2, 3],
        "layer": ["trade"] * 3,
        "operation": ["remove"] * 3,
        "usa_delta_cosine": [-0.5, -0.1, -0.9],
        "chn_delta_cosine": [0.0, 0.0, 0.0],
    })
    result = top_usa_isolating(df, 2)
    assert result.iloc[0]["usa_delta_cosine"] == -0.9
    assert result.iloc[1]["usa_delta_cosine"] == -0.5


# ---------------------------------------------------------------
# Tests: layer heatmap
# ---------------------------------------------------------------
def test_layer_isolation_heatmap():
    """Heatmap aggregation should produce per-layer-operation stats."""
    df = pd.DataFrame({
        "partner_ccode": [200, 220, 200, 220],
        "layer": ["trade", "trade", "igo", "igo"],
        "operation": ["remove", "remove", "add", "add"],
        "usa_delta_cosine": [-0.1, -0.2, 0.05, 0.1],
        "chn_delta_cosine": [0.05, 0.1, -0.05, -0.1],
        "wedge_cosine": [-0.15, -0.3, 0.1, 0.2],
    })
    hm = layer_isolation_heatmap(df)
    assert "layer" in hm.columns
    assert "mean_usa_delta" in hm.columns
    assert "n" in hm.columns
    assert len(hm) == 2  # trade/remove and igo/add


# ---------------------------------------------------------------
# Tests: multi-edge simulation
# ---------------------------------------------------------------
def test_multi_edge_simulation():
    """Named scenario bundles should produce results."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    scenarios = {
        "US loses UK trade + alliance": [
            (200, "trade", "remove"),
            (200, "offensive_alliances", "remove"),
        ],
        "China gains France IGO": [
            (220, "igo", "add"),
        ],
    }

    results = multi_edge_simulation(
        model, dataset, scenarios, seq_len=3,
    )
    assert len(results) == 2
    labels = {r.label for r in results}
    assert "US loses UK trade + alliance" in labels
    assert "China gains France IGO" in labels


def test_multi_edge_result_fields():
    """Multi-edge results should have the expected fields."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    scenarios = {
        "test scenario": [(200, "trade", "remove")],
    }
    results = multi_edge_simulation(model, dataset, scenarios, seq_len=3)
    r = results[0]

    assert isinstance(r.usa_delta_cosine, float)
    assert isinstance(r.chn_delta_cosine, float)
    assert isinstance(r.wedge_cosine, float)
    assert abs(r.wedge_cosine - (r.usa_delta_cosine - r.chn_delta_cosine)) < 1e-6


def test_multi_edge_results_to_dataframe():
    """DataFrame conversion should preserve all scenarios."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    scenarios = {"s1": [(200, "trade", "remove")], "s2": [(365, "igo", "add")]}
    results = multi_edge_simulation(model, dataset, scenarios, seq_len=3)
    df = multi_edge_results_to_dataframe(results)

    assert len(df) == 2
    assert "scenario" in df.columns
    assert set(df["scenario"]) == {"s1", "s2"}


# ---------------------------------------------------------------
# Tests: greedy combinatorial search
# ---------------------------------------------------------------
def test_greedy_combo_search_returns_results():
    """Greedy search should return DataFrames for each objective."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    # First get single-edge results
    results = dual_focal_simulation(model, dataset, seq_len=3)
    df = dual_results_to_dataframe(results)

    combo_results = greedy_combo_search(
        model, dataset, single_edge_df=df,
        top_k=3, max_combo_size=2, seq_len=3,
    )

    assert isinstance(combo_results, dict)
    assert len(combo_results) > 0
    for obj_name, combo_df in combo_results.items():
        assert isinstance(combo_df, pd.DataFrame)


def test_greedy_combo_has_interaction_columns():
    """Combo DataFrames should include interaction effect columns."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    results = dual_focal_simulation(model, dataset, seq_len=3)
    df = dual_results_to_dataframe(results)

    combo_results = greedy_combo_search(
        model, dataset, single_edge_df=df,
        top_k=3, max_combo_size=2, seq_len=3,
    )

    for obj_name, combo_df in combo_results.items():
        if not combo_df.empty:
            expected = [
                "combo_size", "label", "edges",
                "usa_delta_cosine", "chn_delta_cosine", "wedge_cosine",
                "usa_interaction", "chn_interaction",
            ]
            for col in expected:
                assert col in combo_df.columns, f"Missing {col} in {obj_name}"


def test_greedy_combo_all_objectives():
    """Default objectives should produce 4 result sets."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    results = dual_focal_simulation(model, dataset, seq_len=3)
    df = dual_results_to_dataframe(results)

    combo_results = greedy_combo_search(
        model, dataset, single_edge_df=df,
        top_k=3, max_combo_size=2, seq_len=3,
    )

    expected_objectives = {
        "usa_isolating", "china_isolating",
        "pro_usa_wedge", "pro_china_wedge",
    }
    assert set(combo_results.keys()) == expected_objectives


def test_greedy_combo_size_constraint():
    """All combos should respect the max_combo_size parameter."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    results = dual_focal_simulation(model, dataset, seq_len=3)
    df = dual_results_to_dataframe(results)

    max_size = 2
    combo_results = greedy_combo_search(
        model, dataset, single_edge_df=df,
        top_k=3, max_combo_size=max_size, seq_len=3,
    )

    for obj_name, combo_df in combo_results.items():
        if not combo_df.empty:
            assert combo_df["combo_size"].max() <= max_size


def test_greedy_combo_custom_objectives():
    """Should respect a custom subset of objectives."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    results = dual_focal_simulation(model, dataset, seq_len=3)
    df = dual_results_to_dataframe(results)

    combo_results = greedy_combo_search(
        model, dataset, single_edge_df=df,
        objectives=["usa_isolating"],
        top_k=3, max_combo_size=2, seq_len=3,
    )

    assert set(combo_results.keys()) == {"usa_isolating"}


def test_greedy_combo_interaction_math():
    """Interaction = actual combo effect - sum of individual effects."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    results = dual_focal_simulation(model, dataset, seq_len=3)
    df = dual_results_to_dataframe(results)

    combo_results = greedy_combo_search(
        model, dataset, single_edge_df=df,
        top_k=3, max_combo_size=2, seq_len=3,
    )

    # Build single-edge lookup
    single_effects = {}
    for _, row in df.iterrows():
        key = (int(row["partner_ccode"]), row["layer"], row["operation"])
        single_effects[key] = (row["usa_delta_cosine"], row["chn_delta_cosine"])

    for obj_name, combo_df in combo_results.items():
        for _, row in combo_df.iterrows():
            edges = eval(row["edges"])  # list of tuples
            sum_usa = sum(single_effects.get(e, (0, 0))[0] for e in edges)
            sum_chn = sum(single_effects.get(e, (0, 0))[1] for e in edges)

            expected_usa_inter = row["usa_delta_cosine"] - sum_usa
            expected_chn_inter = row["chn_delta_cosine"] - sum_chn

            assert abs(row["usa_interaction"] - expected_usa_inter) < 1e-5, \
                f"USA interaction mismatch for {obj_name}: {row['label']}"
            assert abs(row["chn_interaction"] - expected_chn_inter) < 1e-5, \
                f"CHN interaction mismatch for {obj_name}: {row['label']}"


# ---------------------------------------------------------------
# Tests: full pipeline (run_isolation_analysis)
# ---------------------------------------------------------------
def test_run_isolation_analysis_outputs(tmp_path):
    """Full pipeline should produce expected output files."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    outputs = run_isolation_analysis(
        model=model,
        dataset=dataset,
        out_dir=tmp_path / "isolation_test",
        seq_len=3,
        combo_top_k=3,
        combo_max_size=2,
    )

    assert isinstance(outputs, dict)

    # Core tables
    assert "dual_focal_all" in outputs
    assert "quadrant_summary" in outputs
    assert "top_usa_isolating" in outputs
    assert "top_china_isolating" in outputs
    assert "top_wedge_edges" in outputs

    # Combo tables
    assert "combos_usa_isolating" in outputs
    assert "combos_china_isolating" in outputs

    # Files on disk
    out_dir = tmp_path / "isolation_test"
    assert (out_dir / "dual_focal_all.csv").exists()
    assert (out_dir / "quadrant_summary.csv").exists()
    assert (out_dir / "combos").is_dir()


def test_run_isolation_analysis_skip_combos(tmp_path):
    """Pipeline with skip_combos=True should omit combo outputs."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    outputs = run_isolation_analysis(
        model=model,
        dataset=dataset,
        out_dir=tmp_path / "no_combos",
        seq_len=3,
        skip_combos=True,
    )

    assert "dual_focal_all" in outputs
    assert "combos_usa_isolating" not in outputs
    assert not (tmp_path / "no_combos" / "combos").exists()


def test_run_isolation_analysis_with_scenarios(tmp_path):
    """Pipeline with multi-edge scenarios should produce scenario output."""
    dataset = _make_isolation_dataset()
    model = _make_isolation_model(dataset)

    scenarios = {
        "test scenario": [(200, "trade", "remove")],
    }

    outputs = run_isolation_analysis(
        model=model,
        dataset=dataset,
        out_dir=tmp_path / "with_scenarios",
        seq_len=3,
        multi_edge_scenarios=scenarios,
        skip_combos=True,
    )

    assert "multi_edge" in outputs
    assert (tmp_path / "with_scenarios" / "multi_edge_scenarios.csv").exists()
