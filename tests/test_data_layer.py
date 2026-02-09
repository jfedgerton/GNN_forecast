from __future__ import annotations

from pathlib import Path

from gnn_forecast.data_layer import CanonicalDataConfig, load_canonical_multiplex_data


FIXTURE_DIR = Path("tests/fixtures/tiny_processed")


def test_node_indexing_stable() -> None:
    config = CanonicalDataConfig(data_dir=str(FIXTURE_DIR))
    first = load_canonical_multiplex_data(config)
    second = load_canonical_multiplex_data(config)

    assert first.ccode_to_idx == {2: 0, 710: 1, 840: 2}
    assert first.ccode_to_idx == second.ccode_to_idx
    assert first.idx_to_ccode == {0: 2, 1: 710, 2: 840}


def test_year_loading_and_edge_columns() -> None:
    config = CanonicalDataConfig(data_dir=str(FIXTURE_DIR))
    data = load_canonical_multiplex_data(config, years=[2000, 2001])

    assert sorted(data.yearly_edges.keys()) == [2000, 2001]
    for year in [2000, 2001]:
        assert set(data.yearly_edges[year].keys()) == {"alliance", "igo", "trade"}
        for relation in ["alliance", "igo", "trade"]:
            frame = data.yearly_edges[year][relation]
            assert {"source_ccode", "target_ccode", "tie", "source_idx", "target_idx"}.issubset(frame.columns)
