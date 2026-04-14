"""Smoke tests for the pixelator_core extension (built with maturin)."""

from __future__ import annotations

from pathlib import Path

import pytest

import pixelator_core

# Repo layout: packages/python_bindings/tests/ -> packages/pixelator_core/test_data/
_TEST_DATA = Path(__file__).resolve().parents[2] / "pixelator_core" / "test_data"
SMALL_PARQUET = _TEST_DATA / "mix_40cells_1pc_1000rows.parquet"


@pytest.fixture(scope="module")
def small_parquet_path() -> str:
    assert SMALL_PARQUET.is_file(), f"missing test fixture: {SMALL_PARQUET}"
    return str(SMALL_PARQUET)


def test_public_api() -> None:
    for name in (
        "find_graph_statistics",
        "run_label_propagation",
        "run_leiden",
        "run_hybrid_community_detection",
        "__version__",
    ):
        assert hasattr(pixelator_core, name), f"missing export: {name!r}"


def test_find_graph_statistics(small_parquet_path: str) -> None:
    n_nodes, n_edges, n_components, frac_largest = pixelator_core.find_graph_statistics(
        small_parquet_path
    )
    assert isinstance(n_nodes, int) and n_nodes > 0
    assert isinstance(n_edges, int) and n_edges > 0
    assert isinstance(n_components, int) and n_components >= 1
    assert isinstance(frac_largest, float)
    assert 0.0 < frac_largest <= 1.0


def test_run_leiden(small_parquet_path: str, tmp_path: Path) -> None:
    out = tmp_path / "leiden_partitions.parquet"
    n_partitions, quality = pixelator_core.run_leiden(
        small_parquet_path,
        max_iteration=None,
        partition=None,
        resolution=1.0,
        output=str(out),
        randomness=0.1,
        seed=42,
    )
    assert out.is_file()
    assert isinstance(n_partitions, int) and n_partitions >= 1
    assert isinstance(quality, float)


def test_run_label_propagation(small_parquet_path: str, tmp_path: Path) -> None:
    out = tmp_path / "flp_partitions.parquet"
    n_partitions = pixelator_core.run_label_propagation(
        small_parquet_path,
        epochs=1,
        output=str(out),
    )
    assert out.is_file()
    assert isinstance(n_partitions, int) and n_partitions >= 1


def test_run_hybrid_community_detection(small_parquet_path: str, tmp_path: Path) -> None:
    out = tmp_path / "hybrid_edges.parquet"
    path_returned, pre_stats, post_flp, post_recovery = (
        pixelator_core.run_hybrid_community_detection(
            small_parquet_path,
            resolution=1.0,
            output=str(out),
            flp_epochs=1,
            randomness=0.1,
            seed=42,
            max_iteration=None,
            multiplet_recovery=False,
        )
    )
    assert path_returned == str(out)
    assert out.is_file()
    for stats in (pre_stats, post_flp, post_recovery):
        assert stats.node_count > 0
        assert stats.edge_weight_sum > 0
        assert stats.n_connected_components >= 1
        assert 0.0 < stats.fraction_in_largest_component <= 1.0
