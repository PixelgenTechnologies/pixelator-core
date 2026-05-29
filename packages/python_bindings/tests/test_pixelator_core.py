"""Smoke tests for the pixelator_core extension (built with maturin)."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pyarrow.parquet as pq
import pytest

import pixelator_core

# Repo layout: packages/python_bindings/tests/ -> packages/pixelator_core/test_data/
_TEST_DATA = Path(__file__).resolve().parents[2] / "pixelator_core" / "test_data"
SMALL_PARQUET = _TEST_DATA / "mix_40cells_1pc_1000rows.parquet"


def _assert_partition_of_nodes(communities: object, nodes: set[object]) -> None:
    assert isinstance(communities, list)
    assert all(isinstance(community, set) for community in communities)
    assert all(community for community in communities)

    flattened: set[object] = set().union(*communities) if communities else set()
    assert flattened == nodes

    visited: set[object] = set()
    for community in communities:
        assert visited.isdisjoint(community)
        visited.update(community)


@pytest.fixture(scope="module")
def small_parquet_path() -> str:
    assert SMALL_PARQUET.is_file(), f"missing test fixture: {SMALL_PARQUET}"
    return str(SMALL_PARQUET)


def test_public_api() -> None:
    for name in (
        "find_graph_statistics",
        "run_label_propagation",
        "run_label_propagation_networkx",
        "run_leiden",
        "run_leiden_networkx",
        "run_hybrid_community_detection",
        "PyGraphProperties",
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
    n_nodes, *_ = pixelator_core.find_graph_statistics(small_parquet_path)
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
    assert pq.read_metadata(out).num_rows == n_nodes


def test_run_leiden_with_absolute_merge_threshold(
    small_parquet_path: str, tmp_path: Path
) -> None:
    n_nodes, *_ = pixelator_core.find_graph_statistics(small_parquet_path)
    out = tmp_path / "leiden_partitions_absolute_threshold.parquet"
    n_partitions, quality = pixelator_core.run_leiden(
        small_parquet_path,
        max_iteration=None,
        partition=None,
        resolution=1.0,
        output=str(out),
        randomness=0.1,
        seed=42,
        merge_edge_threshold=10,
    )
    assert out.is_file()
    assert isinstance(n_partitions, int) and n_partitions >= 1
    assert isinstance(quality, float)
    assert pq.read_metadata(out).num_rows == n_nodes


def test_run_leiden_with_relative_merge_threshold(
    small_parquet_path: str, tmp_path: Path
) -> None:
    n_nodes, *_ = pixelator_core.find_graph_statistics(small_parquet_path)
    out = tmp_path / "leiden_partitions_relative_threshold.parquet"
    n_partitions, quality = pixelator_core.run_leiden(
        small_parquet_path,
        max_iteration=None,
        partition=None,
        resolution=1.0,
        output=str(out),
        randomness=0.1,
        seed=42,
        merge_edge_threshold_relative=0.1,
    )
    assert out.is_file()
    assert isinstance(n_partitions, int) and n_partitions >= 1
    assert isinstance(quality, float)
    assert pq.read_metadata(out).num_rows == n_nodes


def test_run_leiden_with_both_merge_thresholds_raises(
    small_parquet_path: str, tmp_path: Path
) -> None:
    out = tmp_path / "leiden_partitions_invalid_thresholds.parquet"
    with pytest.raises(
        ValueError,
        match="`merge_edge_threshold` and `merge_edge_threshold_relative` cannot both be used at the same time",
    ):
        pixelator_core.run_leiden(
            small_parquet_path,
            max_iteration=None,
            partition=None,
            resolution=1.0,
            output=str(out),
            randomness=0.1,
            seed=42,
            merge_edge_threshold=10,
            merge_edge_threshold_relative=0.1,
        )


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


def test_run_leiden_networkx_partition_shape_and_labels() -> None:
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=2)
    graph.add_edge("B", "C")
    graph.add_edge(("tuple", 1), "isolated-link")
    graph.add_node("isolated")

    communities = pixelator_core.run_leiden_networkx(
        graph,
        resolution=1.0,
        randomness=0.1,
        seed=42,
        max_iteration=None,
        partition=None,
        merge_edge_threshold=None,
        merge_edge_threshold_relative=None,
    )

    _assert_partition_of_nodes(communities, set(graph.nodes()))


def test_run_leiden_networkx_deterministic_with_seed() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4)])

    first = pixelator_core.run_leiden_networkx(
        graph,
        resolution=1.0,
        randomness=0.1,
        seed=123,
        max_iteration=None,
        partition=None,
        merge_edge_threshold=None,
        merge_edge_threshold_relative=None,
    )
    second = pixelator_core.run_leiden_networkx(
        graph,
        resolution=1.0,
        randomness=0.1,
        seed=123,
        max_iteration=None,
        partition=None,
        merge_edge_threshold=None,
        merge_edge_threshold_relative=None,
    )
    assert {frozenset(s) for s in first} == {frozenset(s) for s in second}


def test_run_label_propagation_networkx_partition_shape() -> None:
    graph = nx.Graph()
    graph.add_edge("x", "y")
    graph.add_edge("y", "z", weight=3)
    graph.add_node("w")

    communities = pixelator_core.run_label_propagation_networkx(graph, epochs=2)

    _assert_partition_of_nodes(communities, set(graph.nodes()))


def test_networkx_api_rejects_non_graph_objects() -> None:
    with pytest.raises(TypeError, match="NetworkX"):
        pixelator_core.run_leiden_networkx(
            object(),
            resolution=1.0,
            randomness=0.1,
            seed=1,
            max_iteration=None,
            partition=None,
            merge_edge_threshold=None,
            merge_edge_threshold_relative=None,
        )

    with pytest.raises(TypeError, match="NetworkX"):
        pixelator_core.run_label_propagation_networkx(object(), epochs=1)


def test_run_label_propagation_networkx_empty_graph() -> None:
    graph = nx.Graph()
    communities = pixelator_core.run_label_propagation_networkx(graph, epochs=1)
    assert communities == []


def test_run_leiden_networkx_with_both_merge_thresholds_raises() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1)
    with pytest.raises(
        ValueError,
        match="`merge_edge_threshold` and `merge_edge_threshold_relative` cannot both be used at the same time",
    ):
        pixelator_core.run_leiden_networkx(
            graph,
            resolution=1.0,
            randomness=0.1,
            seed=1,
            max_iteration=None,
            partition=None,
            merge_edge_threshold=2,
            merge_edge_threshold_relative=0.2,
        )


def test_run_leiden_networkx_partition_length_mismatch_raises() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])
    with pytest.raises(
        ValueError,
        match="Length of `partition` must match the number of nodes in the NetworkX graph",
    ):
        pixelator_core.run_leiden_networkx(
            graph,
            resolution=1.0,
            randomness=0.1,
            seed=1,
            max_iteration=None,
            partition=[0, 1],
            merge_edge_threshold=None,
            merge_edge_threshold_relative=None,
        )


def test_run_leiden_networkx_rejects_fractional_weights() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=0.5)
    with pytest.raises(TypeError, match="non-negative integer-like"):
        pixelator_core.run_leiden_networkx(
            graph,
            resolution=1.0,
            randomness=0.1,
            seed=1,
            max_iteration=None,
            partition=None,
            merge_edge_threshold=None,
            merge_edge_threshold_relative=None,
        )


def test_run_label_propagation_networkx_rejects_fractional_weights() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=0.5)
    with pytest.raises(TypeError, match="non-negative integer-like"):
        pixelator_core.run_label_propagation_networkx(graph, epochs=1)


def test_run_label_propagation_networkx_rejects_weight_above_u8() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=256)
    with pytest.raises(ValueError, match="exceeds FLP u8 weight range"):
        pixelator_core.run_label_propagation_networkx(graph, epochs=1)
