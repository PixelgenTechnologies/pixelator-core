"""Duplicate UMI-pair counts must survive graph construction.

The statistics helper and hybrid community detection build an 8-bit graph from
the Parquet edgelist the caller supplies. Release builds wrap those counts, so
the published edge-weight total and the Leiden resolution no longer match the
file. These tests are the expected behavior and fail until that wrap is gone.

Run them against a release build (debug builds abort on the same overflow):

    cargo build -p community-detection --release
    cd packages/python_bindings
    uv sync --extra dev --no-install-project
    uv run maturin develop --release
    uv run pytest tests/test_duplicate_edge_weight.py -v
"""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pixelator_core

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "target" / "release" / "community-detection"


def _write_pairs(path: Path, pairs: list[tuple[int, int]]) -> None:
    pq.write_table(
        pa.table(
            {
                "umi1": pa.array([src for src, _ in pairs], type=pa.uint64()),
                "umi2": pa.array([dst for _, dst in pairs], type=pa.uint64()),
            }
        ),
        path,
    )


def _edge_weight_sum(path: Path) -> int:
    _nodes, edge_weight_sum, _components, _fraction = pixelator_core.find_graph_statistics(
        str(path)
    )
    return edge_weight_sum


def test_statistics_count_one_pair_repeated_256_times(tmp_path: Path) -> None:
    """256 copies of one pair are a count of 256, not a wrapped total of 0.

    A wrapped total of zero is what makes the modularity scale infinite.
    """
    path = tmp_path / "repeated_256.parquet"
    _write_pairs(path, [(10, 20)] * 256)

    assert _edge_weight_sum(path) == 256


def test_statistics_count_a_pair_repeated_in_both_directions(tmp_path: Path) -> None:
    """The two directions of one undirected pair add in a wide integer."""
    path = tmp_path / "both_directions.parquet"
    pairs = [(10, 20)] * 200 + [(20, 10)] * 100
    _write_pairs(path, pairs)

    assert _edge_weight_sum(path) == len(pairs)


def test_statistics_sum_every_pair_past_the_byte_boundary(tmp_path: Path) -> None:
    """Distinct pairs that each repeat past 255 still contribute their full counts."""
    path = tmp_path / "several_pairs.parquet"
    pairs: list[tuple[int, int]] = []
    for src in (1, 3, 5, 7):
        pairs.extend([(src, src + 1)] * 300)
    _write_pairs(path, pairs)

    assert _edge_weight_sum(path) == len(pairs)


def test_hybrid_communities_follow_the_unwrapped_edgelist(tmp_path: Path) -> None:
    """Hybrid detection must keep the bridge weight and publish the real total.

    Two triangles joined by one pair repeated 256 times are a single community
    when that bridge keeps its count. Wrapping the bridge to 0 splits them, and
    the filtered edgelist then drops the bridge.
    """
    triangle = [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)]
    pairs = triangle + [(3, 4)] * 256
    edgelist = tmp_path / "bridged.parquet"
    _write_pairs(edgelist, pairs)

    hybrid_out = tmp_path / "hybrid.parquet"
    _returned, pre, _post_flp, post_recovery = pixelator_core.run_hybrid_community_detection(
        str(edgelist),
        resolution=1.0,
        multiplet_recovery=True,
        output=str(hybrid_out),
        flp_epochs=1,
        randomness=0.0,
        seed=0,
        max_iteration=20,
    )
    kept = pq.read_table(hybrid_out)
    assert kept.num_rows == len(pairs)
    assert len(set(kept.column("component").to_pylist())) == 1
    assert pre.edge_weight_sum == len(pairs)
    assert post_recovery.edge_weight_sum == len(pairs)

    leiden_out = tmp_path / "leiden.parquet"
    n_partitions, quality = pixelator_core.run_leiden(
        str(edgelist),
        resolution=1.0,
        max_iteration=20,
        output=str(leiden_out),
        randomness=0.0,
        seed=0,
    )
    assert n_partitions == 1
    assert math.isfinite(quality)


def test_single_repeated_pair_keeps_a_finite_hybrid_total(tmp_path: Path) -> None:
    """One pair repeated 256 times must not publish a zero edge-weight total."""
    path = tmp_path / "repeated_256.parquet"
    _write_pairs(path, [(10, 20)] * 256)
    out = tmp_path / "hybrid.parquet"

    _returned, pre, _post_flp, post_recovery = pixelator_core.run_hybrid_community_detection(
        str(path),
        resolution=1.0,
        multiplet_recovery=True,
        output=str(out),
        flp_epochs=1,
        randomness=0.0,
        seed=0,
        max_iteration=20,
    )

    assert pre.edge_weight_sum == 256
    assert post_recovery.edge_weight_sum == 256
    assert pq.read_table(out).num_rows == 256


def test_cli_stats_reports_the_unwrapped_duplicate_count(tmp_path: Path) -> None:
    """The stats command prints the same total the statistics helper returns."""
    if not _CLI.is_file():
        pytest.fail(f"release CLI is missing at {_CLI}; build community-detection --release first")

    path = tmp_path / "repeated_256.parquet"
    _write_pairs(path, [(10, 20)] * 256)
    env = os.environ.copy()
    env["RUST_LOG"] = "info"
    completed = subprocess.run(
        [str(_CLI), "stats", str(path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Total edge weight (undirected): 256" in completed.stderr
