"""Malformed edgelist parquet files must raise RuntimeError, not abort the call.

Opening a file whose umi1 or umi2 column is missing, is not uint64, or contains
a null currently panics inside the worker thread. detach_io resumes that panic
on the calling thread. These tests pass once each graph entry point reports the
column problem as a RuntimeError and leaves no output file behind.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pixelator_core

_GRAPH_CALLS = (
    "find_graph_statistics",
    "run_connected_components",
    "run_label_propagation",
    "run_leiden",
    "run_hybrid_community_detection",
)


def _write(path: Path, table: pa.Table) -> str:
    pq.write_table(table, path)
    return str(path)


def _uint64_pair(left: list[int], right: list[int]) -> pa.Table:
    return pa.table(
        {
            "umi1": pa.array(left, type=pa.uint64()),
            "umi2": pa.array(right, type=pa.uint64()),
        }
    )


def _uint32_umi1(path: Path) -> str:
    return _write(
        path,
        pa.table(
            {
                "umi1": pa.array([1, 2], type=pa.uint32()),
                "umi2": pa.array([3, 4], type=pa.uint64()),
            }
        ),
    )


def _dictionary_umi1(path: Path) -> str:
    encoded = pa.DictionaryArray.from_arrays(
        pa.array([0, 0, 1], type=pa.int32()),
        pa.array([1, 2], type=pa.uint64()),
    )
    return _write(
        path,
        pa.table(
            {
                "umi1": encoded,
                "umi2": pa.array([3, 4, 5], type=pa.uint64()),
            }
        ),
    )


def _missing_umi1(path: Path) -> str:
    return _write(path, pa.table({"umi2": pa.array([1, 2], type=pa.uint64())}))


def _null_umi1(path: Path) -> str:
    return _write(
        path,
        pa.table(
            {
                "umi1": pa.array([1, None], type=pa.uint64()),
                "umi2": pa.array([3, 4], type=pa.uint64()),
            }
        ),
    )


def _null_umi1_after_first_batch(path: Path) -> str:
    rows = 8193
    left = list(range(1, rows + 1))
    left[-1] = None
    right = list(range(10_000, 10_000 + rows))
    return _write(
        path,
        pa.table(
            {
                "umi1": pa.array(left, type=pa.uint64()),
                "umi2": pa.array(right, type=pa.uint64()),
            }
        ),
    )


def _invoke(name: str, parquet_file: str, output: Path) -> None:
    if name == "find_graph_statistics":
        pixelator_core.find_graph_statistics(parquet_file)
        return
    if name == "run_connected_components":
        pixelator_core.run_connected_components(parquet_file, output=str(output))
        return
    if name == "run_label_propagation":
        pixelator_core.run_label_propagation(parquet_file, epochs=1, output=str(output))
        return
    if name == "run_leiden":
        pixelator_core.run_leiden(
            parquet_file,
            resolution=1.0,
            output=str(output),
            seed=1,
        )
        return
    if name == "run_hybrid_community_detection":
        pixelator_core.run_hybrid_community_detection(
            parquet_file,
            resolution=1.0,
            multiplet_recovery=False,
            output=str(output),
            flp_epochs=1,
            randomness=0.1,
            seed=1,
        )
        return
    raise AssertionError(name)


@pytest.mark.parametrize("entry", _GRAPH_CALLS)
@pytest.mark.parametrize(
    "writer",
    (
        _uint32_umi1,
        _dictionary_umi1,
        _missing_umi1,
        _null_umi1,
        _null_umi1_after_first_batch,
    ),
)
def test_graph_entry_rejects_malformed_edgelist(
    entry: str, writer, tmp_path: Path
) -> None:
    parquet_file = writer(tmp_path / "edges.parquet")
    output = tmp_path / "output.parquet"
    with pytest.raises(RuntimeError):
        _invoke(entry, parquet_file, output)
    assert not output.exists()


def test_valid_uint64_edgelist_still_loads(tmp_path: Path) -> None:
    parquet_file = _write(tmp_path / "edges.parquet", _uint64_pair([1, 3], [2, 4]))
    n_nodes, n_edges, n_components, _fraction = pixelator_core.find_graph_statistics(
        parquet_file
    )
    assert n_nodes == 4
    assert n_edges == 2
    assert n_components >= 1
