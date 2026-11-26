"""Smoke tests for the pixelator_core extension (built with maturin)."""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable
from pathlib import Path

import pixelator_core
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

# Repo layout: packages/python_bindings/tests/ -> packages/pixelator_core/test_data/
_TEST_DATA = Path(__file__).resolve().parents[2] / "pixelator_core" / "test_data"
SMALL_PARQUET = _TEST_DATA / "mix_40cells_1pc_1000rows.parquet"


@pytest.fixture(scope="module")
def small_parquet_path() -> str:
    assert SMALL_PARQUET.is_file(), f"missing test fixture: {SMALL_PARQUET}"
    return str(SMALL_PARQUET)


@pytest.fixture(scope="module")
def gil_check_parquet_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Synthesize a medium-sized parquet file once per module for GIL release tests."""
    n_rows = 200_000
    synthetic_parquet = tmp_path_factory.mktemp("gil_check") / "synthetic_edges.parquet"
    pq.write_table(
        pa.table(
            {
                "umi1": pa.array(range(n_rows), type=pa.uint64()),
                "umi2": pa.array(range(1, n_rows + 1), type=pa.uint64()),
            }
        ),
        synthetic_parquet,
    )
    return str(synthetic_parquet)


@pytest.fixture(scope="module")
def interrupt_parquet_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Synthesize a graph that takes Leiden and FLP at least a second to process."""
    n_nodes = 1_000_000
    umi1 = pa.array(range(n_nodes), type=pa.uint64())
    neighbors = [pc.add(umi1, pa.scalar(k, pa.uint64())) for k in (1, 7, 101)]
    synthetic_parquet = tmp_path_factory.mktemp("interrupt") / "synthetic_edges.parquet"
    pq.write_table(
        pa.table(
            {
                "umi1": pa.concat_arrays([umi1] * len(neighbors)),
                "umi2": pa.concat_arrays(neighbors),
            }
        ),
        synthetic_parquet,
    )
    return str(synthetic_parquet)


def test_public_api() -> None:
    for name in (
        "find_graph_statistics",
        "run_connected_components",
        "run_label_propagation",
        "run_leiden",
        "run_hybrid_community_detection",
        "run_edgelist",
        "encode_dna_2bit",
        "decode_dna_2bit",
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


def test_run_connected_components(small_parquet_path: str, tmp_path: Path) -> None:
    n_nodes, _, n_components, _ = pixelator_core.find_graph_statistics(
        small_parquet_path
    )
    out = tmp_path / "connected_components.parquet"
    n_partitions = pixelator_core.run_connected_components(
        small_parquet_path,
        output=str(out),
    )
    assert out.is_file()
    assert n_partitions == n_components
    assert pq.read_metadata(out).num_rows == n_nodes


def test_run_label_propagation(small_parquet_path: str, tmp_path: Path) -> None:
    out = tmp_path / "flp_partitions.parquet"
    n_partitions = pixelator_core.run_label_propagation(
        small_parquet_path,
        epochs=1,
        output=str(out),
    )
    assert out.is_file()
    assert isinstance(n_partitions, int) and n_partitions >= 1


def test_run_hybrid_community_detection(
    small_parquet_path: str, tmp_path: Path
) -> None:
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


class _GilCheckHandler(logging.Handler):
    """Logging handler invoked by pyo3-log from inside the Rust extension."""

    def __init__(self, on_emit: Callable[[logging.LogRecord], None]) -> None:
        super().__init__(level=logging.DEBUG)
        self._on_emit = on_emit

    def emit(self, record: logging.LogRecord) -> None:
        self._on_emit(record)


@pytest.mark.parametrize(
    ("name", "invoke"),
    [
        (
            "find_graph_statistics",
            lambda parquet_file, _tmp_path: pixelator_core.find_graph_statistics(
                parquet_file
            ),
        ),
        (
            "run_connected_components",
            lambda parquet_file, tmp_path: pixelator_core.run_connected_components(
                parquet_file,
                output=str(tmp_path / "cc.parquet"),
            ),
        ),
        (
            "run_label_propagation",
            lambda parquet_file, tmp_path: pixelator_core.run_label_propagation(
                parquet_file,
                epochs=1,
                output=str(tmp_path / "flp.parquet"),
            ),
        ),
        (
            "run_leiden",
            lambda parquet_file, tmp_path: pixelator_core.run_leiden(
                parquet_file,
                resolution=1.0,
                output=str(tmp_path / "leiden.parquet"),
                seed=42,
            ),
        ),
        (
            "run_hybrid_community_detection",
            lambda parquet_file, tmp_path: (
                pixelator_core.run_hybrid_community_detection(
                    parquet_file,
                    resolution=1.0,
                    multiplet_recovery=True,
                    output=str(tmp_path / "hybrid.parquet"),
                    flp_epochs=1,
                    randomness=0.1,
                    seed=42,
                )
            ),
        ),
    ],
)
def test_pyfunctions_release_gil_during_rust_execution(
    name: str,
    invoke: Callable[[str, Path], object],
    gil_check_parquet_path: str,
    tmp_path: Path,
) -> None:
    """Verify each binding detaches from the GIL while executing pure-Rust steps.

    Inside `detach_io`, `py.detach(...)` emits `"Entering detached Rust execution"`
    before running the pure-Rust closure and `"Exiting detached Rust execution"`
    before returning. `pyo3-log` temporarily re-acquires the GIL to forward each
    `debug!` record to Python's `logging` module and releases the GIL again as
    soon as `Handler.emit` returns.

    On `"Entering detached Rust execution"`, we signal a pre-started background
    Python thread so it contends for the GIL while the main thread is executing
    pure-Rust code, and on `"Exiting detached Rust execution"`, we assert that
    the background thread has already run.
    """
    worker_ran_between_logs = threading.Event()
    worker_ready = threading.Event()
    first_emit_returned = threading.Event()
    worker_ran_before_exit_log = False
    saw_enter_log = False
    saw_exit_log = False

    def background_worker() -> None:
        worker_ready.set()
        # Wait until the entering-detach logging handler call is about to return
        # so that we contend for the GIL while the main thread is inside pure-Rust
        # code between the entering and exiting `detach_io` debug logs.
        first_emit_returned.wait(timeout=5.0)
        worker_ran_between_logs.set()

    worker = threading.Thread(target=background_worker, daemon=True)
    worker.start()
    worker_ready.wait(timeout=5.0)

    def on_rust_log(record: logging.LogRecord) -> None:
        nonlocal saw_enter_log, saw_exit_log, worker_ran_before_exit_log
        msg = record.getMessage()
        if msg == "Entering detached Rust execution":
            saw_enter_log = True
            first_emit_returned.set()
        elif msg == "Exiting detached Rust execution":
            saw_exit_log = True
            # Inside `on_rust_log` the main thread holds the GIL (acquired by
            # `pyo3-log` for the exit log). If `detach_io` did not release the
            # GIL via `py.detach(...)`, the GIL was never released when the
            # enter log returned, so `worker_ran_between_logs` could never be
            # set before the exit log arrives.
            worker_ran_before_exit_log = worker_ran_between_logs.is_set()

    logger = logging.getLogger("pixelator_core_py")
    prev_level = logger.level
    handler = _GilCheckHandler(on_rust_log)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        invoke(gil_check_parquet_path, tmp_path)
    finally:
        first_emit_returned.set()
        worker.join(timeout=5.0)
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    assert saw_enter_log and saw_exit_log, (
        f"{name} did not emit detach_io boundary log records"
    )
    assert worker_ran_before_exit_log, (
        f"background Python thread was starved of the GIL during {name}"
    )


_INTERRUPT_SCRIPT = textwrap.dedent(
    """
    import logging
    import os
    import signal
    import sys

    import pixelator_core

    parquet_file, output = sys.argv[1], sys.argv[2]


    class SendSigintOnEnter(logging.Handler):
        def emit(self, record):
            if record.getMessage() == "Entering detached Rust execution":
                os.kill(os.getpid(), signal.SIGINT)


    logger = logging.getLogger("pixelator_core_py")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(SendSigintOnEnter())

    {call}
    """
)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT is POSIX-only")
@pytest.mark.parametrize(
    "call",
    [
        "pixelator_core.run_label_propagation(parquet_file, output=output)",
        "pixelator_core.run_leiden(parquet_file, resolution=1.0, output=output)",
        (
            "pixelator_core.run_hybrid_community_detection("
            "parquet_file, resolution=1.0, multiplet_recovery=True, output=output)"
        ),
    ],
)
def test_ctrl_c_interrupts_rust_execution(
    call: str, interrupt_parquet_path: str, tmp_path: Path
) -> None:
    """Verify SIGINT raises `KeyboardInterrupt` and stops the Rust work early.

    The child process sends itself SIGINT as soon as the Rust work starts. The
    output parquet is written only after the algorithm finishes, so its absence
    shows the algorithm was cancelled rather than run to completion.
    """
    output = tmp_path / "out.parquet"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _INTERRUPT_SCRIPT.format(call=call),
            interrupt_parquet_path,
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "KeyboardInterrupt" in result.stderr
    assert not output.exists()


def test_encode_decode_dna_2bit() -> None:
    sequences = ["", "A", "ACGT", "ACGTACGTACGTACGTACGTACGTACGTACGT"]
    for seq in sequences:
        encoded = pixelator_core.encode_dna_2bit(seq)
        assert pixelator_core.decode_dna_2bit(encoded, len(seq)) == seq


def test_encode_dna_2bit_rejects_invalid_bases() -> None:
    with pytest.raises(ValueError):
        pixelator_core.encode_dna_2bit("AN")


def test_decode_dna_2bit_rejects_length_beyond_u64() -> None:
    with pytest.raises(ValueError):
        pixelator_core.decode_dna_2bit(0, 33)
