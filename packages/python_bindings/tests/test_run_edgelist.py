"""End-to-end tests for the `run_edgelist` Python binding."""

from __future__ import annotations

from pathlib import Path

import _edgelist_inputs as inputs
import pyarrow.parquet as pq
import pytest

import pixelator_core

N_READS = 200

EDGELIST_COLUMNS = [
    "marker_1",
    "marker_2",
    "umi1",
    "umi2",
    "marker_sequence_1",
    "marker_sequence_2",
]

# Every reason the extractor can report, whether or not it fired.
DROP_REASONS = {
    "has_n",
    "lbs_overlap",
    "low_complexity",
    "low_quality",
    "pid1_ambiguous",
    "pid1_not_in_panel",
    "pid2_ambiguous",
    "pid2_not_in_panel",
    "too_short",
}


@pytest.fixture(scope="module")
def assay() -> inputs.Assay:
    return inputs.load_assay()


@pytest.fixture(scope="module")
def panel() -> inputs.Panel:
    return inputs.load_panel()


@pytest.fixture(scope="module")
def amplicons(assay, panel) -> list[str]:
    return inputs.amplicons(N_READS, assay, panel)


def run_paired(tmp_path: Path, amplicons, assay, panel, **kwargs) -> dict:
    r1, r2 = inputs.write_paired(tmp_path, amplicons)
    return pixelator_core.run_edgelist(
        str(r1), str(r2), "sample", assay, panel, str(tmp_path), **kwargs
    )


def run_single(tmp_path: Path, r1, r2, assay, panel, **kwargs) -> dict:
    return pixelator_core.run_edgelist(
        r1, r2, "sample", assay, panel, str(tmp_path), **kwargs
    )


def rows(path: Path) -> list[tuple]:
    """The parquet's rows, sorted: worker threads write batches in any order."""
    table = pq.read_table(path)
    return sorted(
        tuple(row.values()) for row in table.to_pylist()
    )


def test_paired_end_writes_one_row_per_read(tmp_path, amplicons, assay, panel):
    stats = run_paired(tmp_path, amplicons, assay, panel, n_threads=2)

    table = pq.read_table(tmp_path / "sample.parquet")
    assert table.column_names == EDGELIST_COLUMNS
    assert table.num_rows == N_READS

    assert stats["input_reads"] == N_READS
    assert stats["output_reads"] == N_READS
    assert set(stats["dropped_by"]) == DROP_REASONS
    assert not any(stats["dropped_by"].values())
    assert sum(stats["reads_per_marker_1"].values()) == N_READS
    assert sum(stats["reads_per_marker_2"].values()) == N_READS
    assert set(stats["q30_statistics"]) == {"total", "pid1", "umi1", "pid2", "umi2"}


def test_edgelist_stores_the_panel_sequences_of_each_marker(
    tmp_path, amplicons, assay, panel
):
    run_paired(tmp_path, amplicons, assay, panel, n_threads=2)

    table = pq.read_table(tmp_path / "sample.parquet")
    by_id = {row["marker_id"]: row for row in panel.rows}
    for marker_column, sequence_column, panel_column in (
        ("marker_1", "marker_sequence_1", "sequence_1"),
        ("marker_2", "marker_sequence_2", "sequence_2"),
    ):
        markers = table.column(marker_column).to_pylist()
        sequences = table.column(sequence_column).to_pylist()
        assert sequences == [by_id[marker][panel_column] for marker in markers]


def test_single_forward_writes_one_row_per_read(tmp_path, amplicons, assay, panel):
    forward = inputs.write_single_forward(tmp_path, amplicons)

    stats = run_single(tmp_path, str(forward), None, assay, panel, n_threads=2)

    assert pq.read_metadata(tmp_path / "sample.parquet").num_rows == N_READS
    assert stats["output_reads"] == N_READS


def test_single_reverse_writes_one_row_per_read(tmp_path, amplicons, assay, panel):
    reverse = inputs.write_single_reverse(tmp_path, amplicons)

    stats = run_single(tmp_path, None, str(reverse), assay, panel, n_threads=2)

    assert pq.read_metadata(tmp_path / "sample.parquet").num_rows == N_READS
    assert stats["output_reads"] == N_READS


def test_single_end_modes_agree_with_paired_end(tmp_path, amplicons, assay, panel):
    paired_dir, forward_dir, reverse_dir = (
        tmp_path / name for name in ("paired", "forward", "reverse")
    )
    for directory in (paired_dir, forward_dir, reverse_dir):
        directory.mkdir()

    run_paired(paired_dir, amplicons, assay, panel, n_threads=2)
    run_single(
        forward_dir,
        str(inputs.write_single_forward(forward_dir, amplicons)),
        None,
        assay,
        panel,
        n_threads=2,
    )
    run_single(
        reverse_dir,
        None,
        str(inputs.write_single_reverse(reverse_dir, amplicons)),
        assay,
        panel,
        n_threads=2,
    )

    expected = rows(paired_dir / "sample.parquet")
    assert rows(forward_dir / "sample.parquet") == expected
    assert rows(reverse_dir / "sample.parquet") == expected


@pytest.mark.parametrize("n_threads", [1, 4])
def test_output_does_not_depend_on_thread_count(
    tmp_path, amplicons, assay, panel, n_threads
):
    reference_dir, threaded_dir = tmp_path / "reference", tmp_path / "threaded"
    reference_dir.mkdir()
    threaded_dir.mkdir()

    run_paired(reference_dir, amplicons, assay, panel, n_threads=2)
    run_paired(threaded_dir, amplicons, assay, panel, n_threads=n_threads)

    assert rows(threaded_dir / "sample.parquet") == rows(
        reference_dir / "sample.parquet"
    )


def test_dropped_reads_are_not_saved_by_default(tmp_path, amplicons, assay, panel):
    dropped = [inputs.with_n_in_umi(a) for a in amplicons[:10]] + amplicons[10:]

    stats = run_paired(tmp_path, dropped, assay, panel, n_threads=2)

    assert stats["dropped_by"]["has_n"] == 10
    assert not (tmp_path / "sample.dropped_reads.parquet").exists()


def test_save_dropped_reads_paired(tmp_path, amplicons, assay, panel):
    n_dropped = 10
    reads = [inputs.with_n_in_umi(a) for a in amplicons[:n_dropped]] + amplicons[
        n_dropped:
    ]

    stats = run_paired(
        tmp_path, reads, assay, panel, n_threads=2, save_dropped_reads=True
    )

    assert stats["input_reads"] == N_READS
    assert stats["output_reads"] == N_READS - n_dropped
    assert stats["dropped_by"]["has_n"] == n_dropped
    assert pq.read_metadata(tmp_path / "sample.parquet").num_rows == N_READS - n_dropped

    dropped = pq.read_table(tmp_path / "sample.dropped_reads.parquet").to_pylist()
    assert len(dropped) == n_dropped
    assert {row["reason"] for row in dropped} == {"has_n"}
    assert all(row["sequence"].startswith("N") for row in dropped)
    assert all(row["sequence_2"] is not None for row in dropped)
    assert all(row["read_name"].startswith("read") for row in dropped)


def test_save_dropped_reads_single_end_leaves_mate_columns_null(
    tmp_path, amplicons, assay, panel
):
    reads = [inputs.with_n_in_umi(a) for a in amplicons[:5]] + amplicons[5:]
    forward = inputs.write_single_forward(tmp_path, reads)

    run_single(
        tmp_path, str(forward), None, assay, panel, n_threads=2, save_dropped_reads=True
    )

    dropped = pq.read_table(tmp_path / "sample.dropped_reads.parquet")
    assert dropped.num_rows == 5
    assert dropped.column("sequence_2").null_count == 5
    assert dropped.column("quality_2").null_count == 5


def test_save_dropped_reads_without_drops_writes_an_empty_file(
    tmp_path, amplicons, assay, panel
):
    run_paired(tmp_path, amplicons, assay, panel, n_threads=2, save_dropped_reads=True)

    dropped = pq.read_table(tmp_path / "sample.dropped_reads.parquet")
    assert dropped.num_rows == 0
    assert dropped.column_names == [
        "read_name",
        "sequence",
        "quality",
        "sequence_2",
        "quality_2",
        "reason",
    ]


def test_all_reads_dropped_writes_an_empty_edgelist(tmp_path, amplicons, assay, panel):
    reads = [inputs.with_n_in_umi(a) for a in amplicons]

    stats = run_paired(tmp_path, reads, assay, panel, n_threads=2)

    assert stats["input_reads"] == N_READS
    assert stats["output_reads"] == 0
    assert stats["dropped_by"]["has_n"] == N_READS
    table = pq.read_table(tmp_path / "sample.parquet")
    assert table.num_rows == 0
    assert table.column_names == EDGELIST_COLUMNS


def test_requires_at_least_one_read_file(tmp_path, assay, panel):
    with pytest.raises(ValueError, match="read_forward / read_reverse"):
        run_single(tmp_path, None, None, assay, panel, n_threads=1)


def test_rejects_file_paths_for_assay_and_panel(tmp_path, amplicons, assay, panel):
    forward = str(inputs.write_single_forward(tmp_path, amplicons))

    with pytest.raises(TypeError, match="assay"):
        run_single(
            tmp_path,
            forward,
            None,
            str(inputs.FIXTURES / "proxiome-v2.yaml"),
            panel,
            n_threads=1,
        )
    with pytest.raises(TypeError, match="panel"):
        run_single(
            tmp_path,
            forward,
            None,
            assay,
            inputs.FIXTURES / "proxiome-v2-immuno-155-v2.0.csv",
            n_threads=1,
        )


def test_unwritable_output_dir_raises(tmp_path, amplicons, assay, panel):
    forward = str(inputs.write_single_forward(tmp_path, amplicons))
    missing = tmp_path / "does" / "not" / "exist"

    with pytest.raises(RuntimeError):
        pixelator_core.run_edgelist(
            forward, None, "sample", assay, panel, str(missing), n_threads=2
        )
