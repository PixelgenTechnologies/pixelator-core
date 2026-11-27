"""Spec for gzip FASTQ paths passed to ``pixelator_core.run_edgelist``.

These tests fail while that entry point decodes a mate with an unbounded
``MultiGzDecoder`` and the FASTQ reader keeps the whole decoded record. They
pass once every mate is rejected, before the decoded record is retained and
before either output Parquet exists, when one record exceeds the ceiling below.
A record split across gzip members counts as one record. The Rust test
``oversized_gzip_fastq_is_rejected_without_retaining_it`` also checks that a
long stream of short records still completes without retaining that stream.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import _edgelist_inputs as inputs

import pixelator_core

MAX_RETAINED_KIB = 16 * 1024

CASES = (
    "forward_parallel",
    "forward_sequential",
    "reverse",
    "paired_forward",
    "paired_reverse",
    "paired_both",
    "concat_members",
    "save_dropped",
)


def test_oversized_gzip_fastq_is_rejected_without_retaining_it(tmp_path) -> None:
    """One fresh interpreter per case, so a freed bomb cannot hide the next."""
    failed: list[str] = []
    for case in CASES:
        sentinel = tmp_path / case
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), case, str(sentinel)],
            check=False,
        )
        if completed.returncode != 0 or not sentinel.exists():
            failed.append(case)
    assert not failed, (
        "oversized gzip FASTQ records must be rejected without retaining them: "
        f"{failed}"
    )


def _run_case(case: str, sentinel: Path) -> None:
    _reject_bomb(case)
    sentinel.write_text("ok")


def _reject_bomb(case: str) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        bomb = directory / "bomb.fastq.gz"
        _write_bomb(bomb, concat=case == "concat_members")
        gz_len = bomb.stat().st_size
        assert gz_len < 256 * 1024, gz_len
        short = directory / "short.fastq.gz"
        _write_short(short)
        second = directory / "bomb2.fastq.gz"
        if case == "paired_both":
            _write_bomb(second, concat=False)
        output = directory / "out"
        output.mkdir()
        forward, reverse = _mates(case, bomb, short, second)
        n_threads = 1 if case == "forward_sequential" else 2
        assay, panel = inputs.load_assay(), inputs.load_panel()
        parquet = output / "sample.parquet"
        dropped = output / "sample.dropped_reads.parquet"
        created = _Flag()
        start = _rss_kib()
        peak = _Peak(start, created, (parquet, dropped))
        peak.start()
        try:
            try:
                pixelator_core.run_edgelist(
                    forward,
                    reverse,
                    "sample",
                    assay,
                    panel,
                    str(output),
                    n_threads,
                    save_dropped_reads=case == "save_dropped",
                )
            except Exception:
                rejected = True
            else:
                rejected = False
        finally:
            peak.stop()
        retained = peak.kib - start
        assert rejected, (
            "oversized gzip FASTQ must fail the sample, "
            f"got success after retaining {retained} KiB"
        )
        assert retained < MAX_RETAINED_KIB, (
            f"retained {retained} KiB of decoded FASTQ; ceiling is {MAX_RETAINED_KIB} KiB"
        )
        assert not created.seen and not parquet.exists() and not dropped.exists(), (
            "the sample wrote an output file before rejecting the gzip FASTQ"
        )


def _mates(
    case: str, bomb: Path, short: Path, second: Path
) -> tuple[str | None, str | None]:
    if case in {"forward_parallel", "forward_sequential", "concat_members", "save_dropped"}:
        return str(bomb), None
    if case == "reverse":
        return None, str(bomb)
    if case == "paired_forward":
        return str(bomb), str(short)
    if case == "paired_reverse":
        return str(short), str(bomb)
    if case == "paired_both":
        return str(bomb), str(second)
    raise AssertionError(case)


def _write_bomb(path: Path, concat: bool) -> None:
    import gzip

    if not concat:
        with gzip.open(path, "wb", compresslevel=9) as out:
            out.write(b"@bomb\n")
            chunk = b"A" * (1024 * 1024)
            for _ in range(16):
                out.write(chunk)
            out.write(b"\n+\n")
            chunk = b"I" * (1024 * 1024)
            for _ in range(16):
                out.write(chunk)
            out.write(b"\n")
        return
    seq = b"A" * (16 * 1024 * 1024)
    qual = b"I" * (16 * 1024 * 1024)
    first = b"@bomb\n" + seq[: 8 * 1024 * 1024]
    second = seq[8 * 1024 * 1024 :] + b"\n+\n" + qual + b"\n"
    path.write_bytes(gzip.compress(first, compresslevel=9) + gzip.compress(second, compresslevel=9))


def _write_short(path: Path) -> None:
    import gzip

    with gzip.open(path, "wb") as out:
        out.write(b"@short\nACGT\n+\nIIII\n")


def _rss_kib() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise AssertionError("VmRSS missing")


class _Flag:
    def __init__(self) -> None:
        self.seen = False


class _Peak:
    def __init__(self, start: int, created: _Flag, watch: tuple[Path, ...]) -> None:
        self.kib = start
        self._created = created
        self._watch = watch
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def sample() -> None:
            while not self._stop:
                self.kib = max(self.kib, _rss_kib())
                if any(path.exists() for path in self._watch):
                    self._created.seen = True
                time.sleep(0.001)

        self._thread = threading.Thread(target=sample)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        assert self._thread is not None
        self._thread.join()
        self.kib = max(self.kib, _rss_kib())
        if any(path.exists() for path in self._watch):
            self._created.seen = True


if __name__ == "__main__":
    _run_case(sys.argv[1], Path(sys.argv[2]))
