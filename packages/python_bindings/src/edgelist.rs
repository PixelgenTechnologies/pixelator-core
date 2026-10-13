use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

use flate2::read::MultiGzDecoder;
use pixelator_core::common::Panel;
use pixelator_core::edgelist::{edgelist, EdgelistStats, FastqInput, FilterConfig};
use pixelator_core::encoding::decode_dna_2bit as rust_decode_dna_2bit;
use pixelator_core::encoding::encode_dna_2bit as rust_encode_dna_2bit;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert;

/// How often edgelist progress is logged (reads processed since the last tick).
const PROGRESS_INTERVAL: Duration = Duration::from_secs(10);

/// Encode a DNA string into a 2-bit packed integer.
///
/// Args:
///     dna: DNA sequence containing only A/C/G/T characters.
///
/// Returns:
///     The 2-bit packed representation as an integer.
#[pyfunction]
pub fn encode_dna_2bit(dna: &str) -> PyResult<u64> {
    if dna
        .chars()
        .any(|base| !matches!(base, 'A' | 'C' | 'G' | 'T'))
    {
        return Err(PyValueError::new_err(
            "Unsupported DNA base in input. Allowed bases are A, C, G, T.",
        ));
    }
    Ok(rust_encode_dna_2bit(dna.as_bytes()))
}

/// Decode a 2-bit packed integer back into a DNA string.
///
/// Args:
///     encoded: The 2-bit packed representation as an integer.
///     length: Number of bases to decode (at most 32).
///
/// Returns:
///     The decoded DNA sequence.
#[pyfunction]
pub fn decode_dna_2bit(encoded: u64, length: usize) -> PyResult<String> {
    if length > 32 {
        return Err(PyValueError::new_err(
            "Cannot decode more than 32 bases from a 64-bit integer.",
        ));
    }
    String::from_utf8(rust_decode_dna_2bit(encoded, length))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Build an edgelist Parquet file from FASTQ input(s).
///
/// Provide both `read_forward` and `read_reverse` for paired-end mode, or exactly
/// one of them for single-end mode (forward or reverse). Passing neither raises
/// `ValueError`.
///
/// Args:
///     read_forward: Path to the gzipped forward-strand FASTQ file, or None.
///     read_reverse: Path to the gzipped reverse-strand FASTQ file, or None.
///     sample_name: Sample name used for the `<sample_name>.parquet` output filename.
///     assay: A loaded `PNAAssay` (from `pna_config.get_assay` or
///         `PNAAssay.from_yaml`). File paths are rejected.
///     panel: A loaded `PNAAntibodyPanel` (from `load_antibody_panel` or
///         `PNAAntibodyPanel.from_csv`). File paths are rejected.
///     output_dir: Directory where `<sample_name>.parquet` will be written.
///     n_threads: Number of worker threads to use for processing.
///     mismatch_threshold: Maximum number of mismatches allowed between the PID and the panel.
///         Defaults to `None`, which is treated as exact matching (0 mismatches).
///     min_avg_quality: Minimum average Phred quality required to keep a read.
///         Defaults to `None`, which disables the quality filter. Reads without
///         quality data are kept even when a threshold is set.
///     max_umi_base_freq: Maximum allowed fraction of any single base in a UMI.
///         Defaults to `None`, which disables the complexity filter. A UMI is
///         dropped when one base accounts for strictly more than this fraction.
///     lbs_overlap_max_edits: Maximum bounded infix Levenshtein distance between
///         the whole UMI and any substring of an assay LBS (or its reverse
///         complement) at which the read is dropped as LBS-contaminated.
///         Defaults to 6. The filter cannot be turned off from Python; pass 0
///         for exact-substring-only matching. Values at or above the UMI length
///         drop every read (empty-prefix score). No-op when the assay has no
///         LBS regions long enough to host the UMI.
///
///         Coverage caveat: only LBS references at least `umi_len - max_edits`
///         bases long are used, since a shorter reference would need more than
///         `max_edits` deletions to host the UMI.
///
/// Returns:
///     The run's statistics, keyed to match the fields of
///     `EdgelistSampleReport`: `input_reads`, `output_reads`, `q30_statistics`,
///     `dropped_by`, `reads_per_marker_1` and `reads_per_marker_2`. The caller
///     turns these into the sample report.
#[pyfunction]
#[pyo3(signature = (read_forward, read_reverse, sample_name, assay, panel, output_dir, n_threads, mismatch_threshold=None, min_avg_quality=None, max_umi_base_freq=None, lbs_overlap_max_edits=6))]
#[allow(clippy::too_many_arguments)]
pub fn run_edgelist<'py>(
    py: Python<'py>,
    read_forward: Option<PathBuf>,
    read_reverse: Option<PathBuf>,
    sample_name: String,
    assay: Bound<'py, PyAny>,
    panel: Bound<'py, PyAny>,
    output_dir: PathBuf,
    n_threads: usize,
    mismatch_threshold: Option<u8>,
    min_avg_quality: Option<u8>,
    max_umi_base_freq: Option<f64>,
    lbs_overlap_max_edits: u8,
) -> PyResult<Bound<'py, PyDict>> {
    let assay = convert::assay_from_py(&assay)?;
    let panel = convert::panel_from_py(&panel)?;
    // `edgelist` takes ownership of the panel, but the per-marker tallies are
    // keyed by panel position and need it again to be named.
    let report_panel = panel.clone();
    let input = open_input(read_forward, read_reverse)?;

    // The LBS-overlap filter is always applied; a missing argument maps to the
    // default budget. Passing 0 is allowed (exact-substring-only matching).
    let filters = FilterConfig {
        mismatch_threshold,
        min_avg_quality,
        max_umi_base_freq,
        lbs_overlap_max_edits: Some(lbs_overlap_max_edits),
    };

    let stats = run_logging_progress(py, move |reads_processed| {
        edgelist(
            input,
            sample_name,
            assay,
            panel,
            output_dir,
            n_threads,
            filters,
            reads_processed,
        )
        .map_err(|e| e.to_string())
    })?;

    stats_to_dict(py, &stats, &report_panel)
}

/// Resolves the two optional FASTQ paths into a sequencing mode, opening each.
///
/// Returns a `PyValueError` if both reads are `None`.
fn open_input(
    read_forward: Option<PathBuf>,
    read_reverse: Option<PathBuf>,
) -> PyResult<FastqInput> {
    let open_gz = |path: PathBuf| -> PyResult<Box<dyn Read + Send>> {
        let file = File::open(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Box::new(MultiGzDecoder::new(file)))
    };

    match (read_forward, read_reverse) {
        (Some(f), Some(r)) => Ok(FastqInput::Paired {
            r1: open_gz(f)?,
            r2: open_gz(r)?,
        }),
        (Some(f), None) => Ok(FastqInput::SingleForward(open_gz(f)?)),
        (None, Some(r)) => Ok(FastqInput::SingleReverse(open_gz(r)?)),
        (None, None) => Err(PyValueError::new_err(
            "at least one of read_forward / read_reverse must be provided",
        )),
    }
}

/// Runs `job` on a dedicated thread, logging read throughput every
/// [`PROGRESS_INTERVAL`] until it finishes, and returns what it produced.
///
/// `job` runs with the GIL released and cannot touch Python.
fn run_logging_progress(
    py: Python<'_>,
    job: impl FnOnce(Arc<AtomicU64>) -> Result<EdgelistStats, String> + Send + 'static,
) -> PyResult<EdgelistStats> {
    let reads_processed = Arc::new(AtomicU64::new(0));
    let (done_tx, mut done_rx) = channel::<()>();
    let worker = {
        let reads_processed = reads_processed.clone();
        std::thread::spawn(move || {
            let result = job(reads_processed);
            let _ = done_tx.send(());
            result
        })
    };

    let mut last = 0u64;
    loop {
        // Wait out one interval with the GIL released so other Python threads run.
        // `done_rx` is moved in and handed back because a `&Receiver` is not
        // `Send` and so cannot be captured by a `detach` closure.
        let (rx, timed_out) = py.detach(move || {
            let timed_out = matches!(
                done_rx.recv_timeout(PROGRESS_INTERVAL),
                Err(RecvTimeoutError::Timeout)
            );
            (done_rx, timed_out)
        });
        done_rx = rx;
        if !timed_out {
            break;
        }

        // Back on the calling thread with the GIL held, so `pyo3-log` is safe here.
        let now = reads_processed.load(Ordering::Relaxed);
        let delta = now - last;
        let secs = PROGRESS_INTERVAL.as_secs_f64();
        log::info!(
            "Processed {:.2}M reads ({:.2}M in the last {}s, {:.2}M reads/s)",
            now as f64 / 1_000_000.0,
            delta as f64 / 1_000_000.0,
            PROGRESS_INTERVAL.as_secs(),
            delta as f64 / secs / 1_000_000.0
        );
        last = now;
    }

    py.detach(|| worker.join())
        .expect("edgelist thread panicked")
        .map_err(PyRuntimeError::new_err)
}

/// Mirrors [`EdgelistStats`] as a plain dict, naming markers from `panel` (the
/// one the run used).
fn stats_to_dict<'py>(
    py: Python<'py>,
    stats: &EdgelistStats,
    panel: &Panel,
) -> PyResult<Bound<'py, PyDict>> {
    // Keyed to match the fields of `EdgelistSampleReport`, so the CLI can splat it
    // straight into the model the way the other stages do with `stats.as_json()`.
    let out = PyDict::new(py);
    out.set_item("input_reads", stats.total_reads())?;
    out.set_item("output_reads", stats.total_kept_reads())?;

    let q30 = stats.q30();
    let q30_dict = PyDict::new(py);
    q30_dict.set_item("total", q30.total)?;
    q30_dict.set_item("pid1", q30.pid1)?;
    q30_dict.set_item("umi1", q30.umi1)?;
    q30_dict.set_item("pid2", q30.pid2)?;
    q30_dict.set_item("umi2", q30.umi2)?;
    out.set_item("q30_statistics", q30_dict)?;

    // `dropped_by` must be filled by iterating the ranked pairs rather than handing
    // pyo3 a map: dict insertion order is what carries the descending-count
    // ranking into the JSON report.
    let dropped_by_dict = PyDict::new(py);
    for (reason, count) in stats.dropped_by_frequency() {
        dropped_by_dict.set_item(reason, count)?;
    }
    out.set_item("dropped_by", dropped_by_dict)?;

    // Same reason as `dropped_by`: insertion order carries the ranking.
    for (key, ranked) in [
        ("reads_per_marker_1", stats.marker1_by_frequency(panel)),
        ("reads_per_marker_2", stats.marker2_by_frequency(panel)),
    ] {
        let dict = PyDict::new(py);
        for (marker, count) in ranked {
            dict.set_item(marker, count)?;
        }
        out.set_item(key, dict)?;
    }

    Ok(out)
}
