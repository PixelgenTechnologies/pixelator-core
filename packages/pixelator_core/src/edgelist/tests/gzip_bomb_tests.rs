//! Spec for the gzip FASTQ path used by the Python edgelist entry point.
//!
//! `open_input` in `packages/python_bindings/src/edgelist.rs` opens each mate
//! with `File` + `flate2::read::MultiGzDecoder` and passes those readers to
//! [`super::super::edgelist`]. These tests drive that same pair of calls.
//!
//! They fail while a single decoded FASTQ record is buffered without a ceiling.
//! They pass once every mate is rejected, before the decoded record is retained
//! and before either output Parquet exists, when the record (including one split
//! across gzip members) exceeds the retained-data ceiling below.
//!
//! A long stream of short records is ordinary sequencing input and must still
//! complete without retaining the decompressed stream.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use flate2::read::MultiGzDecoder;

use super::super::{FastqInput, FilterConfig, edgelist};
use crate::test_fixtures::{proxiome_v2_assay, proxiome_v2_panel};

/// Decoded data a rejected sample may retain above the RSS baseline.
const MAX_RETAINED_KIB: u64 = 16 * 1024;

/// Headroom for a run that only ever holds short FASTQ batches.
const MAX_SHORT_STREAM_KIB: u64 = 32 * 1024;

const CASE_ENV: &str = "PIXELATOR_GZIP_BOMB_CASE";
const SENTINEL_ENV: &str = "PIXELATOR_GZIP_BOMB_SENTINEL";

#[test]
fn oversized_gzip_fastq_is_rejected_without_retaining_it() {
    // One process per case. A later case would reuse the first bomb's pages
    // and the RSS delta would no longer show the allocation.
    if std::env::var(CASE_ENV).is_err() {
        let sentinel_dir = tempfile::tempdir().expect("sentinel dir");
        let test_name = std::thread::current()
            .name()
            .expect("the test harness names this thread")
            .to_string();
        let mut failed = Vec::new();
        for case in CASES {
            let sentinel = sentinel_dir.path().join(case);
            let status = Command::new(std::env::current_exe().expect("test executable"))
                .args([test_name.as_str(), "--exact", "--nocapture"])
                .env(CASE_ENV, case)
                .env(SENTINEL_ENV, &sentinel)
                .status()
                .expect("spawn case");
            if !(status.success() && sentinel.exists()) {
                failed.push(*case);
            }
        }
        assert!(
            failed.is_empty(),
            "oversized gzip FASTQ records must be rejected without retaining them: {failed:?}"
        );
        return;
    }

    match std::env::var(CASE_ENV).expect("case").as_str() {
        "forward_parallel" => reject_bomb(Mode::Forward, 2, false, Bomb::SingleRecord),
        "forward_sequential" => reject_bomb(Mode::Forward, 1, false, Bomb::SingleRecord),
        "reverse" => reject_bomb(Mode::Reverse, 2, false, Bomb::SingleRecord),
        "paired_forward" => reject_bomb(Mode::PairedForward, 2, false, Bomb::SingleRecord),
        "paired_reverse" => reject_bomb(Mode::PairedReverse, 2, false, Bomb::SingleRecord),
        "paired_both" => reject_bomb(Mode::PairedBoth, 2, false, Bomb::SingleRecord),
        "concat_members" => reject_bomb(Mode::Forward, 2, false, Bomb::ConcatMembers),
        "save_dropped" => reject_bomb(Mode::Forward, 2, true, Bomb::SingleRecord),
        "short_stream" => expect_short_stream(),
        other => panic!("unknown case {other}"),
    }
    std::fs::write(std::env::var(SENTINEL_ENV).expect("sentinel"), "ok").expect("sentinel");
}

const CASES: &[&str] = &[
    "forward_parallel",
    "forward_sequential",
    "reverse",
    "paired_forward",
    "paired_reverse",
    "paired_both",
    "concat_members",
    "save_dropped",
    "short_stream",
];

enum Mode {
    Forward,
    Reverse,
    PairedForward,
    PairedReverse,
    PairedBoth,
}

enum Bomb {
    /// One FASTQ record whose sequence and quality are each 16 MiB.
    SingleRecord,
    /// That same record split across two gzip members.
    ConcatMembers,
}

/// The three arms of `open_input`: each supplied path is a file wrapped in
/// `MultiGzDecoder`, with no decoded-size ceiling.
fn open_mates(forward: Option<PathBuf>, reverse: Option<PathBuf>) -> FastqInput {
    let open_gz = |path: PathBuf| -> Box<dyn Read + Send> {
        let file = std::fs::File::open(path).expect("open gzip fastq");
        Box::new(MultiGzDecoder::new(file))
    };
    match (forward, reverse) {
        (Some(forward), Some(reverse)) => FastqInput::Paired {
            r1: open_gz(forward),
            r2: open_gz(reverse),
        },
        (Some(forward), None) => FastqInput::SingleForward(open_gz(forward)),
        (None, Some(reverse)) => FastqInput::SingleReverse(open_gz(reverse)),
        (None, None) => panic!("a sample needs a mate"),
    }
}

fn reject_bomb(mode: Mode, n_threads: usize, save_dropped_reads: bool, bomb: Bomb) {
    let dir = tempfile::tempdir().expect("tempdir");
    let bomb_path = dir.path().join("bomb.fastq.gz");
    write_bomb(&bomb_path, &bomb);
    let gz_len = std::fs::metadata(&bomb_path).expect("bomb metadata").len();
    assert!(
        gz_len < 256 * 1024,
        "the fixture gzip is {gz_len} bytes; the test needs a small compressed input"
    );
    let short_path = dir.path().join("short.fastq.gz");
    write_short_fastq(&short_path);
    let second_bomb = dir.path().join("bomb2.fastq.gz");
    if matches!(mode, Mode::PairedBoth) {
        write_bomb(&second_bomb, &bomb);
    }

    let output_dir = dir.path().join("out");
    std::fs::create_dir(&output_dir).expect("output dir");
    let parquet = output_dir.join("sample.parquet");
    let dropped = output_dir.join("sample.dropped_reads.parquet");

    let (forward, reverse) = match mode {
        Mode::Forward => (Some(bomb_path), None),
        Mode::Reverse => (None, Some(bomb_path)),
        Mode::PairedForward => (Some(bomb_path), Some(short_path)),
        Mode::PairedReverse => (Some(short_path), Some(bomb_path)),
        Mode::PairedBoth => (Some(bomb_path), Some(second_bomb)),
    };

    let assay = proxiome_v2_assay();
    let panel = proxiome_v2_panel();
    let created = Arc::new(AtomicBool::new(false));
    let watch = [parquet.clone(), dropped.clone()];
    let (outcome, peak, start) = with_peak_rss(&created, &watch, || {
        edgelist(
            open_mates(forward, reverse),
            "sample".to_string(),
            assay,
            panel,
            &output_dir,
            n_threads,
            FilterConfig::default(),
            save_dropped_reads,
            Arc::new(AtomicU64::new(0)),
        )
        .map_err(|err| err.to_string())
    });

    let retained = peak.saturating_sub(start);
    assert!(
        outcome.is_err(),
        "oversized gzip FASTQ must fail the sample, got Ok after retaining {retained} KiB"
    );
    assert!(
        retained < MAX_RETAINED_KIB,
        "retained {retained} KiB of decoded FASTQ; ceiling is {MAX_RETAINED_KIB} KiB"
    );
    assert!(
        !created.load(Ordering::Relaxed) && !parquet.exists() && !dropped.exists(),
        "the sample wrote an output file before rejecting the gzip FASTQ"
    );
}

fn expect_short_stream() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("many.fastq.gz");
    write_many_short(&path, 4 * 1024 * 1024);
    let output_dir = dir.path().join("out");
    std::fs::create_dir(&output_dir).expect("output dir");
    let assay = proxiome_v2_assay();
    let panel = proxiome_v2_panel();
    let (outcome, peak, start) = with_peak_rss(&Arc::new(AtomicBool::new(false)), &[], || {
        edgelist(
            open_mates(Some(path), None),
            "sample".to_string(),
            assay,
            panel,
            &output_dir,
            2,
            FilterConfig::default(),
            false,
            Arc::new(AtomicU64::new(0)),
        )
        .map_err(|err| err.to_string())
    });
    let retained = peak.saturating_sub(start);
    let stats = outcome.expect("a long stream of short FASTQ records must be accepted");
    assert!(
        stats.total_reads() > 1_000_000,
        "short-record stream was not consumed"
    );
    assert!(
        retained < MAX_SHORT_STREAM_KIB,
        "short FASTQ records retained {retained} KiB; decoded data must stay batched"
    );
}

fn with_peak_rss<T>(
    created: &Arc<AtomicBool>,
    watch: &[PathBuf],
    body: impl FnOnce() -> T,
) -> (T, u64, u64) {
    let start = vm_rss_kib();
    let peak = Arc::new(AtomicU64::new(start));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let peak = Arc::clone(&peak);
        let stop = Arc::clone(&stop);
        let watch: Vec<PathBuf> = watch.to_vec();
        let created = Arc::clone(created);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                peak.fetch_max(vm_rss_kib(), Ordering::Relaxed);
                if watch.iter().any(|path| path.exists()) {
                    created.store(true, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        })
    };
    let result = body();
    stop.store(true, Ordering::Relaxed);
    sampler.join().expect("rss sampler");
    peak.fetch_max(vm_rss_kib(), Ordering::Relaxed);
    if watch.iter().any(|path| path.exists()) {
        created.store(true, Ordering::Relaxed);
    }
    (result, peak.load(Ordering::Relaxed), start)
}

fn vm_rss_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("status");
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .expect("VmRSS")
        .split_whitespace()
        .next()
        .expect("rss value")
        .parse()
        .expect("rss number")
}

fn write_bomb(path: &Path, bomb: &Bomb) {
    let script = match bomb {
        Bomb::SingleRecord => {
            r#"
import gzip, sys
path = sys.argv[1]
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
"#
        }
        Bomb::ConcatMembers => {
            r#"
import gzip, sys
path = sys.argv[1]
seq = b"A" * (16 * 1024 * 1024)
qual = b"I" * (16 * 1024 * 1024)
first = b"@bomb\n" + seq[: 8 * 1024 * 1024]
second = seq[8 * 1024 * 1024 :] + b"\n+\n" + qual + b"\n"
with open(path, "wb") as out:
    out.write(gzip.compress(first, compresslevel=9))
    out.write(gzip.compress(second, compresslevel=9))
"#
        }
    };
    let status = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(path)
        .status()
        .expect("python3");
    assert!(status.success(), "writing the gzip fixture failed");
}

fn write_short_fastq(path: &Path) {
    let file = std::fs::File::create(path).expect("create gzip");
    let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::new(9));
    encoder
        .write_all(b"@short\nACGT\n+\nIIII\n")
        .expect("write gzip");
    encoder.finish().expect("finish gzip");
}

fn write_many_short(path: &Path, n: usize) {
    let script = r#"
import gzip, sys
path, n = sys.argv[1], int(sys.argv[2])
one = b"@r\nACGT\n+\nIIII\n"
with gzip.open(path, "wb", compresslevel=9) as out:
    for _ in range(n):
        out.write(one)
"#;
    let status = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(path)
        .arg(n.to_string())
        .status()
        .expect("python3");
    assert!(status.success(), "writing the short-read gzip failed");
}
