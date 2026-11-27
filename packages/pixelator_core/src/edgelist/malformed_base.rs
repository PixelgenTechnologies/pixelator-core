//! A non-ACGT byte in a protein-identifier or UMI window must drop that read.
//!
//! The edgelist entry point only rejects an uppercase `N` before packing the
//! window. Any other byte (a lowercase letter, an ambiguous IUPAC base, a
//! digit) reaches the 2-bit encoder, which panics. These tests drive the same
//! path a gzipped FASTQ takes: gzip bytes, `MultiGzDecoder`, then [`edgelist`]
//! on the proxiome-v2 assay and panel. Each file contains one malformed read
//! followed by one valid read. The run must finish, keep the valid read, and
//! drop the malformed one.
//!
//! The bad byte sits on the last position of the window, so a check that does
//! not inspect the same slice the encoder packs still panics.

use std::fs::File;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use flate2::Compression;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::{FastqInput, FilterConfig, edgelist};
use crate::edgelist::Strand;
use crate::test_fixtures::{
    OwnedRecord, ReadMixConfig, proxiome_v2_assay, proxiome_v2_marker_sequences, proxiome_v2_panel,
    proxiome_v2_read_pairs,
};

fn valid_mix() -> ReadMixConfig {
    ReadMixConfig {
        frac_has_n: 0.0,
        frac_unknown_pid: 0.0,
        read_len: 150,
    }
}

fn records_to_fastq(records: &[OwnedRecord]) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, rec) in records.iter().enumerate() {
        out.extend_from_slice(format!("@read{i}\n").as_bytes());
        out.extend_from_slice(&rec.seq);
        out.push(b'\n');
        out.extend_from_slice(b"+\n");
        out.extend_from_slice(&rec.qual);
        out.push(b'\n');
    }
    out
}

fn gzip(raw: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(raw).expect("gzip fastq");
    encoder.finish().expect("finish gzip")
}

/// Opens a gzipped FASTQ the way the Python edgelist entry point does:
/// `File::open` then [`MultiGzDecoder`].
fn open_gz(path: &std::path::Path) -> Box<dyn Read + Send> {
    let file = File::open(path).expect("open gzipped fastq");
    Box::new(MultiGzDecoder::new(file))
}

fn last_base(assay: &crate::common::Assay, strand: Strand, region_id: &str) -> usize {
    let (start, end) = strand
        .bounds(assay, region_id)
        .unwrap_or_else(|| panic!("{region_id} bounds on {strand:?}"));
    assert!(end > start, "{region_id} window is empty");
    end - 1
}

fn poke(record: &mut OwnedRecord, index: usize, base: u8) {
    assert!(
        index < record.seq.len(),
        "window offset {index} past read length {}",
        record.seq.len()
    );
    record.seq[index] = base;
}

struct Run {
    kept: usize,
    dropped: usize,
    parquet_rows: usize,
}

fn run_edgelist(input: FastqInput, n_threads: usize) -> Run {
    let tmp = tempfile::tempdir().expect("tempdir");
    let stats = edgelist(
        input,
        "sample".to_string(),
        proxiome_v2_assay(),
        proxiome_v2_panel(),
        tmp.path(),
        n_threads,
        FilterConfig::default(),
        false,
        Arc::new(AtomicU64::new(0)),
    )
    .expect("a malformed pid/umi base must drop that read and still write the edgelist");

    let parquet_path = tmp.path().join("sample.parquet");
    assert!(
        parquet_path.is_file(),
        "edgelist must leave sample.parquet in place"
    );
    let file = File::open(&parquet_path).expect("open edgelist parquet");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("parquet reader")
        .build()
        .expect("build parquet reader");
    let parquet_rows = reader
        .map(|batch| batch.expect("record batch").num_rows())
        .sum();

    Run {
        kept: stats.total_kept_reads(),
        dropped: stats.total_dropped_reads(),
        parquet_rows,
    }
}

fn assert_one_dropped_one_kept(run: &Run) {
    assert_eq!(run.kept, 1, "the valid read must be kept");
    assert_eq!(run.dropped, 1, "the malformed read must be dropped");
    assert_eq!(
        run.parquet_rows, 1,
        "the edgelist has one row per kept read"
    );
}

/// Paired FASTQ. `mate_is_r1` selects which gzipped file receives `base` at the
/// last byte of `region` (forward bounds on R1, reverse bounds on R2).
fn paired_case(mate_is_r1: bool, region: &str, base: u8, n_threads: usize) {
    let assay = proxiome_v2_assay();
    let mut pairs = proxiome_v2_read_pairs(2, valid_mix(), 11);
    let strand = if mate_is_r1 {
        Strand::Forward
    } else {
        Strand::Reverse
    };
    let index = last_base(&assay, strand, region);
    let (r1, r2) = &mut pairs[0];
    if mate_is_r1 {
        poke(r1, index, base);
    } else {
        poke(r2, index, base);
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let r1_path = dir.path().join("r1.fastq.gz");
    let r2_path = dir.path().join("r2.fastq.gz");
    std::fs::write(
        &r1_path,
        gzip(&records_to_fastq(
            &pairs
                .iter()
                .map(|(r1, _)| r1.clone_record())
                .collect::<Vec<_>>(),
        )),
    )
    .expect("write r1");
    std::fs::write(
        &r2_path,
        gzip(&records_to_fastq(
            &pairs
                .iter()
                .map(|(_, r2)| r2.clone_record())
                .collect::<Vec<_>>(),
        )),
    )
    .expect("write r2");

    let run = run_edgelist(
        FastqInput::Paired {
            r1: open_gz(&r1_path),
            r2: open_gz(&r2_path),
        },
        n_threads,
    );
    assert_one_dropped_one_kept(&run);
}

fn single_case(strand: Strand, region: &str, base: u8, n_threads: usize) {
    let assay = proxiome_v2_assay();
    let markers = proxiome_v2_marker_sequences();
    let mut reads =
        crate::test_fixtures::generate_single_reads(&assay, &markers, 2, valid_mix(), 11, strand);
    let index = last_base(&assay, strand, region);
    poke(&mut reads[0], index, base);

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("reads.fastq.gz");
    std::fs::write(&path, gzip(&records_to_fastq(&reads))).expect("write fastq");
    let input = match strand {
        Strand::Forward => FastqInput::SingleForward(open_gz(&path)),
        Strand::Reverse => FastqInput::SingleReverse(open_gz(&path)),
    };
    let run = run_edgelist(input, n_threads);
    assert_one_dropped_one_kept(&run);
}

/// `OwnedRecord` is only moved, so cloning the fields keeps the helper small
/// without adding a `Clone` impl to the fixture type.
trait CloneRecord {
    fn clone_record(&self) -> OwnedRecord;
}

impl CloneRecord for OwnedRecord {
    fn clone_record(&self) -> OwnedRecord {
        OwnedRecord {
            id: self.id.clone(),
            seq: self.seq.clone(),
            qual: self.qual.clone(),
        }
    }
}

#[test]
fn paired_worker_drops_lowercase_a_in_r1_pid1() {
    paired_case(true, "pid-1", b'a', 2);
}

#[test]
fn paired_worker_drops_lowercase_n_in_r1_umi1() {
    paired_case(true, "umi-1", b'n', 2);
}

#[test]
fn paired_worker_drops_ambiguous_r_in_r2_pid2() {
    paired_case(false, "pid-2", b'R', 2);
}

#[test]
fn paired_worker_drops_lowercase_y_in_r2_umi2() {
    paired_case(false, "umi-2", b'y', 2);
}

#[test]
fn paired_worker_drops_digit_in_r1_pid1() {
    paired_case(true, "pid-1", b'0', 2);
}

#[test]
fn single_forward_drops_lowercase_c_in_pid2() {
    single_case(Strand::Forward, "pid-2", b'c', 2);
}

#[test]
fn single_forward_drops_ambiguous_w_in_umi2() {
    single_case(Strand::Forward, "umi-2", b'W', 2);
}

#[test]
fn single_reverse_drops_lowercase_t_in_pid1() {
    single_case(Strand::Reverse, "pid-1", b't', 2);
}

#[test]
fn single_reverse_drops_ambiguous_b_in_umi1() {
    single_case(Strand::Reverse, "umi-1", b'B', 2);
}

/// One worker (`n_threads == 1`) uses the sequential reader. The same byte
/// must still drop the read instead of panicking on the caller.
#[test]
fn single_thread_drops_lowercase_g_in_r1_umi1() {
    paired_case(true, "umi-1", b'g', 1);
}

/// A file whose only record is malformed must still finish and write an
/// edgelist (with no rows), rather than aborting the sample.
#[test]
fn only_malformed_read_still_writes_edgelist() {
    let assay = proxiome_v2_assay();
    let mut pairs = proxiome_v2_read_pairs(1, valid_mix(), 3);
    let index = last_base(&assay, Strand::Forward, "umi-1");
    poke(&mut pairs[0].0, index, b'a');

    let dir = tempfile::tempdir().expect("tempdir");
    let r1_path = dir.path().join("r1.fastq.gz");
    let r2_path = dir.path().join("r2.fastq.gz");
    std::fs::write(
        &r1_path,
        gzip(&records_to_fastq(std::slice::from_ref(&pairs[0].0))),
    )
    .expect("write r1");
    std::fs::write(
        &r2_path,
        gzip(&records_to_fastq(std::slice::from_ref(&pairs[0].1))),
    )
    .expect("write r2");

    let run = run_edgelist(
        FastqInput::Paired {
            r1: open_gz(&r1_path),
            r2: open_gz(&r2_path),
        },
        2,
    );
    assert_eq!(run.kept, 0);
    assert_eq!(run.dropped, 1);
    assert_eq!(run.parquet_rows, 0);
}

/// A non-ACGT byte outside the pid/umi windows is not packed. It must not
/// abort the run or drop the read. Uppercase `N` in a linker is already
/// ignored; this locks the same boundary for any other byte.
#[test]
fn ambiguous_base_outside_pid_umi_is_kept() {
    let assay = proxiome_v2_assay();
    let mut pairs = proxiome_v2_read_pairs(1, valid_mix(), 5);
    let (lbs_start, _) = assay.forward_bounds("lbs-1").expect("lbs-1");
    poke(&mut pairs[0].0, lbs_start, b'R');

    let dir = tempfile::tempdir().expect("tempdir");
    let r1_path = dir.path().join("r1.fastq.gz");
    let r2_path = dir.path().join("r2.fastq.gz");
    std::fs::write(
        &r1_path,
        gzip(&records_to_fastq(std::slice::from_ref(&pairs[0].0))),
    )
    .expect("write r1");
    std::fs::write(
        &r2_path,
        gzip(&records_to_fastq(std::slice::from_ref(&pairs[0].1))),
    )
    .expect("write r2");

    let run = run_edgelist(
        FastqInput::Paired {
            r1: open_gz(&r1_path),
            r2: open_gz(&r2_path),
        },
        2,
    );
    assert_eq!(run.kept, 1);
    assert_eq!(run.dropped, 0);
    assert_eq!(run.parquet_rows, 1);
}
