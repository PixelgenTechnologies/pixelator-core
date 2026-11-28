//! Spec for the edgelist loader's use of the parquet footer row count.
//!
//! These tests fail while `ParquetUMIPairIter` reports `FileMetaData::num_rows`
//! through `ExactSizeIterator` and `create_graph_and_umi_mapping_from_parquet_file`
//! reserves its node map and edge buffer from that hint. They pass once loading
//! a file uses the rows that were actually stored.
//!
//! Every graph entry point goes through that function before it does anything
//! else with the path:
//! - `find_graph_statistics`, `run_connected_components`, `run_label_propagation`,
//!   and `run_hybrid_community_detection` instantiate it with edge weight `u8`
//! - `run_leiden` instantiates it with edge weight `usize`
//! - the CLI `stats` and `flp` commands use `u8`; the CLI `leiden` command uses `usize`
//!
//! The Python and CLI wrappers are not invoked separately. Each one passes the
//! caller-supplied path straight into this function.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::NamedTempFile;

use super::io::{
    ParquetUMIPairIter, create_graph_and_umi_mapping_from_parquet_file,
    write_record_batches_to_path,
};

const PROBE_ENV: &str = "PIXELATOR_FOOTER_ROW_COUNT_PROBE";

/// One stored UMI pair. The footer row count is rewritten after the file is written.
fn write_one_edge(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("umi1", DataType::UInt64, false),
        Field::new("umi2", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![1_u64])),
            Arc::new(UInt64Array::from(vec![2_u64])),
        ],
    )
    .expect("one-edge batch");
    write_record_batches_to_path(path, schema, std::iter::once(batch), None)
        .expect("write one-edge edgelist");
}

/// Replace only the file-level `num_rows` field in a parquet footer.
///
/// Row-group metadata, column pages, and their absolute offsets stay as written.
/// The thrift compact layout this walks is the one `parquet` 60.2.0 writes and reads
/// (`parquet::parquet_thrift` field headers, zigzag varints, and list/struct skipping).
fn rewrite_footer_num_rows(path: &Path, num_rows: i64) {
    let mut bytes = std::fs::read(path).expect("read parquet");
    let n = bytes.len();
    assert!(n >= 12, "parquet file is shorter than a footer");
    assert_eq!(&bytes[n - 4..], b"PAR1");
    let footer_len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    let footer_start = n - 8 - footer_len;
    assert_eq!(&bytes[..4], b"PAR1");
    let footer = bytes[footer_start..n - 8].to_vec();
    let rewritten = splice_file_num_rows(&footer, num_rows);
    bytes.truncate(footer_start);
    bytes.extend_from_slice(&rewritten);
    bytes.extend_from_slice(&(rewritten.len() as u32).to_le_bytes());
    bytes.extend_from_slice(b"PAR1");
    std::fs::write(path, bytes).expect("rewrite parquet footer");
}

fn splice_file_num_rows(footer: &[u8], num_rows: i64) -> Vec<u8> {
    let mut cursor = Cursor::new(footer);
    let mut last_field_id = 0_i16;
    loop {
        let (field_type, field_id) = cursor.read_field_begin(last_field_id);
        if field_type == FIELD_STOP {
            break;
        }
        if field_id == 3 {
            assert_eq!(field_type, FIELD_I64, "file num_rows is an i64");
            let value_start = cursor.index;
            let _old = cursor.read_zigzag_i64();
            let mut out = footer[..value_start].to_vec();
            write_zigzag_i64(&mut out, num_rows);
            out.extend_from_slice(&footer[cursor.index..]);
            return out;
        }
        cursor.skip(field_type, 64);
        last_field_id = field_id;
    }
    panic!("parquet footer has no file-level num_rows field");
}

/// File metadata `num_rows` is the crafted value, and the column pages still hold one pair.
fn assert_crafted_file(path: &Path, num_rows: i64) {
    let file = std::fs::File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("open crafted parquet");
    assert_eq!(
        builder.metadata().file_metadata().num_rows(),
        num_rows,
        "footer row count was not the crafted value"
    );
    let mut rows = 0_usize;
    for batch in builder.with_batch_size(8192).build().unwrap() {
        let batch = batch.unwrap();
        let umi1 = batch
            .column_by_name("umi1")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let umi2 = batch
            .column_by_name("umi2")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(umi1.values(), &[1]);
        assert_eq!(umi2.values(), &[2]);
        rows += batch.num_rows();
    }
    assert_eq!(rows, 1, "crafted file must still contain one stored pair");
}

fn assert_loaded_one_edge(path: &str, weight: &str) {
    match weight {
        "u8" => {
            let (_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(path);
            assert_eq!(graph.get_num_nodes(), 2);
            assert_eq!(graph.get_edge_entry_count(), 1);
            assert_eq!(graph.get_total_edge_weight(), 1);
        }
        "usize" => {
            let (_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<usize>(path);
            assert_eq!(graph.get_num_nodes(), 2);
            assert_eq!(graph.get_edge_entry_count(), 1);
            assert_eq!(graph.get_total_edge_weight(), 1);
        }
        other => panic!("unknown weight {other}"),
    }
}

/// Run `create_graph_and_umi_mapping_from_parquet_file` in a child of this test
/// harness. An allocation abort in-process would kill the harness; the child
/// exiting on that abort is the failure this spec records.
fn assert_child_loads_one_edge(path: &Path, weight: &str) {
    if std::env::var_os(PROBE_ENV).is_some() {
        let spec = std::env::var(PROBE_ENV).unwrap();
        let (weight, path) = spec.split_once('\n').unwrap();
        assert_loaded_one_edge(path, weight);
        let iter = ParquetUMIPairIter::new(path).expect("iterator");
        let (lower, upper) = iter.size_hint();
        assert!(
            lower <= 1 && upper.is_none_or(|n| n <= 1),
            "size hint {lower:?}/{upper:?} still follows the footer row count"
        );
        assert_eq!(iter.count(), 1);
        println!("PIXELATOR_FOOTER_PROBE_OK {weight}");
        return;
    }

    let manifest = std::env::current_exe().expect("test executable");
    let spec = format!("{weight}\n{}", path.display());
    let status = Command::new("timeout")
        .args(["--signal=KILL", "60"])
        .arg(&manifest)
        .arg("--exact")
        .arg(std::thread::current().name().unwrap_or("footer_row_count"))
        .env(PROBE_ENV, spec)
        .env("RUST_BACKTRACE", "0")
        .output()
        .expect("spawn loader child");
    let stderr = String::from_utf8_lossy(&status.stderr);
    let stdout = String::from_utf8_lossy(&status.stdout);
    assert!(
        status.status.success() && stdout.contains(&format!("PIXELATOR_FOOTER_PROBE_OK {weight}")),
        "loading a one-row edgelist whose footer num_rows disagrees with the stored rows \
         must return that one edge instead of reserving the footer count.\n\
         child status: {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        status.status
    );
}

fn crafted_edgelist(num_rows: i64) -> NamedTempFile {
    let file = NamedTempFile::new().expect("temp file");
    write_one_edge(file.path());
    rewrite_footer_num_rows(file.path(), num_rows);
    assert_crafted_file(file.path(), num_rows);
    file
}

#[test]
fn honest_footer_loads_the_single_stored_edge() {
    let file = NamedTempFile::new().unwrap();
    write_one_edge(file.path());
    assert_crafted_file(file.path(), 1);
    assert_loaded_one_edge(file.path().to_str().unwrap(), "u8");
    assert_loaded_one_edge(file.path().to_str().unwrap(), "usize");
}

#[test]
fn negative_footer_row_count_does_not_reserve_usize_max() {
    // `-1` is the footer value whose cast to `usize` is `usize::MAX`.
    let file = crafted_edgelist(-1);
    assert_child_loads_one_edge(file.path(), "u8");
    assert_child_loads_one_edge(file.path(), "usize");
}

#[test]
fn multibillion_footer_row_count_does_not_reserve_that_many_edges() {
    let file = crafted_edgelist(3_000_000_000);
    assert_child_loads_one_edge(file.path(), "u8");
    assert_child_loads_one_edge(file.path(), "usize");
}

const FIELD_STOP: u8 = 0;
const FIELD_BOOL_TRUE: u8 = 1;
const FIELD_BOOL_FALSE: u8 = 2;
const FIELD_BYTE: u8 = 3;
const FIELD_I16: u8 = 4;
const FIELD_I32: u8 = 5;
const FIELD_I64: u8 = 6;
const FIELD_DOUBLE: u8 = 7;
const FIELD_BINARY: u8 = 8;
const FIELD_LIST: u8 = 9;
const FIELD_SET: u8 = 10;
const FIELD_MAP: u8 = 11;
const FIELD_STRUCT: u8 = 12;
const FIELD_UUID: u8 = 13;

struct Cursor<'a> {
    buf: &'a [u8],
    index: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, index: 0 }
    }

    fn read_byte(&mut self) -> u8 {
        let byte = self.buf[self.index];
        self.index += 1;
        byte
    }

    fn read_vlq(&mut self) -> u64 {
        let first = self.read_byte();
        if first & 0x80 == 0 {
            return u64::from(first);
        }
        let mut in_progress = u64::from(first & 0x7f);
        let mut shift = 7;
        loop {
            let byte = self.read_byte();
            in_progress |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return in_progress;
            }
            shift += 7;
        }
    }

    fn read_zigzag_i64(&mut self) -> i64 {
        let val = self.read_vlq();
        (val >> 1) as i64 ^ -((val & 1) as i64)
    }

    /// Compact-protocol field header. Returns `(type, id)`. Type 0 is stop.
    fn read_field_begin(&mut self, last_field_id: i16) -> (u8, i16) {
        let header = self.read_byte();
        let field_type = header & 0x0f;
        if field_type == FIELD_STOP {
            return (FIELD_STOP, 0);
        }
        let delta = header >> 4;
        let id = if delta == 0 {
            self.read_zigzag_i64() as i16
        } else {
            last_field_id + i16::from(delta)
        };
        (field_type, id)
    }

    fn skip(&mut self, field_type: u8, depth: i32) {
        assert!(depth > 0, "thrift skip depth exhausted");
        match field_type {
            FIELD_BOOL_TRUE | FIELD_BOOL_FALSE => {}
            FIELD_BYTE => {
                let _ = self.read_byte();
            }
            FIELD_I16 | FIELD_I32 | FIELD_I64 => {
                let _ = self.read_vlq();
            }
            FIELD_DOUBLE => self.index += 8,
            FIELD_BINARY => {
                let len = self.read_vlq() as usize;
                self.index += len;
            }
            FIELD_STRUCT => loop {
                let (ty, _) = self.read_field_begin(0);
                if ty == FIELD_STOP {
                    break;
                }
                self.skip(ty, depth - 1);
            },
            FIELD_LIST | FIELD_SET => {
                let header = self.read_byte();
                if header == 0 {
                    return;
                }
                let element_type = header & 0x0f;
                let short_len = header >> 4;
                let len = if short_len == 15 {
                    self.read_vlq() as usize
                } else {
                    usize::from(short_len)
                };
                if element_type == 2 {
                    self.index += len;
                } else {
                    let field_type = element_type_to_field(element_type);
                    for _ in 0..len {
                        self.skip(field_type, depth - 1);
                    }
                }
            }
            FIELD_MAP => {
                let size = self.read_vlq() as usize;
                if size > 0 {
                    let kv = self.read_byte();
                    let key_type = kv >> 4;
                    let val_type = kv & 0x0f;
                    for _ in 0..size {
                        self.skip_element(key_type, depth - 1);
                        self.skip_element(val_type, depth - 1);
                    }
                }
            }
            FIELD_UUID => self.index += 16,
            other => panic!("unsupported thrift field type {other}"),
        }
    }

    fn skip_element(&mut self, element_type: u8, depth: i32) {
        if element_type == 2 {
            let _ = self.read_byte();
        } else {
            self.skip(element_type_to_field(element_type), depth);
        }
    }
}

fn element_type_to_field(element_type: u8) -> u8 {
    match element_type {
        2 => FIELD_BOOL_TRUE,
        3 => FIELD_BYTE,
        4 => FIELD_I16,
        5 => FIELD_I32,
        6 => FIELD_I64,
        7 => FIELD_DOUBLE,
        8 => FIELD_BINARY,
        9 => FIELD_LIST,
        10 => FIELD_SET,
        11 => FIELD_MAP,
        12 => FIELD_STRUCT,
        13 => FIELD_UUID,
        other => panic!("unsupported thrift element type {other}"),
    }
}

fn write_zigzag_i64(out: &mut Vec<u8>, val: i64) {
    let sign = i64::from(val < 0);
    let encoded = (((val ^ -sign) << 1) + sign) as u64;
    write_vlq(out, encoded);
}

fn write_vlq(out: &mut Vec<u8>, mut val: u64) {
    while val > 0x7f {
        out.push((val as u8) | 0x80);
        val >>= 7;
    }
    out.push(val as u8);
}
