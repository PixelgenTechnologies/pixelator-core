//! Column errors in an edgelist parquet file must come back as `Err`.
//!
//! `ParquetUMIPairIter::new` and `filter_out_crossing_edges_from_edge_list` panic
//! today when `umi1` or `umi2` is missing, is not a `UInt64` array (a narrower
//! integer, a signed integer, or an Arrow dictionary), or contains a null. These
//! tests call those functions on files that have at least one row so the batch
//! path runs. They pass only when both functions reject the file without panicking
//! and the crossing-edge export does not create its output.

use std::fs::File;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Int64Array, PrimitiveDictionaryBuilder, RecordBatch, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, UInt64Type};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use super::{
    ParquetUMIPairIter, create_graph_and_umi_mapping_from_parquet_file,
    filter_out_crossing_edges_from_edge_list,
};
use crate::common::node_indexing::UmiToNodeIndexMapping;
use crate::common::node_partitioning::{FastNodePartitioning, NodePartitioning};
use crate::common::types::UMIPair;

fn write_batch(path: &Path, fields: Vec<Field>, columns: Vec<ArrayRef>) {
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns).expect("record batch");
    let file = File::create(path).expect("create parquet");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("arrow writer");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close parquet");
}

fn uint64_field(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::UInt64, nullable)
}

fn field_for(name: &str, column: &ArrayRef) -> Field {
    Field::new(name, column.data_type().clone(), column.null_count() > 0)
}

fn write_pair(path: &Path, umi1: ArrayRef, umi2: ArrayRef) {
    write_batch(
        path,
        vec![field_for("umi1", &umi1), field_for("umi2", &umi2)],
        vec![umi1, umi2],
    )
}

fn plain_pair(left: &[u64], right: &[u64]) -> (ArrayRef, ArrayRef) {
    (
        Arc::new(UInt64Array::from(left.to_vec())),
        Arc::new(UInt64Array::from(right.to_vec())),
    )
}

fn dictionary_uint64(values: &[u64]) -> ArrayRef {
    let mut builder = PrimitiveDictionaryBuilder::<Int32Type, UInt64Type>::new();
    for value in values {
        builder.append_value(*value);
    }
    Arc::new(builder.finish())
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "box-dyn panic payload".to_string()
    }
}

struct Edgelist {
    _dir: TempDir,
    path: std::path::PathBuf,
}

fn edgelist_from(write: impl FnOnce(&Path)) -> Edgelist {
    let dir = TempDir::new().expect("temp dir");
    let path = dir.path().join("edges.parquet");
    write(&path);
    Edgelist { _dir: dir, path }
}

fn assert_iterator_rejects(label: &str, path: &Path) {
    let mut iter = match ParquetUMIPairIter::new(path) {
        Err(error) => {
            assert!(
                !error.to_string().is_empty(),
                "{label}: rejection error is empty"
            );
            return;
        }
        Ok(iter) => iter,
    };

    let read = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let mut pairs = Vec::new();
        while let Some(pair) = iter.next() {
            pairs.push(pair);
            if pairs.len() == 4 {
                break;
            }
        }
        pairs
    }));

    match read {
        Err(payload) => panic!(
            "{label}: ParquetUMIPairIter panicked ({}); return the column error from ParquetUMIPairIter::new",
            panic_message(&*payload)
        ),
        Ok(pairs) => panic!(
            "{label}: ParquetUMIPairIter yielded {pairs:?}; return the column error from ParquetUMIPairIter::new"
        ),
    }
}

fn assert_graph_construction_does_not_panic(label: &str, path: &Path) {
    let text = path.to_str().expect("utf-8 path");
    let mut failures = Vec::new();
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        create_graph_and_umi_mapping_from_parquet_file::<u8>(text);
    }));
    if let Err(payload) = outcome {
        failures.push(format!(
            "create_graph_and_umi_mapping_from_parquet_file::<u8> panicked ({})",
            panic_message(&*payload)
        ));
    }
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        create_graph_and_umi_mapping_from_parquet_file::<usize>(text);
    }));
    if let Err(payload) = outcome {
        failures.push(format!(
            "create_graph_and_umi_mapping_from_parquet_file::<usize> panicked ({})",
            panic_message(&*payload)
        ));
    }
    assert!(
        failures.is_empty(),
        "{label}: {}; propagate the iterator error instead of panicking",
        failures.join("; ")
    );
}

fn assert_filter_rejects(label: &str, path: &Path) {
    let dir = TempDir::new().expect("temp dir");
    let output = dir.path().join("filtered.parquet");
    assert!(!output.exists());

    // Mirror hybrid community detection: build the UMI map from the iterator
    // first (nulls become zero there), then export. A mapping that is missing
    // those UMIs panics for a different reason and does not reach the null cell.
    let mut mapping = UmiToNodeIndexMapping::with_capacity(0);
    if let Ok(mut iter) = ParquetUMIPairIter::new(path) {
        let read = std::panic::catch_unwind(AssertUnwindSafe(|| {
            while let Some((src, dst)) = iter.next() {
                mapping.get_or_insert(src);
                mapping.get_or_insert(dst);
            }
        }));
        if read.is_err() {
            mapping = UmiToNodeIndexMapping::with_capacity(0);
        }
    }
    let partitioning =
        FastNodePartitioning::initialize_with_singlet_partitions(mapping.get_num_of_nodes());

    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        filter_out_crossing_edges_from_edge_list(&path, &output, &partitioning, &mapping)
    }));
    let result = match outcome {
        Ok(result) => result,
        Err(payload) => panic!(
            "{label}: filter_out_crossing_edges_from_edge_list panicked ({}); return Err and do not create {}",
            panic_message(&*payload),
            output.display()
        ),
    };
    assert!(
        result.is_err(),
        "{label}: filter_out_crossing_edges_from_edge_list accepted the edgelist"
    );
    assert!(
        !output.exists(),
        "{label}: filtered edgelist {} was created before the file was rejected",
        output.display()
    );
}

fn assert_rejected(label: &str, file: &Edgelist) {
    let mut failures = Vec::new();
    for check in [
        assert_iterator_rejects as fn(&str, &Path),
        assert_graph_construction_does_not_panic,
        assert_filter_rejects,
    ] {
        if let Err(payload) =
            std::panic::catch_unwind(AssertUnwindSafe(|| check(label, &file.path)))
        {
            failures.push(panic_message(&*payload));
        }
    }
    assert!(
        failures.is_empty(),
        "{label} was not rejected on every edgelist read:\n{}",
        failures.join("\n")
    );
}

fn uint32_umi1() -> Edgelist {
    edgelist_from(|path| {
        let umi1: ArrayRef = Arc::new(UInt32Array::from(vec![1u32, 2]));
        let (umi2, _) = plain_pair(&[3, 4], &[3, 4]);
        write_pair(path, umi1, umi2);
    })
}

fn uint32_umi2() -> Edgelist {
    edgelist_from(|path| {
        let (_, umi1) = plain_pair(&[1, 2], &[1, 2]);
        let umi2: ArrayRef = Arc::new(UInt32Array::from(vec![3u32, 4]));
        write_pair(path, umi1, umi2);
    })
}

fn int64_umi1() -> Edgelist {
    edgelist_from(|path| {
        let umi1: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2]));
        let (_, umi2) = plain_pair(&[3, 4], &[3, 4]);
        write_pair(path, umi1, umi2);
    })
}

fn dictionary_umi1() -> Edgelist {
    edgelist_from(|path| {
        let umi1 = dictionary_uint64(&[1, 1, 2]);
        let (_, umi2) = plain_pair(&[3, 4, 5], &[3, 4, 5]);
        write_pair(path, umi1, umi2);
    })
}

fn dictionary_umi2() -> Edgelist {
    edgelist_from(|path| {
        let (umi1, _) = plain_pair(&[1, 2, 3], &[1, 2, 3]);
        let umi2 = dictionary_uint64(&[4, 4, 5]);
        write_pair(path, umi1, umi2);
    })
}

fn missing_umi1() -> Edgelist {
    edgelist_from(|path| {
        let umi2: ArrayRef = Arc::new(UInt64Array::from(vec![1u64, 2]));
        write_batch(path, vec![field_for("umi2", &umi2)], vec![umi2]);
    })
}

fn missing_umi2() -> Edgelist {
    edgelist_from(|path| {
        let umi1: ArrayRef = Arc::new(UInt64Array::from(vec![1u64, 2]));
        write_batch(path, vec![field_for("umi1", &umi1)], vec![umi1]);
    })
}

fn null_umi1() -> Edgelist {
    edgelist_from(|path| {
        let umi1: ArrayRef = Arc::new(UInt64Array::from(vec![Some(1u64), None]));
        let umi2: ArrayRef = Arc::new(UInt64Array::from(vec![Some(3u64), Some(4)]));
        write_pair(path, umi1, umi2);
    })
}

fn null_umi2() -> Edgelist {
    edgelist_from(|path| {
        let umi1: ArrayRef = Arc::new(UInt64Array::from(vec![Some(1u64), Some(2)]));
        let umi2: ArrayRef = Arc::new(UInt64Array::from(vec![Some(3u64), None]));
        write_pair(path, umi1, umi2);
    })
}

/// One full reader batch (8192 rows) of non-null pairs, then a null.
fn null_after_first_batch(null_in_umi1: bool) -> Edgelist {
    edgelist_from(move |path| {
        let rows = 8193;
        let mut left = Vec::with_capacity(rows);
        let mut right = Vec::with_capacity(rows);
        for index in 0..rows {
            let value = Some((index as u64) + 1);
            left.push(value);
            right.push(Some((index as u64) + 10_000));
        }
        if null_in_umi1 {
            left[8192] = None;
        } else {
            right[8192] = None;
        }
        let umi1: ArrayRef = Arc::new(UInt64Array::from(left));
        let umi2: ArrayRef = Arc::new(UInt64Array::from(right));
        write_pair(path, umi1, umi2);
    })
}

#[test]
fn uint32_umi1_is_a_read_error() {
    assert_rejected("uint32 umi1", &uint32_umi1());
}

#[test]
fn uint32_umi2_is_a_read_error() {
    assert_rejected("uint32 umi2", &uint32_umi2());
}

#[test]
fn int64_umi1_is_a_read_error() {
    assert_rejected("int64 umi1", &int64_umi1());
}

#[test]
fn dictionary_umi1_is_a_read_error() {
    assert_rejected("dictionary umi1", &dictionary_umi1());
}

#[test]
fn dictionary_umi2_is_a_read_error() {
    assert_rejected("dictionary umi2", &dictionary_umi2());
}

#[test]
fn missing_umi1_is_a_read_error() {
    assert_rejected("missing umi1", &missing_umi1());
}

#[test]
fn missing_umi2_is_a_read_error() {
    assert_rejected("missing umi2", &missing_umi2());
}

#[test]
fn null_umi1_is_a_read_error() {
    assert_rejected("null umi1", &null_umi1());
}

#[test]
fn null_umi2_is_a_read_error() {
    assert_rejected("null umi2", &null_umi2());
}

#[test]
fn null_umi1_after_a_full_batch_is_a_read_error() {
    assert_rejected(
        "null umi1 in the second batch",
        &null_after_first_batch(true),
    );
}

#[test]
fn null_umi2_after_a_full_batch_is_a_read_error() {
    assert_rejected(
        "null umi2 in the second batch",
        &null_after_first_batch(false),
    );
}

#[test]
fn plain_uint64_edgelist_is_read() {
    let file = edgelist_from(|path| {
        let (umi1, umi2) = plain_pair(&[1, 3], &[2, 4]);
        write_pair(path, umi1, umi2);
    });
    let mut iter = ParquetUMIPairIter::new(&file.path).expect("open uint64 edgelist");
    let pairs = iter.by_ref().collect::<Vec<UMIPair>>();
    assert_eq!(pairs, vec![(1, 2), (3, 4)]);

    let mut mapping = UmiToNodeIndexMapping::with_capacity(4);
    mapping.get_or_insert(1);
    mapping.get_or_insert(2);
    mapping.get_or_insert(3);
    mapping.get_or_insert(4);
    let partitioning =
        FastNodePartitioning::initialize_with_singlet_partitions(mapping.get_num_of_nodes());
    let dir = TempDir::new().expect("temp dir");
    let output = dir.path().join("filtered.parquet");
    filter_out_crossing_edges_from_edge_list(&file.path, &output, &partitioning, &mapping)
        .expect("filter a uint64 edgelist");
    assert!(output.is_file());
}

#[test]
fn nullable_uint64_without_nulls_is_read() {
    let file = edgelist_from(|path| {
        let umi1: ArrayRef = Arc::new(UInt64Array::from(vec![Some(8u64), Some(9)]));
        let umi2: ArrayRef = Arc::new(UInt64Array::from(vec![Some(10u64), Some(11)]));
        // Keep the schema nullable even though every cell has a value.
        write_batch(
            path,
            vec![uint64_field("umi1", true), uint64_field("umi2", true)],
            vec![umi1, umi2],
        );
    });
    let mut iter = ParquetUMIPairIter::new(&file.path).expect("open nullable uint64 edgelist");
    assert_eq!(iter.next(), Some((8, 10)));
    assert_eq!(iter.next(), Some((9, 11)));
    assert_eq!(iter.next(), None);
}
