//! The stats, flp, and leiden commands take an edgelist path from the caller.
//! A missing, non-uint64, or null `umi1`/`umi2` column must exit 1. It must not
//! abort the process, and it must not create the partition file.

use std::fs::File;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use arrow::array::{ArrayRef, UInt32Array, UInt64Array};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

fn write_pair(path: &Path, umi1: ArrayRef, umi2: ArrayRef) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("umi1", umi1.data_type().clone(), umi1.null_count() > 0),
        Field::new("umi2", umi2.data_type().clone(), umi2.null_count() > 0),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![umi1, umi2]).expect("batch");
    let mut writer =
        ArrowWriter::try_new(File::create(path).expect("file"), schema, None).expect("writer");
    writer.write(&batch).expect("write");
    writer.close().expect("close");
}

fn uint32_edgelist(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("uint32.parquet");
    write_pair(
        &path,
        Arc::new(UInt32Array::from(vec![1u32, 2])),
        Arc::new(UInt64Array::from(vec![3u64, 4])),
    );
    path
}

fn null_edgelist(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("nulls.parquet");
    write_pair(
        &path,
        Arc::new(UInt64Array::from(vec![Some(1u64), None])),
        Arc::new(UInt64Array::from(vec![Some(3u64), Some(4)])),
    );
    path
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_community-detection"))
        .args(args)
        .output()
        .expect("run community-detection")
}

fn assert_rejects(label: &str, args: &[&str], output: Option<&Path>) {
    let result = run(args);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(
        result.status.code(),
        Some(1),
        "{label}: expected exit 1, got {:?}. stderr:\n{stderr}",
        result.status.code()
    );
    assert!(
        !stderr.contains("panicked at"),
        "{label}: process panicked:\n{stderr}"
    );
    if let Some(output) = output {
        assert!(
            !output.exists(),
            "{label}: created {} before rejecting the edgelist",
            output.display()
        );
    }
}

#[test]
fn stats_rejects_uint32_umi_column() {
    let dir = TempDir::new().expect("temp");
    let input = uint32_edgelist(dir.path());
    assert_rejects("stats uint32", &["stats", input.to_str().unwrap()], None);
}

#[test]
fn stats_rejects_null_umi_column() {
    let dir = TempDir::new().expect("temp");
    let input = null_edgelist(dir.path());
    assert_rejects("stats null", &["stats", input.to_str().unwrap()], None);
}

#[test]
fn flp_rejects_uint32_umi_column() {
    let dir = TempDir::new().expect("temp");
    let input = uint32_edgelist(dir.path());
    let output = dir.path().join("partitions.parquet");
    assert_rejects(
        "flp uint32",
        &[
            "flp",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ],
        Some(&output),
    );
}

#[test]
fn flp_rejects_null_umi_column() {
    let dir = TempDir::new().expect("temp");
    let input = null_edgelist(dir.path());
    let output = dir.path().join("partitions.parquet");
    assert_rejects(
        "flp null",
        &[
            "flp",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ],
        Some(&output),
    );
}

#[test]
fn leiden_rejects_uint32_umi_column() {
    let dir = TempDir::new().expect("temp");
    let input = uint32_edgelist(dir.path());
    let output = dir.path().join("partitions.parquet");
    assert_rejects(
        "leiden uint32",
        &[
            "leiden",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ],
        Some(&output),
    );
}

#[test]
fn leiden_rejects_null_umi_column() {
    let dir = TempDir::new().expect("temp");
    let input = null_edgelist(dir.path());
    let output = dir.path().join("partitions.parquet");
    assert_rejects(
        "leiden null",
        &[
            "leiden",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ],
        Some(&output),
    );
}
