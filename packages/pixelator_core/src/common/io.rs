use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::StringArray;
use arrow::array::{Array, BooleanArray, PrimitiveArray, RecordBatch, UInt64Array};
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, UInt64Type};
use itertools::Itertools;
use log::info;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReader,
    ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::file::properties::WriterProperties;

use crate::common::graph::Graph;
use crate::common::node_indexing::UmiToNodeIndexMapping;
use crate::common::node_partitioning::NodePartitioning;
use crate::common::types::{EdgeWeight, NodeIdx, UMI, UMIPair};

type IoError = Box<dyn std::error::Error + Send + Sync>;

const UMI_COLUMNS: [&str; 2] = ["umi1", "umi2"];
const READ_BATCH_SIZE: usize = 8192;

/// Open an edge list and check that its `umi1` and `umi2` columns are plain `UInt64`
/// columns without nulls.
///
/// Readers must be built from the returned file and metadata, so that they read the
/// same file that was checked even if the path is replaced in the meantime.
fn open_edgelist<P: AsRef<Path>>(path: P) -> Result<(File, ArrowReaderMetadata), IoError> {
    let file = File::open(path)?;
    let metadata = ArrowReaderMetadata::load(&file, ArrowReaderOptions::default())?;

    let schema = metadata.schema();
    let mut umi_roots = Vec::with_capacity(UMI_COLUMNS.len());
    for name in UMI_COLUMNS {
        let index = schema
            .index_of(name)
            .map_err(|_| format!("edge list has no {name} column"))?;
        let data_type = schema.field(index).data_type();
        if data_type != &DataType::UInt64 {
            return Err(
                format!("edge list column {name} has type {data_type}, expected UInt64").into(),
            );
        }
        umi_roots.push(index);
    }

    if umi_columns_may_contain_nulls(&metadata)? {
        let mask = ProjectionMask::roots(metadata.parquet_schema(), umi_roots);
        let reader =
            ParquetRecordBatchReaderBuilder::new_with_metadata(file.try_clone()?, metadata.clone())
                .with_projection(mask)
                .with_batch_size(READ_BATCH_SIZE)
                .build()?;
        for batch in reader {
            umi_columns(&batch?)?;
        }
    }

    Ok((file, metadata))
}

/// Use the parquet schema and column chunk statistics to rule out nulls in the UMI
/// columns without reading them. Returns `true` when that is not possible.
fn umi_columns_may_contain_nulls(metadata: &ArrowReaderMetadata) -> Result<bool, IoError> {
    let parquet_metadata = metadata.metadata();
    let leaves = parquet_metadata
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .enumerate()
        .filter_map(|(leaf, column)| {
            let parts = column.path().parts();
            UMI_COLUMNS
                .iter()
                .find(|name| parts.len() == 1 && parts[0] == **name)
                .map(|name| (leaf, *name, column.max_def_level()))
        })
        .collect::<Vec<_>>();
    if leaves.len() < UMI_COLUMNS.len() {
        return Ok(true);
    }

    let mut may_contain_nulls = false;
    for (leaf, name, max_def_level) in leaves {
        if max_def_level == 0 {
            continue;
        }
        for row_group in parquet_metadata.row_groups() {
            match row_group
                .column(leaf)
                .statistics()
                .and_then(|stats| stats.null_count_opt())
            {
                Some(0) => {}
                Some(_) => {
                    return Err(format!("edge list column {name} contains null values").into());
                }
                None => may_contain_nulls = true,
            }
        }
    }
    Ok(may_contain_nulls)
}

/// Get the `umi1` and `umi2` columns of a batch, failing if either is missing, is not
/// `UInt64`, or contains nulls.
fn umi_columns(batch: &RecordBatch) -> Result<(UInt64Array, UInt64Array), IoError> {
    let column = |name: &str| -> Result<UInt64Array, IoError> {
        let column = batch
            .column_by_name(name)
            .ok_or_else(|| format!("edge list has no {name} column"))?;
        let column = column
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                format!(
                    "edge list column {name} has type {}, expected UInt64",
                    column.data_type()
                )
            })?;
        if column.null_count() > 0 {
            return Err(format!("edge list column {name} contains null values").into());
        }
        Ok(column.clone())
    };
    Ok((column(UMI_COLUMNS[0])?, column(UMI_COLUMNS[1])?))
}

/// An iterator that yields UMI pairs from a parquet file one by one.
/// These can be mapped to node indices and used to construct a graph.
///
/// [`ParquetUMIPairIter::new`] rejects files whose UMI columns are missing, are not
/// `UInt64`, or contain nulls. If a batch still fails to read, iteration stops early and
/// the error is available from [`ParquetUMIPairIter::take_error`].
pub struct ParquetUMIPairIter {
    reader: ParquetRecordBatchReader,

    expected_size: i64,

    // Buffers for the currently loaded batch
    // We hold the specific typed arrays to avoid downcasting on every row
    col_src: Option<PrimitiveArray<UInt64Type>>,
    col_dst: Option<PrimitiveArray<UInt64Type>>,

    // Pointers for iteration logic
    current_idx: usize,
    batch_len: usize,

    finished: bool,
    error: Option<IoError>,
}

impl ParquetUMIPairIter {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, IoError> {
        let (file, metadata) = open_edgelist(path)?;
        let num_rows = metadata.metadata().file_metadata().num_rows();
        let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata)
            .with_batch_size(READ_BATCH_SIZE)
            .build()?;

        Ok(Self {
            reader,
            expected_size: num_rows,
            col_src: None,
            col_dst: None,
            current_idx: 0,
            batch_len: 0,
            finished: false,
            error: None,
        })
    }

    /// Take the error that ended iteration early, if any.
    pub fn take_error(&mut self) -> Option<IoError> {
        self.error.take()
    }

    fn load_next_batch(&mut self) -> bool {
        let loaded = match self.reader.next() {
            Some(batch) => batch
                .map_err(IoError::from)
                .and_then(|batch| Ok((umi_columns(&batch)?, batch.num_rows()))),
            None => {
                self.finished = true;
                return false;
            }
        };

        match loaded {
            Ok(((src_array, dst_array), num_rows)) => {
                self.col_src = Some(src_array);
                self.col_dst = Some(dst_array);

                self.batch_len = num_rows;
                self.current_idx = 0;
                true
            }
            Err(error) => {
                self.finished = true;
                self.error = Some(error);
                false
            }
        }
    }
}

impl Iterator for ParquetUMIPairIter {
    type Item = UMIPair;

    fn next(&mut self) -> Option<Self::Item> {
        // If we are at the end of the current batch, load the next non-empty one
        while self.current_idx >= self.batch_len {
            if self.finished || !self.load_next_batch() {
                return None; // No more batches
            }
        }

        let src = self.col_src.as_ref()?.value(self.current_idx);
        let dst = self.col_dst.as_ref()?.value(self.current_idx);

        self.current_idx += 1;

        Some((src as UMI, dst as UMI))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.expected_size as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for ParquetUMIPairIter {}

/// Write an iterator of record batches to a parquet file at the given path.
///
/// The `properties` argument configures the underlying parquet writer (e.g.
/// compression); pass `None` to use the writer's defaults.
pub fn write_record_batches_to_path<P: AsRef<Path>, I>(
    path: P,
    schema: SchemaRef,
    record_batches: I,
    properties: Option<WriterProperties>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    I: Iterator<Item = RecordBatch>,
{
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, properties)?;

    for batch in record_batches {
        writer.write(&batch)?;
    }

    writer.close()?;
    Ok(())
}

/// Remove crossing edges from the provided edge list and write the result to
/// a new parquet file.
///
/// # Arguments
/// * `input_edgelist_path`: path to the edge list to filtler
/// * `output_path`: path where the resulting filtered edgelist will be written
/// * `node_partitioning`: partitioning to be used for detecting crossing edges
/// * `mapping`: mapping between umis and node indices
pub fn filter_out_crossing_edges_from_edge_list<PIn: AsRef<Path>, POut: AsRef<Path>, T>(
    input_edgelist_path: &PIn,
    output_path: &POut,
    node_partitioning: &T,
    mapping: &UmiToNodeIndexMapping,
) -> Result<(), IoError>
where
    T: NodePartitioning,
{
    let (file, metadata) = open_edgelist(input_edgelist_path)?;
    let mut fields = metadata
        .schema()
        .fields()
        .iter()
        .cloned()
        .collect::<Vec<Arc<Field>>>();
    let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata)
        .with_batch_size(READ_BATCH_SIZE)
        .build()?;

    let output_file = File::create(output_path)?;
    fields.push(Arc::new(Field::new("component", DataType::Utf8, true)));
    let new_schema = Arc::new(Schema::new(fields));

    let result =
        write_edges_within_partitions(reader, output_file, new_schema, node_partitioning, mapping);
    if result.is_err() {
        let _ = std::fs::remove_file(output_path);
    }
    result
}

fn write_edges_within_partitions<T>(
    reader: ParquetRecordBatchReader,
    output_file: File,
    new_schema: SchemaRef,
    node_partitioning: &T,
    mapping: &UmiToNodeIndexMapping,
) -> Result<(), IoError>
where
    T: NodePartitioning,
{
    let mut writer = ArrowWriter::try_new(output_file, new_schema.clone(), None)?;
    let node_to_partition = node_partitioning.get_node_to_partition_map();
    let partition_of = |umi: u64| -> Result<_, IoError> {
        mapping
            .get_node_index(umi as UMI)
            .and_then(|node| node_to_partition.get(node))
            .ok_or_else(|| format!("umi {umi} not found in umi mapping").into())
    };

    for batch in reader {
        let batch = batch?;
        let (umi1, umi2) = umi_columns(&batch)?;

        let component_col = umi1
            .values()
            .iter()
            .zip(umi2.values().iter())
            .map(|(&umi1, &umi2)| {
                let (c1, c2) = (partition_of(umi1)?, partition_of(umi2)?);
                Ok(if c1 == c2 { Some(c1.to_string()) } else { None })
            })
            .collect::<Result<StringArray, IoError>>()?;

        let mask: BooleanArray = component_col.iter().map(|c| c.is_some()).collect();
        let mut columns = batch.columns().to_vec();
        columns.push(Arc::new(component_col));
        let new_batch = RecordBatch::try_new(new_schema.clone(), columns)?;

        let filtered_batch = compute::filter_record_batch(&new_batch, &mask)?;

        if filtered_batch.num_rows() > 0 {
            writer.write(&filtered_batch)?;
        }
    }

    writer.close()?;

    Ok(())
}

pub fn write_node_partitions_to_parquet<P: AsRef<Path>, T>(
    path: P,
    node_partitioning: &T,
    mapping: &UmiToNodeIndexMapping,
    batch_size: Option<usize>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    T: NodePartitioning,
{
    let schema = Arc::new(Schema::new(vec![
        Field::new("umi", DataType::UInt64, false),
        Field::new("partition_id", DataType::UInt64, false),
    ]));

    let mapping_node_to_partition = node_partitioning
        .get_node_to_partition_map()
        .iter()
        .enumerate()
        .map(|(node_idx, partition_idx)| (mapping.map_node_index_to_umi(node_idx), partition_idx));

    let chunk_size = batch_size.unwrap_or(4096);
    let chunks = mapping_node_to_partition.chunks(chunk_size);

    let record_batches = chunks.into_iter().map(|chunk| {
        let mut umis: Vec<u64> = Vec::with_capacity(chunk_size);
        let mut partitions: Vec<u64> = Vec::with_capacity(chunk_size);

        for (umi, partition) in chunk {
            umis.push(umi as u64);
            partitions.push(*partition as u64);
        }

        let umi_array = Arc::new(UInt64Array::from(umis));
        let partition_array = Arc::new(UInt64Array::from(partitions));

        RecordBatch::try_new(schema.clone(), vec![umi_array, partition_array])
            .expect("Failed to build record batch")
    });

    write_record_batches_to_path(path, schema.clone(), record_batches, None)
}

pub fn create_graph_and_umi_mapping_from_parquet_file<T>(
    parquet_file: &str,
) -> Result<(UmiToNodeIndexMapping, Graph<T>), IoError>
where
    T: EdgeWeight,
{
    info!("Creating UMI mapping and graph edges...");
    let mut pairs = ParquetUMIPairIter::new(parquet_file)?;
    // Mean degree about 3, so the node count is near 2E/3.
    let num_pairs = pairs.size_hint().0;
    let mut umi_mapping = UmiToNodeIndexMapping::with_capacity(num_pairs.saturating_mul(2) / 3);
    let triplets: Vec<(NodeIdx, NodeIdx, T)> = pairs
        .by_ref()
        .map(|(src, dest)| {
            let (src, dest) = (
                umi_mapping.get_or_insert(src),
                umi_mapping.get_or_insert(dest),
            );
            (src.min(dest), src.max(dest), T::one())
        })
        .collect();
    if let Some(error) = pairs.take_error() {
        return Err(error);
    }

    let num_nodes = umi_mapping.get_num_of_nodes();
    info!("Creating graph...");
    let graph = Graph::<T>::from_undirected_triplets(triplets, num_nodes);
    info!(
        "Graph created with {} nodes, {} edge entries, total edge weight {}",
        graph.get_num_nodes(),
        graph.get_edge_entry_count(),
        graph.get_total_edge_weight()
    );
    Ok((umi_mapping, graph))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::common::node_partitioning::FastNodePartitioning;
    use crate::common::types::PartitionId;
    use itertools::izip;
    use parquet::basic::Compression;
    use tempfile::NamedTempFile;

    #[test]
    fn test_filter_edge_list() {
        let test_data = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test_data/mix_40cells_1pc_1000rows.parquet");
        let (umi_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(
            test_data
                .to_str()
                .expect("File name is not a valid UTF-8 string"),
        )
        .expect("Failed to read edge list");
        // Make up partitions based on the mod 4 parity of umis
        let partitioning = FastNodePartitioning::initialize_from_partitions(
            (0..graph.get_num_nodes())
                .map(|node_id| umi_mapping.map_node_index_to_umi(node_id) % 4)
                .collect::<Vec<PartitionId>>(),
        );

        let output_file = NamedTempFile::new().expect("Failed to create tmp file");
        let temp_file = std::fs::File::open(output_file.path()).unwrap();

        let _ = filter_out_crossing_edges_from_edge_list(
            &test_data,
            &output_file,
            &partitioning,
            &umi_mapping,
        );

        let reader_builder = ParquetRecordBatchReaderBuilder::try_new(temp_file).unwrap();
        let reader = reader_builder.build().unwrap();
        assert!(
            reader
                .flat_map(|batch| {
                    let batch = batch.unwrap();
                    let umi1 = batch
                        .column_by_name("umi1")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<PrimitiveArray<UInt64Type>>()
                        .unwrap()
                        .clone();
                    let umi2 = batch
                        .column_by_name("umi2")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<PrimitiveArray<UInt64Type>>()
                        .unwrap()
                        .clone();
                    let component = batch
                        .column_by_name("component")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap()
                        .into_iter()
                        .map(|s| s.unwrap().to_string())
                        .collect::<Vec<String>>();

                    izip!(umi1.into_iter(), umi2.into_iter(), component.into_iter()).collect::<Vec<(
                        Option<u64>,
                        Option<u64>,
                        String,
                    )>>(
                    )
                })
                .map(|(umi1, umi2, component)| (umi1.unwrap(), umi2.unwrap(), component))
                .all(
                    |(umi1, umi2, component)| (umi1 % 4).to_string() == component
                        && (umi2 % 4).to_string() == component
                )
        );
    }

    #[test]
    fn test_write_record_batches_to_path() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::UInt64,
            false,
        )]));

        let batches = vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(UInt64Array::from(vec![1, 2, 3]))],
            )
            .unwrap(),
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(UInt64Array::from(vec![4, 5]))],
            )
            .unwrap(),
        ];

        let output_file = NamedTempFile::new().expect("Failed to create tmp file");

        write_record_batches_to_path(
            output_file.path(),
            schema.clone(),
            batches.into_iter(),
            None,
        )
        .expect("Failed to write record batches");

        let temp_file = std::fs::File::open(output_file.path()).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(temp_file)
            .unwrap()
            .build()
            .unwrap();

        let values = reader
            .flat_map(|batch| {
                batch
                    .unwrap()
                    .column_by_name("value")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .iter()
                    .map(|v| v.unwrap())
                    .collect::<Vec<u64>>()
            })
            .collect::<Vec<u64>>();

        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_write_record_batches_to_path_uses_properties() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::UInt64,
            false,
        )]));

        let batches = vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(UInt64Array::from(vec![1, 2, 3]))],
            )
            .unwrap(),
        ];

        let properties = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let output_file = NamedTempFile::new().expect("Failed to create tmp file");

        write_record_batches_to_path(
            output_file.path(),
            schema.clone(),
            batches.into_iter(),
            Some(properties),
        )
        .expect("Failed to write record batches");

        let temp_file = std::fs::File::open(output_file.path()).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(temp_file).unwrap();

        assert_eq!(
            builder.metadata().row_group(0).column(0).compression(),
            Compression::SNAPPY,
        );
    }
}
