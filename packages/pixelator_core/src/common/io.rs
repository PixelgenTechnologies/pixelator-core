use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::StringArray;
use arrow::array::{Array, BooleanArray, PrimitiveArray, RecordBatch, UInt64Array};
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, UInt64Type};
use itertools::Itertools;
use log::info;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};

use crate::common::graph::Graph;
use crate::common::node_indexing::UmiToNodeIndexMapping;
use crate::common::node_partitioning::NodePartitioning;
use crate::common::types::{EdgeWeight, UMI, UMIPair};

/// An iterator that yields UMI pairs from a parquet file one by one.
/// These can be mapped to node indices and used to construct a graph.
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
}

impl ParquetUMIPairIter {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let num_rows = builder.metadata().file_metadata().num_rows();
        let reader = builder.with_batch_size(8192).build()?;

        Ok(Self {
            reader,
            expected_size: num_rows,
            col_src: None,
            col_dst: None,
            current_idx: 0,
            batch_len: 0,
        })
    }

    fn load_next_batch(&mut self) -> bool {
        match self.reader.next() {
            Some(Ok(batch)) => {
                let src_array = batch
                    .column_by_name("umi1")
                    .expect("Could not find umi1 column in data")
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt64Type>>()
                    .expect("Column umi1 is not UInt64");

                let dst_array = batch
                    .column_by_name("umi2")
                    .expect("Could not find umi2 column in data")
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt64Type>>()
                    .expect("Column umi2 is not UInt64");

                self.col_src = Some(src_array.clone());
                self.col_dst = Some(dst_array.clone());

                self.batch_len = batch.num_rows();
                self.current_idx = 0;
                true
            }
            _ => false, // End of file or error
        }
    }
}

impl Iterator for ParquetUMIPairIter {
    type Item = UMIPair;

    fn next(&mut self) -> Option<Self::Item> {
        // If we are at the end of the current batch, load the next one
        if self.current_idx >= self.batch_len && !self.load_next_batch() {
            return None; // No more batches
        }

        let src = self.col_src.as_ref()?.value(self.current_idx);
        let dst = self.col_dst.as_ref()?.value(self.current_idx);

        self.current_idx += 1;

        Some((src as UMI, dst as UMI))
    }
}

impl ExactSizeIterator for ParquetUMIPairIter {
    fn len(&self) -> usize {
        self.expected_size as usize
    }
}

pub fn write_record_batches_to_path<P: AsRef<Path>, I>(
    path: P,
    schema: SchemaRef,
    record_batches: I,
) -> Result<(), Box<dyn std::error::Error>>
where
    I: Iterator<Item = RecordBatch>,
{
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;

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
) -> Result<(), Box<dyn std::error::Error>>
where
    T: NodePartitioning,
{
    let file = File::open(input_edgelist_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut fields = builder
        .schema()
        .fields()
        .iter()
        .cloned()
        .collect::<Vec<Arc<Field>>>();
    let mut reader = builder.with_batch_size(8192).build()?;

    let output_file = File::create(output_path)?;
    fields.push(Arc::new(Field::new("component", DataType::Utf8, true)));
    let new_schema = Arc::new(Schema::new(fields));
    let mut writer = ArrowWriter::try_new(output_file, new_schema.clone(), None)?;

    while let Some(Ok(batch)) = reader.next() {
        let component1_iter = batch
            .column_by_name("umi1")
            .expect("Could not find umi1 column in data")
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt64Type>>()
            .expect("Column umi1 is not UInt64")
            .iter()
            .map(|umi| {
                let umi = umi.expect("umi1 column contains null values");
                node_partitioning
                    .get_node_to_partition_map()
                    .get(mapping.map_umi_to_node_index(umi as UMI))
                    .unwrap_or_else(|| panic!("umi {} not found in umi mapping", umi))
            });

        let component2_iter = batch
            .column_by_name("umi2")
            .expect("Could not find umi2 column in data")
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt64Type>>()
            .expect("Column umi2 is not UInt64")
            .iter()
            .map(|umi| {
                let umi = umi.expect("umi2 column contains null values");
                node_partitioning
                    .get_node_to_partition_map()
                    .get(mapping.map_umi_to_node_index(umi as UMI))
                    .unwrap_or_else(|| panic!("umi {} not found in umi mapping", umi))
            });

        let component_col = StringArray::from(
            component1_iter
                .zip(component2_iter)
                .map(|(c1, c2)| if c1 == c2 { Some(c1.to_string()) } else { None })
                .collect::<Vec<Option<String>>>(),
        );

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
) -> Result<(), Box<dyn std::error::Error>>
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

    write_record_batches_to_path(path, schema.clone(), record_batches)
}

pub fn create_graph_and_umi_mapping_from_parquet_file<T>(
    parquet_file: &str,
) -> (UmiToNodeIndexMapping, Graph<T>)
where
    T: EdgeWeight,
{
    info!("Creating UMI mapping...");
    let umi_pair_iterator =
        ParquetUMIPairIter::new(parquet_file).expect("Failed to create ParquetUMIPairIter");
    let umis: Vec<UMIPair> = umi_pair_iterator.collect();

    let umi_mapping = UmiToNodeIndexMapping::from_umi_pairs(&umis);

    let num_nodes = umi_mapping.get_num_of_nodes();
    let edges = umi_mapping.map_umi_pair_iterator_to_edge(umis.iter().copied());

    info!("Creating graph...");
    let graph = Graph::<T>::from_edges(edges, num_nodes);
    info!(
        "Graph created with {} nodes, {} edge entries, total edge weight {}",
        graph.get_num_nodes(),
        graph.get_edge_entry_count(),
        graph.get_total_edge_weight()
    );
    (umi_mapping, graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::common::node_partitioning::FastNodePartitioning;
    use crate::common::types::PartitionId;
    use itertools::izip;
    use tempfile::NamedTempFile;

    #[test]
    fn test_filter_edge_list() {
        let test_data = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test_data/mix_40cells_1pc_1000rows.parquet");
        let (umi_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(
            test_data
                .to_str()
                .expect("File name is not a valid UTF-8 string"),
        );
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
}
