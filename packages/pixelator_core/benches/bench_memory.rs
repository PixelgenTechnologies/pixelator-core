use divan::Bencher;
use pixelator_core::common::io::create_graph_and_umi_mapping_from_parquet_file;
use pixelator_core::fast_label_propagation::strategies::{
    AssignmentStrategy, DefaultAssignmentStrategy,
};
use std::path::PathBuf;

use pixelator_core::common::node_partitioning::{
    FastNodePartitioning, LeidenNodePartitioning, NodePartitioning,
};
use pixelator_core::fast_label_propagation::algorithm::fast_label_propagation;
use pixelator_core::leiden::algorithm::leiden;
use pixelator_core::leiden::quality::modularity::Modularity;
use pixelator_core::leiden::weighted_partitioned_graph::WeightedPartitionedGraph;

fn parquet_path(file_name: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("test_data");
    path.push(file_name);
    path
}

/// Peak live heap while building a graph from parquet.
#[divan::bench(sample_count = 1, sample_size = 1)]
fn mem_graph_construction() {
    let path = parquet_path("mix_40cells_0pc.parquet");
    let _ = create_graph_and_umi_mapping_from_parquet_file::<u8>(
        path.to_str().expect("Failed to convert PathBuf to &str"),
    );
}

/// Peak extra heap during FLP; graph load is not counted.
#[divan::bench(sample_count = 1, sample_size = 1)]
fn mem_fast_label_propagation(bencher: Bencher) {
    let path = parquet_path("mix_40cells_0pc.parquet");
    let (_umi_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(
        path.to_str().expect("Failed to convert PathBuf to &str"),
    )
    .expect("Failed to read edge list");
    drop(_umi_mapping);
    let partitioning =
        FastNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
    let assignment_strategy: &dyn AssignmentStrategy<FastNodePartitioning> =
        &DefaultAssignmentStrategy;

    bencher
        .with_inputs(|| partitioning.clone())
        .bench_local_values(|partitioning| {
            fast_label_propagation(&graph, 1, assignment_strategy, partitioning);
        });
}

/// Peak extra heap during Leiden; graph load and setup are not counted.
#[divan::bench(sample_count = 1, sample_size = 1)]
fn mem_leiden_modularity(bencher: Bencher) {
    let path = parquet_path("mix_40cells_1pc.parquet");
    let (_umi_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<usize>(
        path.to_str().expect("Failed to convert PathBuf to &str"),
    )
    .expect("Failed to read edge list");
    drop(_umi_mapping);
    let partitioning =
        LeidenNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
    let quality = Modularity::new(0.1, graph.get_total_edge_weight());
    let wp_graph = WeightedPartitionedGraph::new(graph, partitioning, quality, None, None);

    bencher
        .with_inputs(|| wp_graph.clone())
        .bench_local_refs(|wp_graph| leiden(wp_graph, 1.0, None, None));
}
