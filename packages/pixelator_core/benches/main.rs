use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

mod bench_parquet_io {
    use divan::Bencher;
    use pixelator_core::common::io::{
        ParquetUMIPairIter, create_graph_and_umi_mapping_from_parquet_file,
        write_node_partitions_to_parquet,
    };
    use pixelator_core::common::node_indexing::UmiToNodeIndexMapping;
    use pixelator_core::common::node_partitioning::{FastNodePartitioning, NodePartitioning};
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    #[divan::bench(sample_count = 10)]
    fn bench_parquet_reading(bencher: Bencher) {
        let mut parquet_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        parquet_path.push("test_data/mix_40cells_0pc.parquet");

        bencher.bench_local(move || {
            let _edge_iter: Vec<_> = ParquetUMIPairIter::new(
                parquet_path
                    .to_str()
                    .expect("Failed to convert PathBuf to &str"),
            )
            .expect("Failed to create ParquetEdgeIter")
            .collect();
        });
    }

    #[divan::bench(sample_count = 10)]
    fn bench_parquet_writing(bencher: Bencher) {
        let tmp_file = NamedTempFile::new().expect("Failed to create tmp file");
        let path = tmp_file.path();
        let n = 10000000;
        let random_umi_pairs: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();

        let node_partitioning = FastNodePartitioning::initialize_with_singlet_partitions(n);
        let mapping = UmiToNodeIndexMapping::from_umi_pairs(&random_umi_pairs);

        bencher.bench_local(move || {
            let res = write_node_partitions_to_parquet(path, &node_partitioning, &mapping, None);
            assert!(
                res.is_ok(),
                "Failed to write node partitions to parquet: {:?}",
                res.err()
            );
        });
    }

    #[divan::bench(sample_count = 10)]
    fn bench_create_graph_from_parquet(bencher: Bencher) {
        let mut parquet_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        parquet_path.push("test_data/mix_40cells_0pc.parquet");

        bencher.bench_local(move || {
            let _ = create_graph_and_umi_mapping_from_parquet_file::<u8>(
                parquet_path
                    .to_str()
                    .expect("Failed to convert PathBuf to &str"),
            );
        });
    }
}

mod bench_community_detection {
    use csv::Reader;
    use divan::Bencher;
    use pixelator_core::common::io::create_graph_and_umi_mapping_from_parquet_file;
    use pixelator_core::fast_label_propagation::strategies::{
        AssignmentStrategy, DefaultAssignmentStrategy,
    };
    use std::fs::File;
    use std::path::PathBuf;

    use pixelator_core::common::graph::Graph;
    use pixelator_core::common::node_partitioning::{
        FastNodePartitioning, LeidenNodePartitioning, NodePartitioning,
    };
    use pixelator_core::common::types::Edge;
    use pixelator_core::fast_label_propagation::algorithm::fast_label_propagation;
    use pixelator_core::leiden::algorithm::leiden;
    use pixelator_core::leiden::quality::cpm::ConstantPottsModel;
    use pixelator_core::leiden::quality::modularity::Modularity;
    use pixelator_core::leiden::weighted_partitioned_graph::WeightedPartitionedGraph;

    pub fn read_edges_from_csv(path: &str) -> impl Iterator<Item = Edge<u8>> {
        let file = File::open(path).expect("Failed to open CSV file");
        let mut rdr = Reader::from_reader(file);

        let records: Vec<_> = rdr.records().filter_map(|result| result.ok()).collect();

        records.into_iter().filter_map(|record| {
            if record.len() >= 2 {
                let src = record[0].parse::<usize>().ok()?;
                let dst = record[1].parse::<usize>().ok()?;
                Some(Edge::new(src, dst, None))
            } else {
                None
            }
        })
    }

    pub fn read_partitions_from_csv(path: &str) -> Vec<usize> {
        let file = File::open(path).expect("Failed to open CSV file");
        let mut rdr = Reader::from_reader(file);

        let records: Vec<_> = rdr.records().filter_map(|result| result.ok()).collect();

        let mut partitions: Vec<usize> = vec![0; records.len()];

        records.into_iter().for_each(|record| {
            let node_idx = record[0]
                .parse::<usize>()
                .expect("Failed to parse node index");
            let partition_id = record[1]
                .parse::<usize>()
                .expect("Failed to parse partition id");
            partitions[node_idx] = partition_id;
        });

        partitions
    }

    fn load_graph_from_edgelist_csv() -> Graph<u8> {
        let mut edgelist_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        edgelist_path.push("test_data/sbm_graph.edgelist.csv");

        let edges = read_edges_from_csv(
            edgelist_path
                .to_str()
                .expect("Failed to convert PathBuf to &str"),
        );
        Graph::<u8>::from_edges(edges, 3000)
    }

    fn load_partitions_from_csv() -> LeidenNodePartitioning {
        let mut partitions_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        partitions_path.push("test_data/sbm_graph_partitions.csv");

        let partitions = read_partitions_from_csv(
            partitions_path
                .to_str()
                .expect("Failed to convert PathBuf to &str"),
        );
        LeidenNodePartitioning::initialize_from_partitions(partitions)
    }

    #[divan::bench(sample_count = 100)]
    fn bench_fast_label_propagation(bencher: Bencher) {
        let graph: Graph<u8> = Into::<Graph<u8>>::into(load_graph_from_edgelist_csv());
        let partitioning =
            FastNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());

        let assignment_strategy: &dyn AssignmentStrategy<FastNodePartitioning> =
            &DefaultAssignmentStrategy;

        bencher.bench_local(move || {
            fast_label_propagation(&graph.clone(), 1, assignment_strategy, partitioning.clone());
        });
    }

    #[ignore]
    #[divan::bench(sample_count = 10)]
    fn bench_leiden_cpm(bencher: Bencher) {
        let graph: Graph<usize> = Into::<Graph<usize>>::into(load_graph_from_edgelist_csv());
        let partitioning: LeidenNodePartitioning = load_partitions_from_csv();
        let quality = ConstantPottsModel { resolution: 0.1 };
        let randomness = 1.;
        let wp_graph = WeightedPartitionedGraph::new(graph, partitioning, quality, None, None);

        bencher.bench_local(move || leiden(&mut wp_graph.clone(), randomness, Some(200), None));
    }

    #[divan::bench(sample_count = 100)]
    fn bench_leiden_modularity(bencher: Bencher) {
        let graph: Graph<usize> = Into::<Graph<usize>>::into(load_graph_from_edgelist_csv());
        let partitioning: LeidenNodePartitioning = load_partitions_from_csv();
        let quality = Modularity::new(0.1, graph.get_total_edge_weight());
        let randomness = 1.;
        let wp_graph = WeightedPartitionedGraph::new(graph, partitioning, quality, None, None);

        bencher.bench_local(move || leiden(&mut wp_graph.clone(), randomness, None, None));
    }

    #[divan::bench(sample_count = 3)]
    fn bench_leiden_modularity_medium(bencher: Bencher) {
        let mut parquet_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        parquet_path.push("test_data/mix_40cells_1pc.parquet");
        let (_umi_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<usize>(
            parquet_path
                .to_str()
                .expect("Failed to convert PathBuf to &str"),
        );
        let partitioning =
            LeidenNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
        let quality = Modularity::new(0.1, graph.get_total_edge_weight());
        let randomness = 1.;
        let wp_graph = WeightedPartitionedGraph::new(graph, partitioning, quality, None, None);

        bencher.bench_local(move || leiden(&mut wp_graph.clone(), randomness, None, None));
    }
}
