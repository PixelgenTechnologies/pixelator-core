use divan::Bencher;
use pixelator_core::common::graph::Graph;
use pixelator_core::common::io::create_graph_and_umi_mapping_from_parquet_file;
use std::hint::black_box;
use std::path::PathBuf;

fn load_graph() -> Graph<u8> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("test_data/mix_40cells_0pc.parquet");
    let (_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(
        path.to_str().expect("Failed to convert PathBuf to &str"),
    )
    .expect("Failed to read edge list");
    graph
}

/// Component id of every node on the 40-cell mix.
#[divan::bench(sample_count = 10)]
fn bench_connected_components(bencher: Bencher) {
    let graph = load_graph();
    bencher.bench_local(|| black_box(graph.connected_components()));
}
