use std::sync::OnceLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use pixelator_core::common::graph::Graph;
use pixelator_core::common::io::create_graph_and_umi_mapping_from_parquet_file;
use pixelator_core::common::node_indexing::UmiToNodeIndexMapping;
use pixelator_core::common::types::NodeIdx;
use pixelator_core::graph_collapse::{GraphCollapseExt, find_collapse_target, graph_collapse};
use rayon::ThreadPoolBuilder;

/// 40-cell edgelist with no crossing edges, bundled under `test_data`.
const PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_data/mix_40cells_0pc.parquet"
);
const BOUNDS: [usize; 3] = [2, 4, 6];
const THREADS: usize = 8;
/// Same distance cutoff the Python collapse uses, checked as Hamming distance.
const HAMMING_THRESHOLD: u8 = 2;
/// Nodes above this degree are not searched, as in the default collapse.
const MAX_DEGREE: usize = 1;

struct Dataset {
    graph: Graph<u8>,
    mapping: UmiToNodeIndexMapping,
}

fn dataset() -> &'static Dataset {
    static DATASET: OnceLock<Dataset> = OnceLock::new();
    DATASET.get_or_init(|| {
        let (mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(PATH)
            .expect("Failed to read edge list");
        Dataset { graph, mapping }
    })
}

/// Nodes of degree at most `MAX_DEGREE`, the starts `find_collapse_target`
/// searches from.
fn start_nodes() -> &'static [NodeIdx] {
    static START_NODES: OnceLock<Vec<NodeIdx>> = OnceLock::new();
    START_NODES.get_or_init(|| {
        dataset()
            .graph
            .get_degrees()
            .iter()
            .enumerate()
            .filter_map(|(node_id, &degree)| (degree <= MAX_DEGREE).then_some(node_id))
            .collect()
    })
}

/// `graph_collapse` parallelizes over start nodes on the current Rayon pool.
/// A local pool keeps that at `THREADS` instead of the CPU count or
/// `RAYON_NUM_THREADS`.
fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(THREADS)
            .build()
            .expect("rayon thread pool")
    })
}

/// Exhaust the even-distance BFS from every start node. The parquet load and
/// the start node list are built once, outside the timed closure.
#[divan::bench(args = BOUNDS, sample_count = 3)]
fn bench_bounded_even_breadth_first_search(bencher: Bencher, bound: usize) {
    let graph = &dataset().graph;
    let start_nodes = start_nodes();

    bencher
        .counter(ItemsCount::new(start_nodes.len()))
        .bench_local(|| {
            let mut scratch = Default::default();
            let found: usize = start_nodes
                .iter()
                .map(|&node| {
                    graph
                        .bounded_even_breadth_first_search_iter(node, Some(bound), &mut scratch)
                        .count()
                })
                .sum();
            divan::black_box(found)
        });
}

/// First Hamming match within the bound, for every start node. Stops at that
/// match, unlike `bounded_even_breadth_first_search`, which exhausts the bound.
#[divan::bench(args = BOUNDS, sample_count = 3)]
fn bench_find_collapse_target(bencher: Bencher, bound: usize) {
    let dataset = dataset();
    let start_nodes = start_nodes();

    bencher
        .counter(ItemsCount::new(start_nodes.len()))
        .bench_local(|| {
            let mut scratch = Default::default();
            let found: usize = start_nodes
                .iter()
                .filter(|&&node| {
                    find_collapse_target(
                        &dataset.graph,
                        &dataset.mapping,
                        node,
                        bound,
                        HAMMING_THRESHOLD,
                        &mut scratch,
                    )
                    .is_some()
                })
                .count();
            divan::black_box(found)
        });
}

/// Full collapse: the target search plus chain flattening and loop breaking.
#[divan::bench(args = BOUNDS, sample_count = 3)]
fn bench_graph_collapse(bencher: Bencher, bound: usize) {
    let dataset = dataset();
    let start_nodes = start_nodes();
    let pool = pool();

    bencher
        .counter(ItemsCount::new(start_nodes.len()))
        .bench_local(|| {
            pool.install(|| {
                let (targets, _stats) = graph_collapse(
                    &dataset.graph,
                    &dataset.mapping,
                    bound,
                    HAMMING_THRESHOLD,
                    MAX_DEGREE,
                );
                divan::black_box(targets.len())
            })
        });
}
