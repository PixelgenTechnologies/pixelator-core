use mimalloc::MiMalloc;
use pixelator_core::common::io::create_graph_and_umi_mapping_from_parquet_file;
use pixelator_core::common::node_partitioning::{
    FastNodePartitioning, LeidenNodePartitioning, NodePartitioning,
};
use pixelator_core::fast_label_propagation::algorithm::fast_label_propagation;
use pixelator_core::fast_label_propagation::strategies::{
    AssignmentStrategy, DefaultAssignmentStrategy,
};
use pixelator_core::leiden::algorithm::leiden;
use pixelator_core::leiden::quality::modularity::Modularity;
use pixelator_core::leiden::weighted_partitioned_graph::WeightedPartitionedGraph;
use std::alloc::{GlobalAlloc, Layout};
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

struct TrackingAlloc {
    inner: MiMalloc,
}

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn add(size: usize) {
    let current = CURRENT.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(current, Ordering::Relaxed);
}

fn sub(size: usize) {
    CURRENT.fetch_sub(size, Ordering::Relaxed);
}

fn reset_peak() {
    PEAK.store(CURRENT.load(Ordering::SeqCst), Ordering::SeqCst);
}

fn peak_bytes() -> usize {
    PEAK.load(Ordering::SeqCst)
}

fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc(layout) };
        if !ptr.is_null() {
            add(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.inner.dealloc(ptr, layout) };
        sub(layout.size());
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc_zeroed(layout) };
        if !ptr.is_null() {
            add(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { self.inner.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            sub(layout.size());
            add(new_size);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: TrackingAlloc = TrackingAlloc { inner: MiMalloc };

fn test_data_path(file_name: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("test_data");
    path.push(file_name);
    path
}

fn parse_output_path() -> Option<PathBuf> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(path) = arg.strip_prefix("--output=") {
            return Some(PathBuf::from(path));
        }
        if arg == "--output" {
            return args.next().map(PathBuf::from);
        }
    }
    None
}

fn metric_json(name: &str, bytes: usize) -> String {
    format!(
        r#"{{"name": "{}", "value": {:.4}, "unit": "MiB"}}"#,
        name,
        bytes_to_mib(bytes)
    )
}

fn write_results(path: Option<&Path>, results: &[(&str, usize)]) {
    let body = results
        .iter()
        .map(|(name, bytes)| metric_json(name, *bytes))
        .collect::<Vec<_>>()
        .join(",\n  ");
    let json = format!("[\n  {}\n]\n", body);

    match path {
        Some(path) => {
            fs::write(path, &json).unwrap_or_else(|err| {
                panic!(
                    "Failed to write memory bench JSON to {}: {err}",
                    path.display()
                )
            });
        }
        None => print!("{json}"),
    }

    for (name, bytes) in results {
        eprintln!("{name}: {:.4} MiB", bytes_to_mib(*bytes));
    }
}

fn main() {
    let output_path = parse_output_path();

    let construction_path = test_data_path("mix_40cells_0pc.parquet");
    reset_peak();
    let (_umi_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(
        construction_path
            .to_str()
            .expect("Failed to convert PathBuf to &str"),
    );
    black_box(&_umi_mapping);
    let mem_graph_construction = peak_bytes();
    drop(_umi_mapping);

    let partitioning =
        FastNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
    let assignment_strategy: &dyn AssignmentStrategy<FastNodePartitioning> =
        &DefaultAssignmentStrategy;
    reset_peak();
    let partitions = fast_label_propagation(&graph, 1, assignment_strategy, partitioning);
    black_box(&partitions);
    let mem_fast_label_propagation = peak_bytes();

    drop(partitions);
    drop(graph);

    let leiden_path = test_data_path("mix_40cells_1pc.parquet");
    let (_umi_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<usize>(
        leiden_path
            .to_str()
            .expect("Failed to convert PathBuf to &str"),
    );
    drop(_umi_mapping);
    let partitioning =
        LeidenNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
    let quality = Modularity::new(0.1, graph.get_total_edge_weight());
    let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, quality, None, None);
    reset_peak();
    let statistics = leiden(&mut wp_graph, 1.0, None, None);
    black_box(&statistics);
    black_box(&wp_graph);
    let mem_leiden_modularity = peak_bytes();

    write_results(
        output_path.as_deref(),
        &[
            ("mem_graph_construction", mem_graph_construction),
            ("mem_fast_label_propagation", mem_fast_label_propagation),
            ("mem_leiden_modularity", mem_leiden_modularity),
        ],
    );
}
