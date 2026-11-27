//! Benchmark harness for staged cell detection.
//!
//! Runs one pipeline variant on one ground-truth dataset pair (`*_0pc.parquet` gives the cells,
//! `*_1pc.parquet` is the input with crossing edges) and prints runtime, peak heap and quality
//! metrics. One variant per process, so heap and caches do not carry over between variants.
//!
//! ```text
//! cargo run --release -p pixelator-core --example evaluate_staged_cell_detection -- \
//!     --dataset 40cells --variant B1 --shuffle-seed 42
//! ```

use std::alloc::{GlobalAlloc, Layout};
use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

use mimalloc::MiMalloc;
use rand::prelude::*;
use rustc_hash::FxHashMap as HashMap;

use pixelator_core::common::constants::MIN_PNA_COMPONENT_SIZE;
use pixelator_core::common::graph::Graph;
use pixelator_core::common::io::{
    ParquetUMIPairIter, create_graph_and_umi_mapping_from_parquet_file,
};
use pixelator_core::common::node_indexing::UmiToNodeIndexMapping;
use pixelator_core::common::node_partitioning::{
    FastNodePartitioning, LeidenNodePartitioning, NodePartitioning,
};
use pixelator_core::common::types::NodeIdx;
use pixelator_core::fast_label_propagation::algorithm::fast_label_propagation;
use pixelator_core::fast_label_propagation::strategies::{
    AssignmentStrategy, DefaultAssignmentStrategy,
};
use pixelator_core::four_cycle_compression::algorithm::{
    FourCycleBridgeMode, four_cycle_compression,
};
use pixelator_core::leiden::algorithm::leiden;
use pixelator_core::leiden::quality::modularity::Modularity;
use pixelator_core::leiden::weighted_partitioned_graph::{
    AggregateOptions, WeightedPartitionedGraph,
};

const RESOLUTION: f64 = 1.0;
const RANDOMNESS: f64 = 0.1;
const LEIDEN_SEED: u64 = 0;
const FOUR_CYCLE_MAX_ROUNDS: usize = 4;

struct TrackingAllocator;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { MiMalloc.alloc(layout) };
        if !ptr.is_null() {
            let size = layout.size();
            let prev = LIVE_BYTES.fetch_add(size, AtomicOrdering::Relaxed);
            PEAK_BYTES.fetch_max(prev + size, AtomicOrdering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), AtomicOrdering::Relaxed);
        unsafe { MiMalloc.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator;

fn reset_peak_memory() -> usize {
    let cur = LIVE_BYTES.load(AtomicOrdering::Relaxed);
    PEAK_BYTES.store(cur, AtomicOrdering::Relaxed);
    cur
}

fn peak_delta_mb(base_live: usize) -> f64 {
    let peak = PEAK_BYTES.load(AtomicOrdering::Relaxed);
    peak.saturating_sub(base_live) as f64 / (1024.0 * 1024.0)
}

/// Load a Parquet edge list, optionally shuffling the edge rows before node indices are
/// assigned, so that node indices are not sorted by ground-truth cell.
fn load_graph_from_parquet_with_edge_order(
    parquet_file: &str,
    shuffle_edges_seed: Option<u64>,
) -> (UmiToNodeIndexMapping, Graph<u8>) {
    let Some(seed) = shuffle_edges_seed else {
        return create_graph_and_umi_mapping_from_parquet_file::<u8>(parquet_file);
    };

    let pairs_iter =
        ParquetUMIPairIter::new(parquet_file).expect("Failed to create ParquetUMIPairIter");
    let mut raw_pairs: Vec<_> = pairs_iter.collect();
    let mut rng = StdRng::seed_from_u64(seed);
    raw_pairs.shuffle(&mut rng);

    let est_nodes = raw_pairs.len().saturating_mul(2) / 3;
    let mut umi_mapping = UmiToNodeIndexMapping::with_capacity(est_nodes);
    let mut is_b = Vec::with_capacity(est_nodes);

    let triplets: Vec<(NodeIdx, NodeIdx, u8)> = raw_pairs
        .into_iter()
        .map(|(src, dest)| {
            let u = umi_mapping.get_or_insert(src);
            if u == is_b.len() {
                is_b.push(false);
            }
            let v = umi_mapping.get_or_insert(dest);
            if v == is_b.len() {
                is_b.push(true);
            }
            (u.min(v), u.max(v), 1u8)
        })
        .collect();

    let num_nodes = umi_mapping.get_num_of_nodes();
    let graph =
        Graph::<u8>::from_undirected_triplets(triplets, num_nodes).with_bipartite_side_mask(is_b);
    (umi_mapping, graph)
}

/// Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) between two labelings.
fn compute_ari_and_nmi(a: &[u32], b: &[u32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let nf = n as f64;

    let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::default();
    let mut a_counts: HashMap<u32, usize> = HashMap::default();
    let mut b_counts: HashMap<u32, usize> = HashMap::default();

    for (&la, &lb) in a.iter().zip(b.iter()) {
        *pair_counts.entry((la, lb)).or_insert(0) += 1;
        *a_counts.entry(la).or_insert(0) += 1;
        *b_counts.entry(lb).or_insert(0) += 1;
    }

    let choose2 = |k: usize| -> f64 {
        let kf = k as f64;
        kf * (kf - 1.0) * 0.5
    };

    let sum_nij: f64 = pair_counts.values().map(|&c| choose2(c)).sum();
    let sum_ai: f64 = a_counts.values().map(|&c| choose2(c)).sum();
    let sum_bj: f64 = b_counts.values().map(|&c| choose2(c)).sum();
    let total_pairs = choose2(n);
    let expected = (sum_ai * sum_bj) / total_pairs;
    let max_idx = 0.5 * (sum_ai + sum_bj);
    let ari = if (max_idx - expected).abs() < 1e-12 {
        1.0
    } else {
        (sum_nij - expected) / (max_idx - expected)
    };

    let mut mi = 0.0f64;
    for (&(la, lb), &nij) in &pair_counts {
        let p_ij = nij as f64 / nf;
        let p_i = a_counts[&la] as f64 / nf;
        let p_j = b_counts[&lb] as f64 / nf;
        mi += p_ij * (p_ij / (p_i * p_j)).ln();
    }
    let entropy = |counts: &HashMap<u32, usize>| -> f64 {
        counts
            .values()
            .map(|&c| {
                let p = c as f64 / nf;
                -p * p.ln()
            })
            .sum()
    };
    let h_a = entropy(&a_counts);
    let h_b = entropy(&b_counts);
    let nmi = if h_a + h_b < 1e-12 {
        1.0
    } else {
        2.0 * mi / (h_a + h_b)
    };

    (ari, nmi)
}

/// Everything needed to score a partition of the `1pc` graph against the `0pc` cells.
///
/// The edge list is kept apart from the graph so that a pipeline can consume the graph, as
/// `hybrid_community_detection` does.
struct Scoring {
    /// Ground-truth cell of every `1pc` node: its connected component in the `0pc` graph.
    cell_of_node: Vec<u32>,
    cell_sizes: Vec<u32>,
    /// Every undirected edge of the `1pc` graph once, as `(u, v)` with `u < v`.
    edges: Vec<(u32, u32)>,
    num_crossing_edges: usize,
}

struct FoldStats {
    num_groups: usize,
    crossing_folded: usize,
    intra_folded: usize,
}

struct FinalStats {
    num_components: usize,
    large_components: usize,
    nodes_in_small_components: usize,
    crossing_retained: usize,
    intra_preserved: usize,
    over_split_cells: usize,
    under_split_components: usize,
    ari: f64,
    nmi: f64,
}

impl Scoring {
    fn new(path_0pc: &str, mapping_1pc: &UmiToNodeIndexMapping, graph_1pc: &Graph<u8>) -> Self {
        let (mapping_0pc, graph_0pc) =
            create_graph_and_umi_mapping_from_parquet_file::<u8>(path_0pc);
        let comp_0pc = graph_0pc.connected_components();
        drop(graph_0pc);

        let n = graph_1pc.get_num_nodes();
        let cell_of_node: Vec<u32> = (0..n)
            .map(|u| {
                let umi = mapping_1pc.map_node_index_to_umi(u);
                comp_0pc[mapping_0pc.map_umi_to_node_index(umi)] as u32
            })
            .collect();
        drop(mapping_0pc);
        drop(comp_0pc);

        let num_cells = cell_of_node.iter().max().map_or(0, |&m| m as usize + 1);
        let mut cell_sizes = vec![0u32; num_cells];
        for &c in &cell_of_node {
            cell_sizes[c as usize] += 1;
        }

        let adj = graph_1pc.get_adjacency_matrix();
        let indptr_raw = adj.indptr();
        let indptr = indptr_raw.as_slice().unwrap();
        let indices = adj.indices();
        let mut edges = Vec::with_capacity(indices.len() / 2);
        let mut num_crossing_edges = 0usize;
        for u in 0..n {
            for &v in &indices[indptr[u]..indptr[u + 1]] {
                if u < v {
                    edges.push((u as u32, v as u32));
                    if cell_of_node[u] != cell_of_node[v] {
                        num_crossing_edges += 1;
                    }
                }
            }
        }

        Scoring {
            cell_of_node,
            cell_sizes,
            edges,
            num_crossing_edges,
        }
    }

    fn num_nodes(&self) -> usize {
        self.cell_of_node.len()
    }

    fn num_intra_edges(&self) -> usize {
        self.edges.len() - self.num_crossing_edges
    }

    fn is_crossing(&self, u: u32, v: u32) -> bool {
        self.cell_of_node[u as usize] != self.cell_of_node[v as usize]
    }

    /// Edges with both endpoints in one group of `groups`.
    fn fold_stats(&self, groups: &[usize], num_groups: usize) -> FoldStats {
        let mut crossing_folded = 0usize;
        let mut intra_folded = 0usize;
        for &(u, v) in &self.edges {
            if groups[u as usize] == groups[v as usize] {
                if self.is_crossing(u, v) {
                    crossing_folded += 1;
                } else {
                    intra_folded += 1;
                }
            }
        }
        FoldStats {
            num_groups,
            crossing_folded,
            intra_folded,
        }
    }

    /// Connected components of the graph once edges between communities are removed.
    fn final_components(&self, community_of_node: &[usize]) -> Vec<u32> {
        fn find(parents: &mut [u32], mut x: u32) -> u32 {
            while parents[x as usize] != x {
                parents[x as usize] = parents[parents[x as usize] as usize];
                x = parents[x as usize];
            }
            x
        }

        let n = self.num_nodes();
        let mut parents: Vec<u32> = (0..n as u32).collect();
        for &(u, v) in &self.edges {
            if community_of_node[u as usize] == community_of_node[v as usize] {
                let ru = find(&mut parents, u);
                let rv = find(&mut parents, v);
                if ru != rv {
                    parents[ru.max(rv) as usize] = ru.min(rv);
                }
            }
        }

        const UNSET: u32 = u32::MAX;
        let mut dense = vec![UNSET; n];
        let mut next = 0u32;
        let mut components = vec![0u32; n];
        for u in 0..n as u32 {
            let root = find(&mut parents, u) as usize;
            if dense[root] == UNSET {
                dense[root] = next;
                next += 1;
            }
            components[u as usize] = dense[root];
        }
        components
    }

    fn final_stats(&self, community_of_node: &[usize]) -> FinalStats {
        let components = self.final_components(community_of_node);
        let num_components = components.iter().max().map_or(0, |&m| m as usize + 1);

        let mut component_sizes = vec![0usize; num_components];
        for &c in &components {
            component_sizes[c as usize] += 1;
        }
        let large_components = component_sizes
            .iter()
            .filter(|&&size| size >= MIN_PNA_COMPONENT_SIZE)
            .count();
        let nodes_in_small_components = component_sizes
            .iter()
            .filter(|&&size| size < MIN_PNA_COMPONENT_SIZE)
            .sum();

        let mut crossing_retained = 0usize;
        let mut intra_preserved = 0usize;
        for &(u, v) in &self.edges {
            if components[u as usize] == components[v as usize] {
                if self.is_crossing(u, v) {
                    crossing_retained += 1;
                } else {
                    intra_preserved += 1;
                }
            }
        }

        // Nodes shared by each (cell, component) pair.
        let mut overlap: HashMap<(u32, u32), u32> = HashMap::default();
        for (&cell, &component) in self.cell_of_node.iter().zip(components.iter()) {
            *overlap.entry((cell, component)).or_insert(0) += 1;
        }
        // Over-split: two or more components each hold at least 10% of the cell.
        let mut big_parts_of_cell = vec![0u32; self.cell_sizes.len()];
        // Under-split: the component holds more than half of two or more cells.
        let mut cells_mostly_in_component = vec![0u32; num_components];
        for (&(cell, component), &shared) in &overlap {
            let cell_size = self.cell_sizes[cell as usize] as u64;
            if shared as u64 * 10 >= cell_size {
                big_parts_of_cell[cell as usize] += 1;
            }
            if shared as u64 * 2 > cell_size {
                cells_mostly_in_component[component as usize] += 1;
            }
        }
        let over_split_cells = big_parts_of_cell.iter().filter(|&&c| c >= 2).count();
        let under_split_components = cells_mostly_in_component
            .iter()
            .filter(|&&c| c >= 2)
            .count();

        let (ari, nmi) = compute_ari_and_nmi(&self.cell_of_node, &components);

        FinalStats {
            num_components,
            large_components,
            nodes_in_small_components,
            crossing_retained,
            intra_preserved,
            over_split_cells,
            under_split_components,
            ari,
            nmi,
        }
    }
}

struct StageRecord {
    name: &'static str,
    secs: f64,
    /// Peak heap during the stage, above the heap held before the pipeline started.
    peak_mb: f64,
    fold: Option<FoldStats>,
}

/// Times each stage of a pipeline and tracks its peak heap.
struct Recorder {
    base_live: usize,
    stages: Vec<StageRecord>,
}

impl Recorder {
    fn new() -> Self {
        Recorder {
            base_live: reset_peak_memory(),
            stages: Vec::new(),
        }
    }

    fn run<T>(&mut self, name: &'static str, stage: impl FnOnce() -> T) -> T {
        reset_peak_memory();
        let start = Instant::now();
        let out = stage();
        let secs = start.elapsed().as_secs_f64();
        let peak_mb = peak_delta_mb(self.base_live);
        self.stages.push(StageRecord {
            name,
            secs,
            peak_mb,
            fold: None,
        });
        out
    }

    /// Attach fold statistics to the stage that just ran.
    fn score_groups(&mut self, scoring: &Scoring, groups: &impl NodePartitioning) {
        let fold = scoring.fold_stats(groups.get_node_to_partition_map(), groups.num_partitions());
        self.stages.last_mut().unwrap().fold = Some(fold);
    }

    fn total_secs(&self) -> f64 {
        self.stages.iter().map(|s| s.secs).sum()
    }

    fn peak_mb(&self) -> f64 {
        self.stages.iter().map(|s| s.peak_mb).fold(0.0, f64::max)
    }
}

/// Modularity Leiden at production settings on `graph`, starting from `partition`.
///
/// `aggregate_first` contracts the starting partition before Leiden runs, as
/// `hybrid_community_detection` does after FLP.
fn modularity_leiden(
    graph: Graph<u8>,
    partition: LeidenNodePartitioning,
    aggregate_first: bool,
) -> Vec<usize> {
    let quality = Modularity::new(RESOLUTION, graph.get_total_edge_weight());
    let graph = Graph::<usize>::from(graph);
    let mut wp_graph =
        WeightedPartitionedGraph::new(graph, partition, quality, None, Some(LEIDEN_SEED));
    if aggregate_first {
        wp_graph.aggregate(AggregateOptions::All);
    }
    leiden(&mut wp_graph, RANDOMNESS, None, None);
    wp_graph.get_ancestor_to_partition_map()
}

/// Queue FLP as `hybrid_community_detection` runs it: one epoch, default strategy.
fn queue_flp(graph: &Graph<u8>) -> FastNodePartitioning {
    let strategy: &dyn AssignmentStrategy<FastNodePartitioning> = &DefaultAssignmentStrategy;
    fast_label_propagation::<FastNodePartitioning>(
        graph,
        1,
        strategy,
        FastNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes()),
    )
}

/// 4-cycle compression in its highest-precision mode ("3b").
fn four_cycle_groups(graph: &Graph<u8>) -> FastNodePartitioning {
    let is_b = graph
        .get_bipartite_side_mask()
        .expect("Graph loaded from parquet must have a bipartite side mask");
    four_cycle_compression::<FastNodePartitioning>(
        graph,
        is_b,
        FourCycleBridgeMode::StrictMajorityOnly,
        FOUR_CYCLE_MAX_ROUNDS,
    )
}

/// Runs `variant` and returns the community of every node, or `None` for a variant that stops
/// after its early stage.
fn run_variant(
    variant: &str,
    graph: Graph<u8>,
    scoring: &Scoring,
    recorder: &mut Recorder,
) -> Option<Vec<usize>> {
    match variant {
        // Current hybrid path: queue FLP, then modularity Leiden. "B1+" is the same pipeline, named
        // apart so that runs after the Leiden speedups are not mixed up with the `dev` baseline.
        "B1" | "B1+" => {
            let groups = recorder.run("flp", || queue_flp(&graph));
            recorder.score_groups(scoring, &groups);
            Some(recorder.run("leiden", || {
                modularity_leiden(graph, groups.into_node_partitioning(), true)
            }))
        }
        // Plain modularity Leiden from single nodes.
        "B2" => Some(recorder.run("leiden", || {
            let singletons =
                LeidenNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
            modularity_leiden(graph, singletons, false)
        })),
        // 4-cycle compression, then modularity Leiden.
        "B3" => {
            let groups = recorder.run("four_cycle", || four_cycle_groups(&graph));
            recorder.score_groups(scoring, &groups);
            Some(recorder.run("leiden", || {
                modularity_leiden(graph, groups.into_node_partitioning(), true)
            }))
        }
        // 4-cycle compression only, no final stage.
        "3b" => {
            let groups = recorder.run("four_cycle", || four_cycle_groups(&graph));
            recorder.score_groups(scoring, &groups);
            None
        }
        other => {
            eprintln!("Unknown variant '{other}'. Use B1, B1+, B2, B3 or 3b.");
            std::process::exit(1);
        }
    }
}

/// Peak resident set size of this process in MB, from `/proc/self/status` (Linux only).
fn peak_rss_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

fn pct(part: usize, whole: usize) -> f64 {
    100.0 * part as f64 / whole as f64
}

fn print_dataset_diagnostics(graph: &Graph<u8>) {
    let n = graph.get_num_nodes();
    let adj = graph.get_adjacency_matrix();
    let indptr_raw = adj.indptr();
    let indptr = indptr_raw.as_slice().unwrap();
    let degree_one = (0..n).filter(|&u| indptr[u + 1] - indptr[u] == 1).count();

    let components = graph.connected_components();
    let num_components = components.iter().max().map_or(0, |&m| m + 1);
    let mut sizes = vec![0usize; num_components];
    for &c in &components {
        sizes[c] += 1;
    }
    let largest = sizes.iter().copied().max().unwrap_or(0);

    println!(
        "DIAG degree1_nodes={} degree1_pct={:.3} components={} largest_component_pct={:.4}",
        degree_one,
        pct(degree_one, n),
        num_components,
        pct(largest, n)
    );
}

fn run(dataset_prefix: &str, variant: &str, shuffle_seed: Option<u64>) {
    let test_data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data");
    let path_0pc = test_data.join(format!("{dataset_prefix}_0pc.parquet"));
    let path_1pc = test_data.join(format!("{dataset_prefix}_1pc.parquet"));

    let (mapping_1pc, graph) =
        load_graph_from_parquet_with_edge_order(path_1pc.to_str().unwrap(), shuffle_seed);
    let scoring = Scoring::new(path_0pc.to_str().unwrap(), &mapping_1pc, &graph);
    drop(mapping_1pc);

    let seed_label = shuffle_seed.map_or("none".to_string(), |s| s.to_string());
    let gt_cells = scoring.cell_sizes.len();
    let gt_large_cells = scoring
        .cell_sizes
        .iter()
        .filter(|&&size| size as usize >= MIN_PNA_COMPONENT_SIZE)
        .count();
    println!(
        "DATASET name={} variant={} shuffle_seed={} nodes={} edges={} crossing_edges={} crossing_pct={:.2} gt_cells={} gt_cells_ge_8k={}",
        dataset_prefix,
        variant,
        seed_label,
        scoring.num_nodes(),
        scoring.edges.len(),
        scoring.num_crossing_edges,
        pct(scoring.num_crossing_edges, scoring.edges.len()),
        gt_cells,
        gt_large_cells
    );
    print_dataset_diagnostics(&graph);

    let mut recorder = Recorder::new();
    let communities = run_variant(variant, graph, &scoring, &mut recorder);

    let mut result = format!(
        "RESULT\tdataset={dataset_prefix}\tvariant={variant}\tseed={seed_label}\ttotal_secs={:.3}\tpeak_heap_mb={:.1}",
        recorder.total_secs(),
        recorder.peak_mb()
    );
    for stage in &recorder.stages {
        println!(
            "STAGE name={} secs={:.3} peak_heap_mb={:.1}",
            stage.name, stage.secs, stage.peak_mb
        );
        result += &format!(
            "\t{0}_secs={1:.3}\t{0}_peak_heap_mb={2:.1}",
            stage.name, stage.secs, stage.peak_mb
        );
        if let Some(fold) = &stage.fold {
            println!(
                "  groups={} crossing_folded={}/{} ({:.4}%) intra_folded={:.2}%",
                fold.num_groups,
                fold.crossing_folded,
                scoring.num_crossing_edges,
                pct(fold.crossing_folded, scoring.num_crossing_edges),
                pct(fold.intra_folded, scoring.num_intra_edges())
            );
            result += &format!(
                "\t{0}_groups={1}\t{0}_crossing_folded={2}\t{0}_crossing_folded_pct={3:.4}\t{0}_intra_folded_pct={4:.3}",
                stage.name,
                fold.num_groups,
                fold.crossing_folded,
                pct(fold.crossing_folded, scoring.num_crossing_edges),
                pct(fold.intra_folded, scoring.num_intra_edges())
            );
        }
    }
    println!(
        "TOTAL secs={:.3} peak_heap_mb={:.1}",
        recorder.total_secs(),
        recorder.peak_mb()
    );
    // Read before scoring, so that scoring allocations are not counted.
    if let Some(rss) = peak_rss_mb() {
        println!("PROCESS peak_rss_mb={rss:.1} (includes loading and ground truth)");
        result += &format!("\tprocess_peak_rss_mb={rss:.1}");
    }

    if let Some(communities) = communities {
        let stats = scoring.final_stats(&communities);
        println!(
            "FINAL over_split_cells={} under_split_components={} crossing_retained={}/{} ({:.3}%) intra_preserved={:.4}%",
            stats.over_split_cells,
            stats.under_split_components,
            stats.crossing_retained,
            scoring.num_crossing_edges,
            pct(stats.crossing_retained, scoring.num_crossing_edges),
            pct(stats.intra_preserved, scoring.num_intra_edges())
        );
        println!(
            "FINAL ari={:.6} nmi={:.6} components={} components_ge_8k={} (gt={}) nodes_in_small_components={}",
            stats.ari,
            stats.nmi,
            stats.num_components,
            stats.large_components,
            gt_large_cells,
            stats.nodes_in_small_components
        );
        result += &format!(
            "\tover_split_cells={}\tunder_split_components={}\tcrossing_retained={}\tcrossing_retained_pct={:.4}\tintra_preserved_pct={:.4}\tari={:.6}\tnmi={:.6}\tcomponents={}\tcomponents_ge_8k={}\tnodes_in_small_components={}",
            stats.over_split_cells,
            stats.under_split_components,
            stats.crossing_retained,
            pct(stats.crossing_retained, scoring.num_crossing_edges),
            pct(stats.intra_preserved, scoring.num_intra_edges()),
            stats.ari,
            stats.nmi,
            stats.num_components,
            stats.large_components,
            stats.nodes_in_small_components
        );
    }
    println!("{result}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut dataset = "40cells".to_string();
    let mut variant = "B1".to_string();
    let mut shuffle_seed: Option<u64> = Some(42);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dataset" if i + 1 < args.len() => {
                dataset = args[i + 1].clone();
                i += 2;
            }
            "--variant" if i + 1 < args.len() => {
                variant = args[i + 1].clone();
                i += 2;
            }
            "--shuffle-seed" if i + 1 < args.len() => {
                shuffle_seed = Some(args[i + 1].parse().expect("Invalid seed"));
                i += 2;
            }
            "--no-shuffle" => {
                shuffle_seed = None;
                i += 1;
            }
            other => {
                eprintln!("Unknown argument '{other}'.");
                std::process::exit(1);
            }
        }
    }

    let dataset_prefix = match dataset.as_str() {
        "40" | "40cells" | "mix_40cells" => "mix_40cells",
        "200" | "200cells" | "mix_200cells" => "mix_200cells",
        other => {
            eprintln!("Unknown dataset '{other}'. Use --dataset 40cells or --dataset 200cells.");
            std::process::exit(1);
        }
    };
    run(dataset_prefix, &variant, shuffle_seed);
}
