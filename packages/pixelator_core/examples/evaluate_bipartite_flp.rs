use std::alloc::{GlobalAlloc, Layout};
use std::cmp::Ordering;
use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

use itertools::Itertools;
use mimalloc::MiMalloc;
use rand::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use smallvec::SmallVec;

use pixelator_core::common::constants::MIN_PNA_COMPONENT_SIZE;
use pixelator_core::common::graph::Graph;
use pixelator_core::common::io::create_graph_and_umi_mapping_from_parquet_file;
use pixelator_core::common::node_partitioning::{
    FastNodePartitioning, LeidenNodePartitioning, NodePartitioning,
};
use pixelator_core::common::types::NodeIdx;
use pixelator_core::common::utils::{NodeQueue, mode_retain_smallest, mode_smallest};
use pixelator_core::fast_label_propagation::algorithm::{
    DEFAULT_BIPARTITE_FLP_MAX_ROUNDS, DEFAULT_BIPARTITE_FLP_MIN_CHANGED_FRACTION,
    alternating_bipartite_flp, clean_partitions,
};
use pixelator_core::leiden::algorithm::leiden;
use pixelator_core::leiden::quality::modularity::Modularity;
use pixelator_core::leiden::weighted_partitioned_graph::{
    AggregateOptions, WeightedPartitionedGraph,
};

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

/// Legacy random tie-breaking mode (matching `origin/dev` before deterministic tie-breaking).
fn mode_random(numbers: &mut [usize], rng: &mut impl Rng) -> usize {
    numbers.sort_unstable();
    let ((_, best_n), _) = numbers.iter().dedup_with_count().fold(
        ((0usize, 0usize), 0usize),
        |((best_count, best_n), ties), (c, &n)| match c.cmp(&best_count) {
            Ordering::Less => ((best_count, best_n), ties),
            Ordering::Greater => ((c, n), 1),
            Ordering::Equal => {
                let ties = ties + 1;
                if rng.random_bool(1.0 / ties as f64) {
                    ((c, n), ties)
                } else {
                    ((best_count, best_n), ties)
                }
            }
        },
    );
    best_n
}

/// PR #33 mode: retain `preferred` if tied for max count; otherwise break ties randomly.
fn mode_retain_random(numbers: &mut [usize], preferred: usize, rng: &mut impl Rng) -> usize {
    numbers.sort_unstable();
    let ((_, best_n), _) = numbers.iter().dedup_with_count().fold(
        ((0usize, 0usize), 0usize),
        |((best_count, best_n), ties), (c, &n)| match c.cmp(&best_count) {
            Ordering::Less => ((best_count, best_n), ties),
            Ordering::Greater => ((c, n), 1),
            Ordering::Equal => {
                if best_n == preferred {
                    ((best_count, best_n), ties)
                } else if n == preferred {
                    ((c, n), 1)
                } else {
                    let ties = ties + 1;
                    if rng.random_bool(1.0 / ties as f64) {
                        ((c, n), ties)
                    } else {
                        ((best_count, best_n), ties)
                    }
                }
            }
        },
    );
    best_n
}

#[derive(Clone, Copy)]
enum QueueTieVariant {
    BaselineRandom,
    RetainRandom,
    SmallestOnly,
    RetainSmallest,
}

fn run_queue_flp_variant(
    graph: &Graph<u8>,
    variant: QueueTieVariant,
    seed: u64,
) -> (FastNodePartitioning, usize, f64, f64, f64, f64) {
    let n = graph.get_num_nodes();
    let base_mem = reset_peak_memory();

    let mut node_partition = FastNodePartitioning::initialize_with_singlet_partitions(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut nodes: Vec<NodeIdx> = (0..n).collect();
    nodes.shuffle(&mut rng);

    let t0 = Instant::now();
    let mut node_queue = NodeQueue::from(nodes);
    let mut iterations = 0usize;
    let adj = graph.get_adjacency_matrix();

    while let Some(node_i) = node_queue.pop_front() {
        iterations += 1;
        let neighbors_view = adj.outer_view(node_i).unwrap();
        if neighbors_view.nnz() == 0 {
            continue;
        }
        let neighbors = neighbors_view.indices();
        let mut partitions = SmallVec::<[usize; 8]>::new();
        partitions.extend(
            neighbors
                .iter()
                .map(|x| node_partition.get_partition_for_node(x)),
        );

        let cur = node_partition.get_partition_for_node(&node_i);
        let chosen = match variant {
            QueueTieVariant::BaselineRandom => mode_random(partitions.as_mut_slice(), &mut rng),
            QueueTieVariant::RetainRandom => {
                mode_retain_random(partitions.as_mut_slice(), cur, &mut rng)
            }
            QueueTieVariant::SmallestOnly => mode_smallest(partitions.as_mut_slice()),
            QueueTieVariant::RetainSmallest => mode_retain_smallest(partitions.as_mut_slice(), cur),
        };

        if chosen != cur {
            node_partition.update_partition(node_i, chosen);
            for &nb in neighbors {
                if node_partition.get_partition_for_node(&nb) != chosen {
                    node_queue.push_back(nb);
                }
            }
        }
    }
    let flp_loop_sec = t0.elapsed().as_secs_f64();
    let flp_loop_mem_mb = peak_delta_mb(base_mem);
    drop(node_queue);

    let t_clean = Instant::now();
    let cleaned = clean_partitions(graph, node_partition);
    let clean_sec = t_clean.elapsed().as_secs_f64();
    let flp_total_mem_mb = peak_delta_mb(base_mem);
    (
        cleaned,
        iterations,
        flp_loop_sec,
        clean_sec,
        flp_loop_mem_mb,
        flp_total_mem_mb,
    )
}

/// Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) between two labelings.
fn compute_ari_and_nmi(a: &[usize], b: &[usize]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let nf = n as f64;

    let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::default();
    let mut a_counts: HashMap<usize, usize> = HashMap::default();
    let mut b_counts: HashMap<usize, usize> = HashMap::default();

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
        if nij > 0 {
            let p_ij = nij as f64 / nf;
            let p_i = a_counts[&la] as f64 / nf;
            let p_j = b_counts[&lb] as f64 / nf;
            mi += p_ij * (p_ij / (p_i * p_j)).ln();
        }
    }
    let h_a: f64 = a_counts
        .values()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / nf;
            -p * p.ln()
        })
        .sum();
    let h_b: f64 = b_counts
        .values()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / nf;
            -p * p.ln()
        })
        .sum();
    let nmi = if h_a + h_b < 1e-12 {
        1.0
    } else {
        2.0 * mi / (h_a + h_b)
    };

    (ari, nmi)
}

struct GroundTruthData {
    /// For each node index `u` in `graph_1pc`, its ground-truth cell component ID in `graph_0pc`.
    gt_cell_of_1pc_node: Vec<usize>,
    /// Undirected true crossing edges `(u, v)` (`u < v`) in `graph_1pc` where `gt_cell[u] != gt_cell[v]`.
    true_crossing_edges: Vec<(NodeIdx, NodeIdx)>,
    /// Undirected true intra-cell edges `(u, v)` (`u < v`) in `graph_1pc` where `gt_cell[u] == gt_cell[v]`.
    true_intra_edges: Vec<(NodeIdx, NodeIdx)>,
    /// Number of ground-truth cells (>= 8,000 nodes) in `0pc`.
    gt_cells_ge_8k: usize,
    /// Total connected components in `0pc`.
    gt_total_components: usize,
}

fn build_ground_truth(
    path_0pc: &str,
    mapping_1pc: &pixelator_core::common::node_indexing::UmiToNodeIndexMapping,
    graph_1pc: &Graph<u8>,
) -> GroundTruthData {
    let (mapping_0pc, graph_0pc) = create_graph_and_umi_mapping_from_parquet_file::<u8>(path_0pc);
    let comp_0pc = graph_0pc.connected_components();

    let mut comp_sizes: HashMap<usize, usize> = HashMap::default();
    for &c in &comp_0pc {
        *comp_sizes.entry(c).or_insert(0) += 1;
    }
    let gt_total_components = comp_sizes.len();
    let gt_cells_ge_8k = comp_sizes
        .values()
        .filter(|&&sz| sz >= MIN_PNA_COMPONENT_SIZE)
        .count();

    let n_1pc = graph_1pc.get_num_nodes();
    let mut gt_cell_of_1pc_node = Vec::with_capacity(n_1pc);
    for u_1pc in 0..n_1pc {
        let umi = mapping_1pc.map_node_index_to_umi(u_1pc);
        let u_0pc = mapping_0pc.map_umi_to_node_index(umi);
        gt_cell_of_1pc_node.push(comp_0pc[u_0pc]);
    }

    let adj = graph_1pc.get_adjacency_matrix();
    let indptr_raw = adj.indptr();
    let indptr = indptr_raw.as_slice().unwrap();
    let indices = adj.indices();

    let mut true_crossing_edges = Vec::new();
    let mut true_intra_edges = Vec::new();
    for u in 0..n_1pc {
        let cu = gt_cell_of_1pc_node[u];
        for &v in &indices[indptr[u]..indptr[u + 1]] {
            if u < v {
                if cu != gt_cell_of_1pc_node[v] {
                    true_crossing_edges.push((u, v));
                } else {
                    true_intra_edges.push((u, v));
                }
            }
        }
    }

    GroundTruthData {
        gt_cell_of_1pc_node,
        true_crossing_edges,
        true_intra_edges,
        gt_cells_ge_8k,
        gt_total_components,
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_variant(
    name: &str,
    graph_1pc: &Graph<u8>,
    gt: &GroundTruthData,
    flp_partition: FastNodePartitioning,
    flp_sec: f64,
    clean_sec: f64,
    flp_loop_mem_mb: f64,
    flp_total_mem_mb: f64,
    baseline_final_components: Option<&[usize]>,
) -> Vec<usize> {
    let flp_map = flp_partition.get_node_to_partition_map();
    let flp_num_partitions = flp_partition.num_partitions();

    // Criterion 2: How many true crossing edges are folded inside FLP partitions?
    let folded_crossing_edges = gt
        .true_crossing_edges
        .iter()
        .filter(|&&(u, v)| flp_map[u] == flp_map[v])
        .count();
    let total_true_crossing = gt.true_crossing_edges.len();
    let folded_crossing_pct = 100.0 * folded_crossing_edges as f64 / total_true_crossing as f64;

    let flp_coarsened_intra = gt
        .true_intra_edges
        .iter()
        .filter(|&&(u, v)| flp_map[u] == flp_map[v])
        .count();
    let flp_coarsened_intra_pct =
        100.0 * flp_coarsened_intra as f64 / gt.true_intra_edges.len() as f64;

    // Run Leiden stage
    let base_leiden_mem = reset_peak_memory();
    let t_leiden = Instant::now();
    let quality_function = Modularity::new(1.0, graph_1pc.get_total_edge_weight());
    let graph_usize = Graph::<usize>::from(graph_1pc.clone());
    let partition: LeidenNodePartitioning = NodePartitioning::into_node_partitioning(flp_partition);

    let mut wp_graph =
        WeightedPartitionedGraph::new(graph_usize, partition, quality_function, None, Some(0));
    wp_graph.aggregate(AggregateOptions::All);
    leiden(&mut wp_graph, 0.1, None, None);
    let post_leiden_stats = wp_graph.get_statistics();
    let leiden_sec = t_leiden.elapsed().as_secs_f64();
    let leiden_mem_mb = peak_delta_mb(base_leiden_mem);

    let final_partition_map = wp_graph.get_ancestor_to_partition_map();

    // Criterion 3: Connected components formed after removing crossing edges
    let final_components =
        graph_1pc.connected_components_by(|u, v| final_partition_map[u] == final_partition_map[v]);

    let mut comp_sizes: HashMap<usize, usize> = HashMap::default();
    for &c in &final_components {
        *comp_sizes.entry(c).or_insert(0) += 1;
    }
    let total_final_components = comp_sizes.len();
    let cells_ge_8k = comp_sizes
        .values()
        .filter(|&&sz| sz >= MIN_PNA_COMPONENT_SIZE)
        .count();
    let stranded_nodes: usize = comp_sizes
        .values()
        .filter(|&&sz| sz < MIN_PNA_COMPONENT_SIZE)
        .sum();

    // True crossing edges remaining vs removed after filtering
    let leaked_crossing_edges = gt
        .true_crossing_edges
        .iter()
        .filter(|&&(u, v)| final_components[u] == final_components[v])
        .count();
    let removed_crossing_edges = total_true_crossing - leaked_crossing_edges;
    let removed_crossing_pct = 100.0 * removed_crossing_edges as f64 / total_true_crossing as f64;

    // True intra-cell edges preserved vs cut after filtering
    let preserved_intra_edges = gt
        .true_intra_edges
        .iter()
        .filter(|&&(u, v)| final_components[u] == final_components[v])
        .count();
    let preserved_intra_pct =
        100.0 * preserved_intra_edges as f64 / gt.true_intra_edges.len() as f64;

    let (ari_vs_gt, nmi_vs_gt) = compute_ari_and_nmi(&gt.gt_cell_of_1pc_node, &final_components);
    let (ari_vs_base, nmi_vs_base) = baseline_final_components
        .map(|base| compute_ari_and_nmi(base, &final_components))
        .unwrap_or((1.0, 1.0));

    println!("--------------------------------------------------------------------------------");
    println!("Variant: {}", name);
    println!(
        "  [1. Performance]   FLP={:.2}s | Clean={:.2}s | Leiden={:.2}s | Total={:.2}s | FLP_Loop_Heap={:.1} MB | FLP+Clean_Heap={:.1} MB | Leiden_Heap={:.1} MB",
        flp_sec,
        clean_sec,
        leiden_sec,
        flp_sec + clean_sec + leiden_sec,
        flp_loop_mem_mb,
        flp_total_mem_mb,
        leiden_mem_mb
    );
    println!(
        "  [2. FLP Folding]   FLP_Partitions={} | True_Crossing_Folded={} / {} ({:.3}%) | Intra_Cell_Coarsened_in_FLP={:.2}%",
        flp_num_partitions,
        folded_crossing_edges,
        total_true_crossing,
        folded_crossing_pct,
        flp_coarsened_intra_pct
    );
    println!(
        "  [3. Components]    ARI_vs_GT={:.6} | NMI_vs_GT={:.6} | ARI_vs_Base={:.6} | NMI_vs_Base={:.6}",
        ari_vs_gt, nmi_vs_gt, ari_vs_base, nmi_vs_base
    );
    println!(
        "                     True_Crossing_Removed={}/{} ({:.3}%) [Leaked={} ({} from FLP, {} from Leiden)] | True_Intra_Preserved={:.4}%",
        removed_crossing_edges,
        total_true_crossing,
        removed_crossing_pct,
        leaked_crossing_edges,
        folded_crossing_edges,
        leaked_crossing_edges.saturating_sub(folded_crossing_edges),
        preserved_intra_pct
    );
    println!(
        "                     Final_Components={} | Cells(>=8k)={} (GT={}) | Stranded_Nodes={} | Modularity_Q={:.2}",
        total_final_components,
        cells_ge_8k,
        gt.gt_cells_ge_8k,
        stranded_nodes,
        post_leiden_stats.quality
    );

    final_components
}

fn run_dataset_evaluation(dataset_prefix: &str) {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path_0pc = manifest_dir
        .join("test_data")
        .join(format!("{}_0pc.parquet", dataset_prefix));
    let path_1pc = manifest_dir
        .join("test_data")
        .join(format!("{}_1pc.parquet", dataset_prefix));

    println!("\n================================================================================");
    println!("EVALUATING DATASET PAIR: {} (0pc vs 1pc)", dataset_prefix);
    println!("================================================================================");

    let base_load_mem = reset_peak_memory();
    let (mapping_1pc, graph_1pc) =
        create_graph_and_umi_mapping_from_parquet_file::<u8>(path_1pc.to_str().unwrap());
    let graph_load_live_mb = LIVE_BYTES
        .load(AtomicOrdering::Relaxed)
        .saturating_sub(base_load_mem) as f64
        / (1024.0 * 1024.0);

    let gt = build_ground_truth(path_0pc.to_str().unwrap(), &mapping_1pc, &graph_1pc);
    drop(mapping_1pc);

    println!(
        "Graph (1pc): N={} | E_total={} | E_intra_gt={} | E_crossing_gt={} ({:.2}%)",
        graph_1pc.get_num_nodes(),
        graph_1pc.get_edge_entry_count(),
        gt.true_intra_edges.len(),
        gt.true_crossing_edges.len(),
        100.0 * gt.true_crossing_edges.len() as f64 / graph_1pc.get_edge_entry_count() as f64
    );
    println!(
        "Ground Truth (0pc): Total_Components={} | Cells(>=8k)={} | Graph+Mask_Live_Heap={:.1} MB",
        gt.gt_total_components, gt.gt_cells_ge_8k, graph_load_live_mb
    );

    // 0. Baseline (random tie-breaking)
    let (p_base, _, l_base, c_base, ml_base, mt_base) =
        run_queue_flp_variant(&graph_1pc, QueueTieVariant::BaselineRandom, 42);
    let base_components = evaluate_variant(
        "0. Queue Baseline (random tie)",
        &graph_1pc,
        &gt,
        p_base,
        l_base,
        c_base,
        ml_base,
        mt_base,
        None,
    );

    // PR #33 (retain + random tie-breaking)
    let (p_pr33, _, l_pr33, c_pr33, ml_pr33, mt_pr33) =
        run_queue_flp_variant(&graph_1pc, QueueTieVariant::RetainRandom, 42);
    let _ = evaluate_variant(
        "PR#33 Queue (retain + random tie)",
        &graph_1pc,
        &gt,
        p_pr33,
        l_pr33,
        c_pr33,
        ml_pr33,
        mt_pr33,
        Some(&base_components),
    );

    // 1a. Queue: Smallest ID on tie
    let (p_1a, _, l_1a, c_1a, ml_1a, mt_1a) =
        run_queue_flp_variant(&graph_1pc, QueueTieVariant::SmallestOnly, 42);
    let _ = evaluate_variant(
        "1a. Queue (Smallest ID on tie)",
        &graph_1pc,
        &gt,
        p_1a,
        l_1a,
        c_1a,
        ml_1a,
        mt_1a,
        Some(&base_components),
    );

    // 1b. Queue: Retain + Smallest ID on tie
    let (p_1b, _, l_1b, c_1b, ml_1b, mt_1b) =
        run_queue_flp_variant(&graph_1pc, QueueTieVariant::RetainSmallest, 42);
    let _ = evaluate_variant(
        "1b. Queue (Retain + Smallest ID)",
        &graph_1pc,
        &gt,
        p_1b,
        l_1b,
        c_1b,
        ml_1b,
        mt_1b,
        Some(&base_components),
    );

    // 2. Convergent Alternating Bipartite FLP (<1% changed, 4T)
    let is_b = graph_1pc
        .get_bipartite_side_mask()
        .expect("Graph loaded from parquet must have bipartite side mask");
    let bip_loop_only_mb =
        (graph_1pc.get_num_nodes() * std::mem::size_of::<usize>()) as f64 / (1024.0 * 1024.0);

    let base_bip_mem = reset_peak_memory();
    let t_bip = Instant::now();
    let p_bip: FastNodePartitioning = alternating_bipartite_flp(
        &graph_1pc,
        is_b,
        DEFAULT_BIPARTITE_FLP_MIN_CHANGED_FRACTION,
        DEFAULT_BIPARTITE_FLP_MAX_ROUNDS,
    );
    let bip_total_flp_sec = t_bip.elapsed().as_secs_f64();
    let bip_peak_mem_mb = peak_delta_mb(base_bip_mem);

    let _ = evaluate_variant(
        "2. Alternating Bipartite FLP (<1% changed, parallel)",
        &graph_1pc,
        &gt,
        p_bip,
        bip_total_flp_sec,
        0.0, // included in alternating_bipartite_flp
        bip_loop_only_mb,
        bip_peak_mem_mb,
        Some(&base_components),
    );

    // Also evaluate 1-pass (max_rounds = 1) and <10% (min_changed_fraction = 0.10) for comparison
    for (label, frac, max_r) in [
        (
            "2-alt1. Alternating Bipartite FLP (1-pass, max_rounds=1)",
            1.0,
            1,
        ),
        ("2-alt2. Alternating Bipartite FLP (<10% changed)", 0.10, 10),
    ] {
        let base_m = reset_peak_memory();
        let t0 = Instant::now();
        let p_alt: FastNodePartitioning = alternating_bipartite_flp(&graph_1pc, is_b, frac, max_r);
        let alt_sec = t0.elapsed().as_secs_f64();
        let alt_mem = peak_delta_mb(base_m);
        let _ = evaluate_variant(
            label,
            &graph_1pc,
            &gt,
            p_alt,
            alt_sec,
            0.0,
            bip_loop_only_mb,
            alt_mem,
            Some(&base_components),
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut target = "all".to_string();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--dataset" && i + 1 < args.len() {
            target = args[i + 1].clone();
            i += 2;
        } else {
            target = args[i].clone();
            i += 1;
        }
    }

    match target.as_str() {
        "40" | "40cells" | "mix_40cells" => {
            run_dataset_evaluation("mix_40cells");
        }
        "200" | "200cells" | "mix_200cells" => {
            run_dataset_evaluation("mix_200cells");
        }
        "all" => {
            run_dataset_evaluation("mix_40cells");
            run_dataset_evaluation("mix_200cells");
        }
        other => {
            eprintln!(
                "Unknown dataset '{}'. Use --dataset 40cells, --dataset 200cells, or --dataset all.",
                other
            );
            std::process::exit(1);
        }
    }
}
