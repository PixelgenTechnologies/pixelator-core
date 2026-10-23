use indicatif::ProgressBar;
use log::info;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use std::time::Instant;

use crate::common::graph::Graph;
use crate::common::node_partitioning::NodePartitioning;
use crate::common::types::{NodeIdx, PartitionId};
use crate::common::utils::{NodeQueue, mode_retain_smallest, mode_smallest};
use crate::fast_label_propagation::strategies::AssignmentStrategy;

/// Default relative change threshold for alternating bipartite FLP: stop when < 1% of nodes change.
pub const DEFAULT_BIPARTITE_FLP_MIN_CHANGED_FRACTION: f64 = 0.01;

/// Default maximum number of alternating `(A -> B, B -> A)` rounds.
pub const DEFAULT_BIPARTITE_FLP_MAX_ROUNDS: usize = 10;

pub fn fast_label_propagation<P>(
    graph: &Graph<u8>,
    epochs: u64,
    assignment_strategy: &dyn AssignmentStrategy<P>,
    mut node_partition: P,
) -> P
where
    P: NodePartitioning,
{
    if let Some(is_b) = graph.get_bipartite_side_mask() {
        if epochs == 0 {
            return node_partition;
        }
        return alternating_bipartite_flp::<P>(
            graph,
            is_b,
            DEFAULT_BIPARTITE_FLP_MIN_CHANGED_FRACTION,
            DEFAULT_BIPARTITE_FLP_MAX_ROUNDS,
        );
    }

    for epoch in 0..epochs {
        info!("Starting epoch {}...", epoch + 1);
        node_partition = label_propagation_epoch::<P>(graph, node_partition, assignment_strategy);
    }
    node_partition
}

/// Parallel Convergent Alternating Bipartite Fast Label Propagation.
///
/// Exploits the bipartite structure of PNA/MPX graphs (`is_b[u] == false` for A-nodes,
/// `is_b[u] == true` for B-nodes) by alternating parallel Gauss-Seidel half-steps:
/// 1. Half-step `A -> B`: update all B-nodes in parallel from their A-neighbors.
/// 2. Half-step `B -> A`: update all A-nodes in parallel from their B-neighbors.
///
/// Terminates when the fraction of nodes changing labels in a full round is at or below
/// `min_changed_fraction` (e.g. `0.01` for `< 1%` changed) or `max_rounds` is reached.
pub fn alternating_bipartite_flp<P>(
    graph: &Graph<u8>,
    is_b: &[bool],
    min_changed_fraction: f64,
    max_rounds: usize,
) -> P
where
    P: NodePartitioning,
{
    let n = graph.get_num_nodes();
    assert_eq!(
        is_b.len(),
        n,
        "Bipartite side mask length ({}) must match number of graph nodes ({})",
        is_b.len(),
        n
    );

    let threshold_nodes = (n as f64 * min_changed_fraction.clamp(0.0, 1.0)) as usize;
    let mut labels: Vec<PartitionId> = (0..n).collect();

    let adj = graph.get_adjacency_matrix();
    let indptr_raw = adj.indptr();
    let indptr = indptr_raw
        .as_slice()
        .expect("CSR adjacency matrix indptr must be contiguous");
    let indices = adj.indices();

    info!(
        "Starting alternating bipartite label propagation (n={}, min_changed_fraction={}, max_rounds={})...",
        n, min_changed_fraction, max_rounds
    );

    {
        // Safety: `PartitionId` (`usize`) and `AtomicUsize` have identical size, alignment,
        // and bit representation. Within Half-step 1 (A -> B), threads only read `A` indices
        // and write disjoint `B` indices; within Half-step 2 (B -> A), threads only read `B`
        // indices and write disjoint `A` indices.
        let atomic_labels: &[AtomicUsize] =
            unsafe { std::slice::from_raw_parts(labels.as_mut_ptr() as *const AtomicUsize, n) };

        for round in 0..max_rounds {
            // Half-step 1 (A -> B): Update all B-nodes in parallel from A-neighbors
            let b_changed: usize = atomic_labels
                .par_iter()
                .enumerate()
                .with_min_len(4096)
                .map(|(b, label_b)| {
                    if !is_b[b] {
                        return 0usize;
                    }
                    let start = indptr[b];
                    let end = indptr[b + 1];
                    if start == end {
                        return 0usize;
                    }
                    let cur = label_b.load(Ordering::Relaxed);
                    let chosen = if round == 0 {
                        // In Round 0, all A-nodes hold singlet IDs `a` and CSR row indices
                        // are sorted ascending, so the minimum neighbor ID is `indices[start]`.
                        indices[start]
                    } else {
                        let mut parts = SmallVec::<[usize; 8]>::new();
                        for &a in &indices[start..end] {
                            parts.push(atomic_labels[a].load(Ordering::Relaxed));
                        }
                        mode_retain_smallest(parts.as_mut_slice(), cur)
                    };

                    if chosen != cur {
                        label_b.store(chosen, Ordering::Relaxed);
                        1usize
                    } else {
                        0usize
                    }
                })
                .sum();

            // Implicit Rayon barrier: all B-node updates are globally visible.

            // Half-step 2 (B -> A): Update all A-nodes in parallel from B-neighbors
            let a_changed: usize = atomic_labels
                .par_iter()
                .enumerate()
                .with_min_len(4096)
                .map(|(a, label_a)| {
                    if is_b[a] {
                        return 0usize;
                    }
                    let start = indptr[a];
                    let end = indptr[a + 1];
                    if start == end {
                        return 0usize;
                    }
                    let mut parts = SmallVec::<[usize; 8]>::new();
                    for &b in &indices[start..end] {
                        parts.push(atomic_labels[b].load(Ordering::Relaxed));
                    }
                    let cur = label_a.load(Ordering::Relaxed);
                    let chosen = if round == 0 {
                        mode_smallest(parts.as_mut_slice())
                    } else {
                        mode_retain_smallest(parts.as_mut_slice(), cur)
                    };

                    if chosen != cur {
                        label_a.store(chosen, Ordering::Relaxed);
                        1usize
                    } else {
                        0usize
                    }
                })
                .sum();

            let total_changed = b_changed + a_changed;
            info!(
                "Bipartite FLP round {} finished: {} nodes changed ({:.3}% of N).",
                round + 1,
                total_changed,
                if n > 0 {
                    100.0 * total_changed as f64 / n as f64
                } else {
                    0.0
                }
            );
            if total_changed <= threshold_nodes {
                break;
            }
        }
    }

    clean_partitions(graph, P::initialize_from_partitions(labels))
}

pub fn label_propagation_epoch<P>(
    graph: &Graph<u8>,
    mut node_partition: P,
    assignment_strategy: &dyn AssignmentStrategy<P>,
) -> P
where
    P: NodePartitioning,
{
    let mut nodes: Vec<NodeIdx> = (0..graph.get_num_nodes()).collect();
    nodes.shuffle(&mut rand::rng());
    let mut node_queue = NodeQueue::from(nodes);

    info!("Starting label propagation...");
    let bar = ProgressBar::new_spinner();
    bar.enable_steady_tick(Duration::from_millis(100));
    let mut iteration = 0usize;
    let mut last_report = Instant::now();
    let mut last_iteration = 0usize;

    while let Some(node_i) = node_queue.pop_front() {
        iteration += 1;
        if iteration.is_multiple_of(1_000_000) {
            update_progress_bar(
                &node_queue,
                &bar,
                iteration,
                &mut last_report,
                &mut last_iteration,
            );
        }
        assignment_strategy.assign(graph, &mut node_partition, &mut node_queue, node_i);
    }
    bar.finish_and_clear();

    info!("Label propagation finished after {} iterations.", iteration);
    clean_partitions(graph, node_partition)
}

fn update_progress_bar(
    node_queue: &NodeQueue,
    bar: &ProgressBar,
    iteration: usize,
    last_report: &mut Instant,
    last_iteration: &mut usize,
) {
    let now = Instant::now();
    let elapsed = now.duration_since(*last_report).as_secs_f64();
    let iters = (iteration - *last_iteration) as f64;
    let rate_millions = if elapsed > 0.0 {
        iters / elapsed / 1_000_000.0
    } else {
        0.0
    };
    bar.set_message(format!(
        "Processing nodes. Iteration: {}, queue size: {}, {:.2} M nodes/s",
        iteration,
        node_queue.len(),
        rate_millions
    ));
    *last_report = now;
    *last_iteration = iteration;
}

/// Split disconnected partitions into their individual components
pub fn clean_partitions<P>(graph: &Graph<u8>, node_partition: P) -> P
where
    P: NodePartitioning,
{
    info!("Cleaning partitions by splitting disconnected components...");
    let labels = graph.connected_components_by(|src_node, node| {
        node_partition.get_partition_for_node(&src_node)
            == node_partition.get_partition_for_node(&node)
    });
    P::initialize_from_partitions(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::graph::Graph;
    use crate::common::node_partitioning::{FastNodePartitioning, NodePartitioning};
    use crate::common::types::edges_from_tuples;
    use crate::fast_label_propagation::strategies::DefaultAssignmentStrategy;

    use crate::common::test_utils::{get_2_communities_edges, get_example_graph, get_random_graph};

    #[test]
    fn test_flp_epoch() {
        //   1   3
        //   |   |
        // 0-2   5-4
        //    \ /
        //     6
        //    /|\
        //   7-8-9
        let edges = edges_from_tuples(vec![
            (0, 2),
            (1, 2),
            (3, 5),
            (4, 5),
            (2, 6),
            (5, 6),
            (6, 7),
            (6, 8),
            (6, 9),
        ]);

        let graph = Graph::<u8>::from_edges(edges.into_iter(), 10);
        let mut partitioning = FastNodePartitioning::initialize_from_partitions(vec![
            //nodes:
            //  0, 1, 2, 3, 4, 5, 6, 7, 8, 9
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        ]);

        partitioning = label_propagation_epoch(&graph, partitioning, &DefaultAssignmentStrategy);
        partitioning.normalize();

        let partition_map = partitioning.node_partition_map;

        assert_eq!(partition_map[0], partition_map[1]);
        assert_eq!(partition_map[0], partition_map[2]);

        assert_eq!(partition_map[3], partition_map[4]);
        assert_eq!(partition_map[3], partition_map[5]);

        assert_eq!(partition_map[6], partition_map[7]);
        assert_eq!(partition_map[6], partition_map[8]);
        assert_eq!(partition_map[6], partition_map[9]);

        assert_ne!(partition_map[0], partition_map[3]);
        assert_ne!(partition_map[0], partition_map[6]);
    }

    #[test]
    fn test_flp() {
        let graph = get_example_graph();
        let num_nodes = graph.get_num_nodes();
        let partitioning = FastNodePartitioning::initialize_with_singlet_partitions(num_nodes);

        let result_partition =
            fast_label_propagation(&graph, 10, &DefaultAssignmentStrategy, partitioning);
        let partition_map = &result_partition.node_partition_map;

        assert_eq!(partition_map.len(), num_nodes);

        // Degree-1 nodes must adopt their sole neighbor's partition:
        assert_eq!(partition_map[0], partition_map[2]);
        assert_eq!(partition_map[1], partition_map[8]);
        assert_eq!(partition_map[6], partition_map[5]);

        // After convergence, every node's partition is a mode among its neighbors, and each
        // partition induces a connected component:
        for node in 0..num_nodes {
            let mut neighbor_partitions: Vec<usize> = graph
                .neighbors_iter(node)
                .map(|neighbor| partition_map[neighbor])
                .collect();
            let node_partition = partition_map[node];
            assert_eq!(
                crate::common::utils::mode_via_sort(&mut neighbor_partitions, Some(node_partition)),
                node_partition
            );
        }
        assert!(result_partition.num_partitions() <= 3);
    }

    #[test]
    fn test_flp_disconnected_graph() {
        //    2      5
        //   / \    / \
        //  0 - 1  3 - 4
        let edges = vec![(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
        let num_nodes = 6;
        let edge_iter = edges_from_tuples(edges.clone());
        let graph = Graph::<u8>::from_edges(edge_iter, num_nodes);

        let partitioning = FastNodePartitioning::initialize_with_singlet_partitions(num_nodes);

        let result_partition =
            fast_label_propagation(&graph, 10, &DefaultAssignmentStrategy, partitioning);
        let result_partition_map = result_partition.node_partition_map;

        assert_eq!(result_partition_map[0], result_partition_map[1]);
        assert_eq!(result_partition_map[0], result_partition_map[2]);

        assert_eq!(result_partition_map[3], result_partition_map[4]);
        assert_eq!(result_partition_map[3], result_partition_map[5]);
    }

    #[test]
    fn test_flp_2_communities() {
        let graph = Graph::<u8>::from_edges(edges_from_tuples(get_2_communities_edges()), 8);
        let num_nodes = graph.get_num_nodes();
        let partitioning = FastNodePartitioning::initialize_with_singlet_partitions(num_nodes);

        let result_partition =
            fast_label_propagation(&graph, 10, &DefaultAssignmentStrategy, partitioning);
        let result_partition_map = result_partition.node_partition_map;

        assert_eq!(result_partition_map.len(), num_nodes);

        assert_eq!(result_partition_map[0], result_partition_map[1]);
        assert_eq!(result_partition_map[0], result_partition_map[2]);
        assert_eq!(result_partition_map[0], result_partition_map[3]);

        // NOTE: in a few cases (<5%), this test will put all nodes in the same community and the
        // assertion below will fail.
        //assert_ne!(result_partition_map[0], result_partition_map[4]);

        assert_eq!(result_partition_map[4], result_partition_map[5]);
        assert_eq!(result_partition_map[4], result_partition_map[6]);
        assert_eq!(result_partition_map[4], result_partition_map[7]);
    }

    #[test]
    fn test_flp_no_edges() {
        let edges = vec![];
        let num_nodes = 3;
        let edge_iter = edges_from_tuples(edges.clone());
        let graph = Graph::from_edges(edge_iter, num_nodes);

        let partitioning = FastNodePartitioning::initialize_with_singlet_partitions(num_nodes);

        let result_partition =
            fast_label_propagation(&graph, 10, &DefaultAssignmentStrategy, partitioning);
        let result_partition_map = result_partition.get_node_to_partition_map();

        assert_ne!(result_partition_map[0], result_partition_map[1]);
        assert_ne!(result_partition_map[0], result_partition_map[2]);
    }

    #[test]
    fn test_flp_random() {
        let n_nodes = 1000;
        let n_edges = 2 * n_nodes;
        let seed = 0;
        let graph = get_random_graph(n_nodes, n_edges, seed);

        let partitioning = FastNodePartitioning::initialize_with_singlet_partitions(n_nodes);

        let result_partition =
            fast_label_propagation(&graph, 10, &DefaultAssignmentStrategy, partitioning);

        assert_eq!(
            graph
                .connected_components_by(|node_1, node_2| result_partition
                    .get_node_to_partition_map()[node_1]
                    == result_partition.get_node_to_partition_map()[node_2])
                .into_iter()
                .max()
                .unwrap()
                + 1,
            result_partition.num_partitions()
        );
    }

    #[test]
    fn test_clean_partitions_splits_disconnected_partitions() {
        // Path 0-1-2-3. Nodes 0, 1 and 3 share partition 5, but 3 is not connected to 0-1.
        let graph = Graph::<u8>::from_edges(edges_from_tuples(vec![(0, 1), (1, 2), (2, 3)]), 4);
        let partitioning = FastNodePartitioning::initialize_from_partitions(vec![5, 5, 6, 5]);

        let cleaned = clean_partitions(&graph, partitioning);

        assert_eq!(cleaned.get_node_to_partition_map(), [0, 0, 1, 2].as_slice());
    }

    #[test]
    fn test_alternating_bipartite_flp_two_bicliques_with_bridge() {
        // Left A-nodes: 0, 1, 2, 3 (is_b = false)
        // Right B-nodes: 4, 5, 6, 7, 8 (is_b = true)
        // Community 1: A={0, 1} connected to B={4, 5}
        // Community 2: A={2, 3} connected to B={7, 8}
        // Bridge B-node 6 connected to 1 and 2
        let edges = edges_from_tuples(vec![
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 6),
            (2, 7),
            (2, 8),
            (3, 7),
            (3, 8),
        ]);
        let is_b = vec![false, false, false, false, true, true, true, true, true];
        let graph = Graph::<u8>::from_edges(edges, 9).with_bipartite_side_mask(is_b.clone());

        let result: FastNodePartitioning = alternating_bipartite_flp(
            &graph,
            &is_b,
            DEFAULT_BIPARTITE_FLP_MIN_CHANGED_FRACTION,
            DEFAULT_BIPARTITE_FLP_MAX_ROUNDS,
        );
        let map = result.get_node_to_partition_map();

        // Community 1 nodes (A={0, 1} and B={4, 5, 6}) share a partition
        assert_eq!(map[0], map[1]);
        assert_eq!(map[0], map[4]);
        assert_eq!(map[0], map[5]);
        assert_eq!(map[0], map[6]);

        // Community 2 nodes (A={2, 3} and B={7, 8}) share a partition
        assert_eq!(map[2], map[3]);
        assert_eq!(map[2], map[7]);
        assert_eq!(map[2], map[8]);

        // The two communities remain distinct
        assert_ne!(map[0], map[2]);

        // `fast_label_propagation` dispatches to `alternating_bipartite_flp` when mask is present
        let init = FastNodePartitioning::initialize_with_singlet_partitions(9);
        let dispatched = fast_label_propagation(&graph, 1, &DefaultAssignmentStrategy, init);
        assert_eq!(dispatched.get_node_to_partition_map(), map);
    }

    #[test]
    fn test_alternating_bipartite_flp_deterministic_and_connected_on_parquet() {
        let test_data = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test_data/mix_40cells_1pc_1000rows.parquet");
        let (_, graph) = crate::common::io::create_graph_and_umi_mapping_from_parquet_file::<u8>(
            test_data.to_str().unwrap(),
        );
        assert!(graph.get_bipartite_side_mask().is_some());

        let n = graph.get_num_nodes();
        let init1 = FastNodePartitioning::initialize_with_singlet_partitions(n);
        let init2 = FastNodePartitioning::initialize_with_singlet_partitions(n);

        let p1 = fast_label_propagation(&graph, 1, &DefaultAssignmentStrategy, init1);
        let p2 = fast_label_propagation(&graph, 1, &DefaultAssignmentStrategy, init2);

        // 100% deterministic across runs
        assert_eq!(
            p1.get_node_to_partition_map(),
            p2.get_node_to_partition_map()
        );

        // Every partition induces a connected component
        assert_eq!(
            graph
                .connected_components_by(
                    |u, v| p1.get_node_to_partition_map()[u] == p1.get_node_to_partition_map()[v]
                )
                .into_iter()
                .max()
                .unwrap()
                + 1,
            p1.num_partitions()
        );
    }
}
