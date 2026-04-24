use log::debug;

use crate::common::node_partitioning::NodePartitioning;
use crate::common::types::NodeIdx;
use crate::common::utils::NodeQueue;
use crate::leiden::quality::QualityMetrics;
use crate::leiden::weighted_partitioned_graph::{
    AggregateOptions, PartitionedGraphStatistics, WeightedPartitionedGraph,
};
use rustc_hash::FxHashSet as HashSet;

use std::cmp::min;

pub fn leiden<Q: QualityMetrics>(
    wp_graph: &mut WeightedPartitionedGraph<Q>,
    randomness: f64,
    max_iteration: Option<usize>,
    threshold_options: Option<ThresholdOptions>,
) -> PartitionedGraphStatistics {
    let mut it = 0;
    debug!("Starting Leiden algorithm");
    loop {
        if let Some(max_iteration) = max_iteration
            && it >= max_iteration
        {
            wp_graph.aggregate(AggregateOptions::All);
            wp_graph.normalize_partition();
            break;
        }

        debug!("Moving nodes fast (iteration {})", it);
        let delta = move_nodes_fast(wp_graph);
        debug!(
            "it={}, delta={}, #partitions={}, #nodes={}",
            &it,
            &delta,
            wp_graph.get_partitioning().num_partitions(),
            wp_graph.get_graph().get_num_nodes()
        );

        if wp_graph.get_partitioning().num_partitions() == wp_graph.get_graph().get_num_nodes() {
            break;
        }

        debug!("Aggregating graph (iteration {})", it);
        wp_graph.aggregate(AggregateOptions::OnlyWellConnected(randomness));
        debug!("Normalizing partition (iteration {})", it);
        wp_graph.normalize_partition();

        it += 1;
    }

    if let Some(threshold_options) = threshold_options {
        debug!("Merging highly connected nodes");
        merge_highly_connected_nodes(wp_graph, threshold_options);
    }

    debug!("Getting statistics");
    let statistics = wp_graph.get_statistics();
    debug!("Leiden algorithm finished");
    statistics
}

fn move_nodes_fast<Q: QualityMetrics>(wp_graph: &mut WeightedPartitionedGraph<Q>) -> f64 {
    let mut node_queue = NodeQueue::from((0..wp_graph.get_graph().get_num_nodes()).collect());
    let mut total_delta = 0.;
    let mut aggregated_edge_weights = Vec::new();
    while let Some(node_i) = node_queue.pop_front() {
        let maybe_best_community =
            wp_graph.best_community_for_node(node_i, &mut aggregated_edge_weights);

        let (best_community, best_delta) = if let Some((bc, bd)) = maybe_best_community {
            (bc, bd)
        } else {
            continue;
        };

        if best_delta > 0.0 {
            total_delta += best_delta;
            // Note that we are using the update partition fast here, which will
            // not guarantee that the partition_to_node_map is up to date.
            // To ensure that the partition_to_node_map is up to date, we need to call
            // rebuild_partition_to_node_map_if_stale after the local moving phase.
            // See below.
            wp_graph.update_partition_fast(node_i, best_community);

            wp_graph
                .get_graph()
                .neighbors_iter(node_i)
                .filter(|neighbor| {
                    wp_graph.get_partitioning().get_partition_for_node(neighbor) != best_community
                })
                .for_each(|neighbor_to_update| {
                    node_queue.push_back(neighbor_to_update);
                });
        }
    }

    // Make sure we rebuild the partition_to_node_map after the local moving phase.
    // To ensure that the partition_to_node_map is up to date, for later callers.
    wp_graph
        .get_partitioning_mut()
        .rebuild_partition_to_node_map_if_stale();

    total_delta
}

pub enum ThresholdOptions {
    Absolute(usize),
    Relative(f64),
}

/// Merge nodes connected by strong edges
///
/// The threshold to identify such edges can either be specified in two ways:
/// - in terms of absolute number of edges, e.g. any two partitions with more than X edges between
///   them will be merged
/// - in terms of number of edges relative to the size of the smallest of the two partitions, e.g.
///   if any two partitions will be merged if there are connected by more than more than X edges per
///   umis in the smallest partition
fn merge_highly_connected_nodes<Q: QualityMetrics>(
    wp_graph: &mut WeightedPartitionedGraph<Q>,
    merge_threshold: ThresholdOptions,
) {
    let graph = wp_graph.get_graph();
    let nodes_to_merge = graph
        .connected_components_by(|node_1, node_2| match merge_threshold {
            ThresholdOptions::Absolute(t) => graph.get_edge_weight(node_1, node_2).unwrap() > t,
            ThresholdOptions::Relative(rt) => {
                let node_weights = wp_graph.get_node_weights();
                graph.get_edge_weight(node_1, node_2).unwrap() as f64
                    > rt * min(node_weights[node_1], node_weights[node_2]) as f64
            }
        })
        .collect::<Vec<HashSet<NodeIdx>>>();

    for nodes in nodes_to_merge {
        let mut nodes_iter = nodes.into_iter();
        let pid =
            wp_graph.get_partitioning().get_node_to_partition_map()[nodes_iter.next().unwrap()];
        for node_id in nodes_iter {
            // Using the update partition fast here, which will
            // not guarantee that the partition_to_node_map is up to date.
            // To ensure that the partition_to_node_map is up to date, we need to call
            // rebuild_partition_to_node_map_if_stale after the merge phase.
            // See below.
            wp_graph.update_partition_fast(node_id, pid);
        }
    }
    // Make sure we rebuild the partition_to_node_map after the merge phase.
    // To ensure that the partition_to_node_map is up to date, for later callers.
    wp_graph
        .get_partitioning_mut()
        .rebuild_partition_to_node_map_if_stale();

    wp_graph.aggregate(AggregateOptions::All);
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::common::graph::Graph;
    use crate::common::node_partitioning::LeidenNodePartitioning;
    use crate::common::types::{Edge, edges_from_tuples};
    use crate::leiden::quality::cpm::ConstantPottsModel;
    use crate::leiden::quality::modularity::Modularity;

    use crate::common::test_utils::{get_2_communities_graph, get_example_graph, get_random_graph};

    #[test]
    fn test_leiden_cpm() {
        let graph = get_example_graph::<usize>();
        let num_nodes = graph.get_num_nodes();
        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(num_nodes);
        let cpm = ConstantPottsModel { resolution: 0.1 };
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, cpm, None, None);
        let randomness = 1.;

        leiden(&mut wp_graph, randomness, Some(100), None);

        assert_eq!(wp_graph.get_ancestor_to_partition_map(), vec![0; num_nodes]);
    }

    #[test]
    fn test_leiden_cpm_disconnected_graph() {
        //    2      5
        //   / \    / \
        //  0 - 1  3 - 4
        let edges = vec![(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
        let num_nodes = 6;
        let edge_iter = edges_from_tuples(edges.clone());
        let graph = Graph::<usize>::from_edges(edge_iter, num_nodes);

        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(num_nodes);
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let randomness = 1.;
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, cpm, None, None);

        leiden(&mut wp_graph, randomness, None, None);
        let result_partition_map = wp_graph.get_ancestor_to_partition_map();

        assert_eq!(result_partition_map[0], result_partition_map[1]);
        assert_eq!(result_partition_map[0], result_partition_map[2]);

        assert_eq!(result_partition_map[3], result_partition_map[4]);
        assert_eq!(result_partition_map[3], result_partition_map[5]);
    }

    #[test]
    fn test_leiden_cpm_2_communities() {
        let graph = get_2_communities_graph();
        let num_nodes = graph.get_num_nodes();
        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(num_nodes);
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let randomness = 1.;
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, cpm, None, None);

        leiden(&mut wp_graph, randomness, None, None);

        let result_partition_map = wp_graph.get_ancestor_to_partition_map();

        assert_eq!(result_partition_map.len(), num_nodes);

        assert_eq!(result_partition_map[0], result_partition_map[1]);
        assert_eq!(result_partition_map[0], result_partition_map[2]);
        assert_eq!(result_partition_map[0], result_partition_map[3]);

        assert_ne!(result_partition_map[0], result_partition_map[4]);

        assert_eq!(result_partition_map[4], result_partition_map[5]);
        assert_eq!(result_partition_map[4], result_partition_map[6]);
        assert_eq!(result_partition_map[4], result_partition_map[7]);
    }

    #[test]
    fn test_leiden_modularity() {
        let graph = get_example_graph::<usize>();
        let num_nodes = graph.get_num_nodes();
        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(num_nodes);
        let quality = Modularity::new(0.1, graph.get_total_edge_weight());
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, quality, None, None);
        let randomness = 1.;

        leiden(&mut wp_graph, randomness, Some(100), None);

        assert_eq!(wp_graph.get_ancestor_to_partition_map(), vec![0; num_nodes]);
    }

    #[test]
    fn test_leiden_modularity_disconnected_graph() {
        //    2      5
        //   / \    / \
        //  0 - 1  3 - 4
        let edges = vec![(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
        let num_nodes = 6;
        let edge_iter = edges_from_tuples(edges.clone());
        let graph = Graph::<usize>::from_edges(edge_iter, num_nodes);
        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(num_nodes);
        let quality = Modularity::new(0.5, graph.get_total_edge_weight());
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, quality, None, None);
        let randomness = 1.;

        leiden(&mut wp_graph, randomness, None, None);
        let result_partition_map = wp_graph.get_ancestor_to_partition_map();

        assert_eq!(result_partition_map[0], result_partition_map[1]);
        assert_eq!(result_partition_map[0], result_partition_map[2]);

        assert_ne!(result_partition_map[0], result_partition_map[3]);

        assert_eq!(result_partition_map[3], result_partition_map[4]);
        assert_eq!(result_partition_map[3], result_partition_map[5]);
    }

    #[test]
    fn test_leiden_modularity_2_communities() {
        let graph = get_2_communities_graph::<usize>();
        let num_nodes = graph.get_num_nodes();
        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(num_nodes);
        let quality = Modularity::new(0.5, graph.get_total_edge_weight());
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, quality, None, None);
        let randomness = 1.;

        leiden(&mut wp_graph, randomness, None, None);
        let result_partition_map = wp_graph.get_ancestor_to_partition_map();

        assert_eq!(result_partition_map.len(), num_nodes);

        assert_eq!(result_partition_map[0], result_partition_map[1]);
        assert_eq!(result_partition_map[0], result_partition_map[2]);
        assert_eq!(result_partition_map[0], result_partition_map[3]);

        assert_ne!(result_partition_map[0], result_partition_map[4]);

        assert_eq!(result_partition_map[4], result_partition_map[5]);
        assert_eq!(result_partition_map[4], result_partition_map[6]);
        assert_eq!(result_partition_map[4], result_partition_map[7]);
    }

    #[test]
    fn test_leiden_modularity_no_edges() {
        let edges = vec![];
        let num_nodes = 3;
        let edge_iter = edges_from_tuples(edges.clone());
        let graph = Graph::<usize>::from_edges(edge_iter, num_nodes);

        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(num_nodes);
        let quality = Modularity::new(0.5, graph.get_total_edge_weight());
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, quality, None, None);
        let randomness = 1.;

        leiden(&mut wp_graph, randomness, None, None);
        let result_partition_map = wp_graph.get_partitioning().get_node_to_partition_map();

        assert_ne!(result_partition_map[0], result_partition_map[1]);
        assert_ne!(result_partition_map[0], result_partition_map[2]);
    }

    #[test]
    fn test_leiden_uses_absolute_merge_threshold() {
        let edges = vec![
            Edge::new(0, 1, Some(6)),
            Edge::new(1, 2, Some(6)),
            Edge::new(2, 0, Some(15)),
        ];
        let num_nodes = 3;
        let graph = Graph::<usize>::from_edges(edges.into_iter(), num_nodes);
        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(num_nodes);
        let quality = Modularity::new(0.5, graph.get_total_edge_weight());
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, quality, None, None);

        leiden(
            &mut wp_graph,
            1.0,
            Some(0),
            Some(ThresholdOptions::Absolute(10)),
        );

        assert_eq!(wp_graph.get_graph().get_num_nodes(), 2);
        assert_eq!(wp_graph.get_graph().get_edge_weight(0, 1), Some(12));
    }

    #[test]
    fn test_leiden_uses_relative_merge_threshold() {
        let edges = vec![
            Edge::new(0, 1, Some(4)),
            Edge::new(1, 2, Some(6)),
            Edge::new(2, 0, Some(15)),
        ];
        let num_nodes = 3;
        let graph = Graph::<usize>::from_edges(edges.into_iter(), num_nodes);
        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(num_nodes);
        let quality = Modularity::new(0.5, graph.get_total_edge_weight());
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, quality, None, None);

        leiden(
            &mut wp_graph,
            1.0,
            Some(0),
            Some(ThresholdOptions::Relative(5.0)),
        );

        assert_eq!(wp_graph.get_graph().get_num_nodes(), 2);
        assert_eq!(wp_graph.get_graph().get_edge_weight(0, 1), Some(10));
    }

    #[test]
    fn test_move_nodes_fast() {
        let graph = get_example_graph::<usize>();
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let partitioning =
            LeidenNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, cpm, None, None);
        move_nodes_fast(&mut wp_graph);

        // I have gone over these results manually to verify that they make sense
        assert_eq!(wp_graph.get_partitioning().num_partitions(), 4);
        assert_eq!(
            wp_graph.get_partitioning().get_node_to_partition_map(),
            vec![2, 8, 2, 7, 7, 7, 6, 7, 8],
        );
    }

    #[test]
    fn test_merge_highly_connected_nodes_absolute() {
        // NB the current merge strategy can lead to partitions being highly connected after
        // applying the merge function. For instance, for this test case, partitions [0, 2] and 1 will
        // end up being connected with a weight of 12., above the original threshold of 10
        let edges = vec![
            Edge::new(0, 1, Some(6)),
            Edge::new(1, 2, Some(6)),
            Edge::new(2, 0, Some(15)),
        ];

        let num_nodes = 3;
        let graph = Graph::<usize>::from_edges(edges.into_iter(), num_nodes);
        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(num_nodes);
        let quality = Modularity::new(0.5, graph.get_total_edge_weight());
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, quality, None, None);

        merge_highly_connected_nodes(&mut wp_graph, ThresholdOptions::Absolute(10));

        assert_eq!(wp_graph.get_graph().get_num_nodes(), 2);

        // NB this includes a self edge
        assert_eq!(wp_graph.get_graph().get_total_edge_weight(), 27);
        assert_eq!(wp_graph.get_graph().get_edge_weight(0, 1), Some(12));
    }

    #[test]
    fn test_merge_highly_connected_nodes_relative() {
        let edges = vec![
            Edge::new(0, 1, Some(4)),
            Edge::new(1, 2, Some(6)),
            Edge::new(2, 0, Some(15)),
        ];

        let num_nodes = 3;
        let graph = Graph::<usize>::from_edges(edges.into_iter(), num_nodes);
        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(num_nodes);
        let quality = Modularity::new(0.5, graph.get_total_edge_weight());
        let mut wp_graph = WeightedPartitionedGraph::new(graph, partitioning, quality, None, None);

        merge_highly_connected_nodes(&mut wp_graph, ThresholdOptions::Relative(5.));

        assert_eq!(wp_graph.get_graph().get_num_nodes(), 2);
        assert_eq!(wp_graph.get_graph().get_total_edge_weight(), 25);
        assert_eq!(wp_graph.get_graph().get_edge_weight(0, 1), Some(10));
    }

    #[test]
    fn test_leiden_random_sparse() {
        let n_nodes = 250;
        let n_edges = 2 * n_nodes;
        let seed = 0;
        let graph = get_random_graph(n_nodes, n_edges, seed);

        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(n_nodes);
        let quality = Modularity::new(0.5, n_edges);
        let mut wp_graph =
            WeightedPartitionedGraph::new(graph.clone(), partitioning, quality, None, None);
        let randomness = 1.;

        leiden(&mut wp_graph, randomness, None, None);
        let result_partition = LeidenNodePartitioning::initialize_from_partitions(
            wp_graph.get_ancestor_to_partition_map(),
        );

        assert_eq!(
            graph
                .connected_components_by(|node_1, node_2| result_partition
                    .get_node_to_partition_map()[node_1]
                    == result_partition.get_node_to_partition_map()[node_2])
                .count(),
            result_partition.num_partitions()
        );
    }

    #[test]
    fn test_leiden_random_dense() {
        let n_nodes = 250;
        let n_edges = 10 * n_nodes;
        let seed = 0;
        let graph = get_random_graph(n_nodes, n_edges, seed);

        let partitioning = LeidenNodePartitioning::initialize_with_singlet_partitions(n_nodes);
        let quality = Modularity::new(4., n_edges);
        let mut wp_graph =
            WeightedPartitionedGraph::new(graph.clone(), partitioning, quality, None, None);
        let randomness = 1.;

        leiden(&mut wp_graph, randomness, None, None);
        let result_partition = LeidenNodePartitioning::initialize_from_partitions(
            wp_graph.get_ancestor_to_partition_map(),
        );

        assert_eq!(
            graph
                .connected_components_by(|node_1, node_2| result_partition
                    .get_node_to_partition_map()[node_1]
                    == result_partition.get_node_to_partition_map()[node_2])
                .count(),
            result_partition.num_partitions()
        );
    }
}
