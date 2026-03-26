use crate::common::graph::Graph;
use crate::common::types::NodeIdx;
use crate::leiden::quality::QualityMetrics;

#[derive(Clone)]
pub struct ConstantPottsModel {
    pub resolution: f64,
}

impl QualityMetrics for ConstantPottsModel {
    fn resolution(&self) -> f64 {
        self.resolution
    }

    fn node_size(&self, _graph: &Graph<usize>, _node_id: NodeIdx) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rustc_hash::FxHashSet as HashSet;

    use crate::common::node_partitioning::{LeidenNodePartitioning, NodePartitioning};
    use crate::common::types::{Edge, edges_from_tuples};
    use crate::leiden::weighted_partitioned_graph::WeightedPartitionedGraph;

    use crate::common::test_utils::get_example_graph;

    #[test]
    fn test_cpm_get_well_connected_nodes_in_subset() {
        let graph = get_example_graph::<usize>();
        let partitioning =
            LeidenNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let wp_graph = WeightedPartitionedGraph::new(graph, partitioning, cpm, None, None);
        let subset: HashSet<usize> = vec![2, 3, 4, 5, 6, 7, 8].into_iter().collect();
        let subset_size = subset
            .iter()
            .map(|&id| wp_graph.get_node_weights()[id])
            .sum::<usize>();

        let mut well_connected_nodes = subset
            .iter()
            .filter(|&&node_id| wp_graph.node_is_well_connected(node_id, &subset, subset_size))
            .copied()
            .collect::<Vec<usize>>();
        well_connected_nodes.sort();

        assert_eq!(well_connected_nodes, vec![4, 5, 7, 8])
    }

    #[test]
    fn test_cpm_well_connected_nodes_in_subset_weighted_edges() {
        let cpm = ConstantPottsModel { resolution: 0.1 };

        let edges = vec![Edge {
            src: 0,
            dest: 1,
            weight: 2,
        }];
        let graph = Graph::<usize>::from_edges(edges.into_iter(), 2);
        let node_weights = vec![2, 7];
        let partitions = LeidenNodePartitioning::initialize_with_singlet_partitions(2);
        let wp_graph =
            WeightedPartitionedGraph::new(graph, partitions, cpm, Some(node_weights), None);

        let subset: HashSet<usize> = vec![0, 1].into_iter().collect();
        let subset_size = subset
            .iter()
            .map(|&id| wp_graph.get_node_weights()[id])
            .sum::<usize>();

        let mut well_connected_nodes = subset
            .iter()
            .filter(|&&node_id| wp_graph.node_is_well_connected(node_id, &subset, subset_size))
            .copied()
            .collect::<Vec<usize>>();

        well_connected_nodes.sort();

        assert_eq!(well_connected_nodes, vec![0, 1]);
    }

    #[test]
    fn test_cpm_well_connected_communities() {
        let graph = get_example_graph::<usize>();
        let partitions = LeidenNodePartitioning::initialize_with_singlet_partitions(9);
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let wp_graph = WeightedPartitionedGraph::new(graph, partitions, cpm, None, None);
        let subset: HashSet<usize> = vec![2, 3, 4, 5, 6, 7, 8].into_iter().collect();
        let subset_size = subset
            .iter()
            .map(|&id| wp_graph.get_node_weights()[id])
            .sum::<usize>();
        let communities_to_evaluate: HashSet<usize> = vec![2, 3, 4, 5, 7].into_iter().collect();

        let well_connected_communities = communities_to_evaluate
            .into_iter()
            .filter(|community_id| {
                wp_graph.community_is_well_connected(community_id, &subset, subset_size)
            })
            .collect::<HashSet<usize>>();

        let expected = HashSet::from_iter(vec![4, 5, 7]);
        assert_eq!(well_connected_communities, expected);
    }

    #[test]
    fn test_cpm_well_connected_communities_tetra() {
        //  0−−−−−1
        //   \ 3 /
        //    \|/
        //     2
        // This tests makes sure intra-community edges are not counted in the following formula:
        // E(C, S − C) ≥ γ ‖C‖ ⋅ (‖S‖ − ‖C‖)
        let edges = vec![(0, 1), (1, 2), (2, 0), (2, 3)];
        let edge_iter = edges_from_tuples(edges);
        let graph = Graph::<usize>::from_edges(edge_iter, 6);
        let partitions = LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 0, 1]);
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let wp_graph =
            WeightedPartitionedGraph::new(graph, partitions.clone(), cpm.clone(), None, None);

        let subset: HashSet<usize> = vec![0, 1, 2, 3].into_iter().collect();
        assert!(!wp_graph.community_is_well_connected(&0, &subset, 4));
    }

    #[test]
    fn test_cpm_intra_community_edges_increase_quality() {
        // Create a graph with two communities, each with three nodes and two edges within each community, plus some inter-community edges
        // Partition: [0,1,2] in community 0, [3,4,5] in community 1
        let edges = vec![
            (0, 1),
            (1, 2), // community 0
            (3, 4),
            (4, 5), // community 1
            (2, 3), // inter-community edge
            (0, 4), // more inter-community
        ];
        let edge_iter = edges_from_tuples(edges.clone());
        let graph = Graph::<usize>::from_edges(edge_iter, 6);
        let partitions = LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 0, 1, 1, 1]);
        let cpm = ConstantPottsModel { resolution: 0.5 };
        let wp_graph =
            WeightedPartitionedGraph::new(graph, partitions.clone(), cpm.clone(), None, None);
        let q_before = wp_graph.quality();

        // Add more intra-community edges: (0,2) and (3,5)
        let mut more_edges = edges.clone();
        more_edges.extend(vec![(0, 2), (3, 5)]); // Add one more intra edge to each community
        let edge_iter = edges_from_tuples(more_edges);
        let graph_more = Graph::<usize>::from_edges(edge_iter, 6);
        let wp_graph_more =
            WeightedPartitionedGraph::new(graph_more, partitions.clone(), cpm.clone(), None, None);
        let q_after = wp_graph_more.quality();

        assert!(
            q_after > q_before,
            "Adding intra-community edges should increase CPM quality (before: {}, after: {})",
            q_before,
            q_after
        );
    }
}
