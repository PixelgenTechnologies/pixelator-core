//! Graph statistics utilities.
//!
//! This module provides tools for computing basic statistics and properties of graphs,
//! such as node and edge counts, number of connected components, and the fraction of nodes
//! in the largest connected component.
//!

use crate::common::constants::MIN_PNA_COMPONENT_SIZE;
use crate::common::{graph::Graph, types::EdgeWeight};
use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;

/// Stores basic properties/statistics of a graph.
#[derive(Debug, Clone)]
pub struct GraphProperties {
    /// Number of nodes in the graph.
    pub node_count: usize,
    /// Sum of undirected edge weights in the graph.
    pub edge_weight_sum: usize,
    /// Number of connected components in the graph.
    pub n_connected_components: usize,
    /// Fraction of nodes in the largest connected component.
    pub fraction_in_largest_component: f64,
    /// Number of nodes in components smaller than 8000 umis
    pub stranded_nodes: usize,
    /// Distribution of the component sizes
    pub component_size_distribution: HashMap<usize, usize>,
}

impl GraphProperties {
    /// Computes graph properties for the given graph.
    pub fn new<T: EdgeWeight>(graph: &Graph<T>) -> Self {
        let node_count = graph.get_num_nodes();
        let edge_weight_sum = graph.get_total_edge_weight();

        let labels = graph.component_labels();
        let mut component_sizes =
            vec![0usize; labels.iter().copied().max().map(|max| max + 1).unwrap_or(0)];
        for label in labels {
            component_sizes[label] += 1;
        }
        let component_count = component_sizes.len();

        let fraction_in_largest_component =
            component_sizes.iter().max().copied().unwrap_or(0) as f64 / node_count as f64;
        let stranded_nodes = component_sizes
            .iter()
            .filter(|&&n_nodes| n_nodes < MIN_PNA_COMPONENT_SIZE)
            .sum();
        let component_size_distribution = component_sizes
            .iter()
            .copied()
            .counts()
            .into_iter()
            .collect();

        GraphProperties {
            node_count,
            edge_weight_sum,
            n_connected_components: component_count,
            fraction_in_largest_component,
            stranded_nodes,
            component_size_distribution,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::{GraphProperties, HashMap};
    use crate::common::graph::Graph;
    use crate::common::types::Edge;
    use crate::common::types::edges_from_tuples;

    #[test]
    fn test_graph_properties_single_component() {
        // Graph: 0-1-2
        let edges = edges_from_tuples(vec![(0, 1), (1, 2)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);
        let props = GraphProperties::new(&graph);
        assert_eq!(props.node_count, 3);
        assert_eq!(props.edge_weight_sum, 2);
        assert_eq!(props.n_connected_components, 1);
        assert!((props.fraction_in_largest_component - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_graph_properties_multiple_components() {
        // Graph: 0-1  2 (isolated)
        let edges = edges_from_tuples(vec![(0, 1)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);
        let props = GraphProperties::new(&graph);
        assert_eq!(props.node_count, 3);
        assert_eq!(props.edge_weight_sum, 1);
        assert_eq!(props.n_connected_components, 2);
        // Largest component: 0-1 (size 2), fraction = 2/3
        assert!((props.fraction_in_largest_component - (2.0 / 3.0)).abs() < 1e-8);
    }

    #[test]
    fn test_graph_properties_all_isolated() {
        // Graph: 0 1 2 (all isolated)
        let edges: Vec<Edge<u8>> = vec![];
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);
        let props = GraphProperties::new(&graph);
        assert_eq!(props.node_count, 3);
        assert_eq!(props.edge_weight_sum, 0);
        assert_eq!(props.n_connected_components, 3);
        assert!((props.fraction_in_largest_component - (1.0 / 3.0)).abs() < 1e-8);
    }

    #[test]
    fn test_graph_properties_match_set_component_sizes() {
        use crate::common::constants::MIN_PNA_COMPONENT_SIZE;
        use crate::common::test_utils::get_random_graph;
        use itertools::Itertools;

        let graph = get_random_graph::<u8>(200, 400, 2);
        let mut reference_sizes = vec![0usize; graph.get_num_nodes()];
        let mut component_count = 0usize;
        for component in graph.connected_components() {
            let size = component.len();
            reference_sizes[component_count] = size;
            component_count += 1;
        }
        reference_sizes.truncate(component_count);

        let props = GraphProperties::new(&graph);
        let largest = reference_sizes.iter().copied().max().unwrap_or(0);
        let stranded = reference_sizes
            .iter()
            .filter(|&&size| size < MIN_PNA_COMPONENT_SIZE)
            .sum::<usize>();
        let distribution: HashMap<usize, usize> = reference_sizes
            .iter()
            .copied()
            .counts()
            .into_iter()
            .collect();

        assert_eq!(props.n_connected_components, component_count);
        assert_eq!(
            props.fraction_in_largest_component,
            largest as f64 / graph.get_num_nodes() as f64
        );
        assert_eq!(props.stranded_nodes, stranded);
        assert_eq!(props.component_size_distribution, distribution);
    }
}
