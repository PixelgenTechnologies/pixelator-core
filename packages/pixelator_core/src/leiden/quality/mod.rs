use crate::common::graph::Graph;
use crate::common::types::NodeIdx;

pub trait QualityMetrics {
    /// Resolution paramater used to compute quality and well connected nodes and communities. In
    /// the Leiden paper, this is referenced as γ.
    fn resolution(&self) -> f64;

    /// Metric to determine the node size i.e. ||.||
    ///
    /// In the CPM, all nodes have a constant size of 1. In the context of modularity, the size of
    /// a node is the number of edges it has (namely, its degree).
    fn node_size(&self, graph: &Graph<usize>, node_id: NodeIdx) -> usize;
}

pub mod cpm;
pub mod modularity;
