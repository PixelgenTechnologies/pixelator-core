use rustc_hash::FxHashSet as HashSet;
use sprs::{CsMat, TriMat};
use std::collections::VecDeque;
use std::iter;

use crate::common::types::{Edge, EdgeWeight, NodeIdx};

mod statistics;

pub use statistics::GraphProperties;

/// A simple undirected weighted graph stored as a sparse adjacency matrix.
///
/// The input is interpreted as undirected: whenever you add an edge `(u, v)` with `u != v`,
/// we store it in both adjacency entries `(u, v)` and `(v, u)`. Self-loops `(u, u)` are stored
/// once as `(u, u)`.
///
/// The `Graph` exposes two common "edge size" metrics that differ in meaning:
/// - `edge_entry_count`: counts adjacency entries (non-zero matrix entries in the
///   upper-triangular part of the matrix, i.e. we count each undirected edge once).
/// - `total_edge_weight`: sums the weight values associated with those entries.
///
/// This distinction is important when edges can carry weights: many algorithms (and our
/// quality metrics) need the total *weight*, not just the number of adjacency entries.
///
/// In particular, when we build an aggregated graph during Leiden (`WeightedPartitionedGraph::aggregate`),
/// each "node" represents a set of original nodes (a community at the current level) and each
/// adjacency weight represents the sum of weights of all original edges between the two sets.
/// Therefore, `edge_entry_count` tells you only how many distinct neighbor-community pairs
/// exist, while `total_edge_weight` preserves how strongly communities are connected.
#[derive(Clone)]
pub struct Graph<T> {
    adjacency_matrix: CsMat<T>,
    num_nodes: usize,
    edge_entry_count: usize,
    total_edge_weight: usize,
    norm: f64,
    degrees: Vec<usize>,
}

impl<T: EdgeWeight> Graph<T> {
    pub fn from_edges<I>(edges: I, num_nodes: NodeIdx) -> Self
    where
        I: Iterator<Item = Edge<T>>,
    {
        let mut tri_mat = TriMat::<T>::new((num_nodes, num_nodes));
        for Edge { src, dest, weight } in edges {
            tri_mat.add_triplet(src, dest, weight);
            if src != dest {
                tri_mat.add_triplet(dest, src, weight);
            }
        }

        let adjacency_matrix = tri_mat.to_csr();

        Self::from_adjacency_matrix(adjacency_matrix)
    }

    pub fn from_adjacency_matrix(adjacency_matrix: CsMat<T>) -> Self {
        let num_nodes = adjacency_matrix.rows();
        let edge_entry_count = adjacency_matrix
            .view()
            .iter_rbr()
            .filter(|(_, (node_1, node_2))| node_1 <= node_2)
            .count();
        let total_edge_weight = adjacency_matrix
            .view()
            .iter_rbr()
            .filter(|(_, (node_1, node_2))| node_1 <= node_2)
            .map(|(&weight, _)| weight.into())
            .sum::<usize>();
        let degrees = adjacency_matrix.degrees();
        let degrees_sum = degrees.iter().sum::<usize>();
        let norm = 1. / (degrees_sum * degrees_sum) as f64;
        Graph {
            adjacency_matrix,
            num_nodes,
            edge_entry_count,
            total_edge_weight,
            norm,
            degrees,
        }
    }

    pub fn get_adjacency_matrix(&self) -> &CsMat<T> {
        &self.adjacency_matrix
    }

    pub fn get_num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of undirected adjacency entries in the graph.
    ///
    /// Each undirected edge contributes one entry (`src <= dest`), and self loops
    /// also contribute one entry.
    pub fn get_edge_entry_count(&self) -> usize {
        self.edge_entry_count
    }

    /// Returns the sum of all undirected edge weights in the graph.
    ///
    /// The sum is taken over the same set of undirected adjacency entries as
    /// [`Self::get_edge_entry_count`] (upper triangle `src <= dest`, including diagonal/self-loops).
    ///
    /// For unweighted graphs where every edge has weight `1`, this equals
    /// [`Self::get_edge_entry_count`].
    pub fn get_total_edge_weight(&self) -> usize {
        self.total_edge_weight
    }

    pub fn get_edge_weight(&self, src: NodeIdx, dst: NodeIdx) -> Option<T> {
        self.adjacency_matrix.get(src, dst).copied()
    }

    pub fn get_norm(&self) -> f64 {
        self.norm
    }

    pub fn get_degrees(&self) -> &Vec<usize> {
        &self.degrees
    }

    /// Returns a list of nodes connected to `node`. This does not include `node` even when it is
    /// connected to itself.
    pub fn neighbors(&self, node: NodeIdx) -> Vec<NodeIdx> {
        self.get_adjacency_matrix()
            .outer_view(node)
            .unwrap_or_else(|| panic!("Node index out of bounds: {}", node))
            .iter()
            .map(|(neighbor_idx, _)| neighbor_idx)
            .filter(|&neighbor_id| neighbor_id != node)
            .collect()
    }

    /// Returns an iterator over node indices connected to `node` (excluding `node` itself).
    ///
    /// This is the allocation-free counterpart of [`Self::neighbors`], intended for tight loops.
    pub fn neighbors_iter(&self, node: NodeIdx) -> impl Iterator<Item = NodeIdx> + '_ {
        if node >= self.num_nodes {
            panic!("Node index out of bounds: {}", node);
        }

        // We store the graph as CSR, so we can iterate a row without allocating intermediate Vecs.
        // The neighbors for `node` are the column indices in `indices[indptr[node]..indptr[node+1]]`.
        let indptr_base = self.get_adjacency_matrix().indptr();
        let indptr = indptr_base
            .as_slice()
            .expect("CSR indptr should expose a contiguous slice");
        let indices = self.get_adjacency_matrix().indices();
        let start = indptr[node];
        let end = indptr[node + 1];

        indices[start..end]
            .iter()
            .copied()
            .filter(move |&neighbor_id| neighbor_id != node)
    }

    /// Returns a list of edge connected to `node`, including self edges looping back
    pub fn get_edges_from(&self, node: NodeIdx) -> Vec<Edge<T>> {
        self.get_adjacency_matrix()
            .outer_view(node)
            .unwrap_or_else(|| panic!("Node index out of bounds: {}", node))
            .iter()
            .map(|(neighbor_id, &weight)| Edge::new(node, neighbor_id, Some(weight)))
            .collect()
    }

    /// Returns an iterator over edges incident to `node` (including self-loops).
    ///
    /// This is the allocation-free counterpart of [`Self::get_edges_from`], intended for tight loops.
    pub fn edges_from_iter(&self, node: NodeIdx) -> impl Iterator<Item = Edge<T>> + '_ {
        if node >= self.num_nodes {
            panic!("Node index out of bounds: {}", node);
        }

        // CSR: iterate row entries by matching `indices` and `data` in lockstep.
        let indptr_base = self.get_adjacency_matrix().indptr();
        let indptr = indptr_base
            .as_slice()
            .expect("CSR indptr should expose a contiguous slice");
        let indices = self.get_adjacency_matrix().indices();
        let data = self.get_adjacency_matrix().data();
        let start = indptr[node];
        let end = indptr[node + 1];

        indices[start..end]
            .iter()
            .copied()
            .zip(data[start..end].iter().copied())
            .map(move |(neighbor_id, weight)| Edge::new(node, neighbor_id, Some(weight)))
    }

    /// Iterate over all edges of the graph
    ///
    /// NB: this will iterate over edges in both directions
    pub fn get_edges_iter(&self) -> impl Iterator<Item = Edge<T>> {
        self.adjacency_matrix
            .view()
            .iter_rbr()
            .map(|(&weight, (node_1, node_2))| Edge::new(node_1, node_2, Some(weight)))
    }

    /// Get all edges where at least one of the nodes is in the selection
    /// Note that this will include edges that go outside the selection, and edges
    /// that loop back to the same node.
    /// Every egde will only be included once, i.e. (0,1) and (1,0) will not both be included
    /// as (0,1). The edges will be returned in arbitrary order.
    pub fn get_edges_in_selection(
        &self,
        selection: &HashSet<NodeIdx>,
    ) -> impl Iterator<Item = Edge<T>> {
        selection.iter().flat_map(|&node| {
            self.edges_from_iter(node)
                .filter(|Edge { src, dest, .. }| !(src > dest && selection.contains(dest)))
        })
    }

    /// Returns connected components of the graph
    pub fn connected_components(&self) -> impl Iterator<Item = HashSet<NodeIdx>> {
        self.connected_components_by(|_, _| true)
    }

    /// Returns connected components where nodes in each components fulfill the `filter` criteria.
    ///
    /// This criteria must be transitive, ie. filter(a, b) ∧ filter(b, c) ⇒ filter(a, c)
    pub fn connected_components_by<F>(&self, filter: F) -> impl Iterator<Item = HashSet<NodeIdx>>
    where
        F: Fn(NodeIdx, NodeIdx) -> bool,
    {
        let mut node_queue: VecDeque<NodeIdx> = VecDeque::from_iter(0..self.get_num_nodes());
        let mut visited: HashSet<NodeIdx> = HashSet::default();

        iter::from_fn(move || {
            let start_node = self.find_first_non_visited_node(&mut node_queue, &visited)?;
            let component = self.breath_first_search(start_node, &mut visited, &filter);
            Some(component)
        })
    }

    fn find_first_non_visited_node(
        &self,
        node_queue: &mut VecDeque<NodeIdx>,
        visited: &HashSet<NodeIdx>,
    ) -> Option<NodeIdx> {
        while let Some(node) = node_queue.pop_front() {
            if !visited.contains(&node) {
                return Some(node);
            }
        }
        None
    }

    fn breath_first_search<F>(
        &self,
        start_node: NodeIdx,
        visited: &mut HashSet<NodeIdx>,
        filter: &F,
    ) -> HashSet<NodeIdx>
    where
        F: Fn(NodeIdx, NodeIdx) -> bool,
    {
        let mut component: HashSet<NodeIdx> = HashSet::default();
        let mut node_queue = VecDeque::from(vec![start_node]);

        while let Some(node) = node_queue.pop_front() {
            if visited.contains(&node) {
                continue;
            }
            visited.insert(node);
            component.insert(node);

            for neighbor in self.neighbors(node) {
                if visited.contains(&neighbor) || !filter(node, neighbor) {
                    continue;
                }
                node_queue.push_back(neighbor);
            }
        }
        component
    }
}

impl From<Graph<u8>> for Graph<usize> {
    fn from(graph: Graph<u8>) -> Graph<usize> {
        let shape = graph.adjacency_matrix.shape();
        let indptr = graph.adjacency_matrix.indptr();
        let indices = graph.adjacency_matrix.indices();
        let data = graph.adjacency_matrix.data();
        let adjacency_matrix = CsMat::new(
            shape,
            indptr.as_slice().unwrap().into(),
            indices.into(),
            data.iter().map(|&v| v as usize).collect(),
        );

        Graph::<usize>::from_adjacency_matrix(adjacency_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::common::types::edges_from_tuples;

    #[test]
    fn test_graph_creation_and_stats() {
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 0)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);
        assert_eq!(graph.get_num_nodes(), 3); // 0, 1, 2
        assert_eq!(graph.get_edge_entry_count(), 3);
        assert_eq!(graph.get_total_edge_weight(), 3);
    }

    #[test]
    fn test_edge_metrics() {
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 0), (0, 0), (0, 1)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);
        assert_eq!(graph.get_edge_entry_count(), 4);
        assert_eq!(graph.get_total_edge_weight(), 5);
    }

    #[test]
    fn test_graph_neighbors() {
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 0)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);
        let neighbors_0 = graph.neighbors(0);
        assert!(neighbors_0.contains(&1));
        assert!(neighbors_0.contains(&2));
        assert!(!neighbors_0.contains(&0));
    }

    #[test]
    #[should_panic(expected = "Node index out of bounds")]
    fn test_graph_neighbors_out_of_bounds() {
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 0)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);
        graph.neighbors(3); // out of bounds
    }

    #[test]
    fn test_graph_neighbors_complex() {
        //      0
        //     / \
        //    1   4
        //    |
        //    2
        //   / \
        //  3   5
        //
        //  6 (isolated)

        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 3), (0, 4), (2, 5)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 7);

        // Node 0: neighbors 1, 4
        let n0 = graph.neighbors(0);
        assert_eq!(n0.len(), 2);
        assert!(n0.contains(&1));
        assert!(n0.contains(&4));

        // Node 2: neighbors 1, 3, 5
        let n2 = graph.neighbors(2);
        assert_eq!(n2.len(), 3);
        assert!(n2.contains(&1));
        assert!(n2.contains(&3));
        assert!(n2.contains(&5));

        // Node 6: isolated
        let n6 = graph.neighbors(6);
        assert_eq!(n6.len(), 0);

        // Node 3: only neighbor is 2
        let n3 = graph.neighbors(3);
        assert_eq!(n3, vec![2]);
    }

    #[test]
    fn test_neighbors_iter_equivalent_graph_neighbors_complex() {
        // Same graph and expectations as `test_graph_neighbors_complex`,
        // but validating the allocation-free neighbor iterator.
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 3), (0, 4), (2, 5)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 7);

        let mut n0_alloc = graph.neighbors(0);
        n0_alloc.sort_unstable();
        let mut n0_iter: Vec<NodeIdx> = graph.neighbors_iter(0).collect();
        n0_iter.sort_unstable();
        assert_eq!(n0_alloc, n0_iter);

        let mut n2_alloc = graph.neighbors(2);
        n2_alloc.sort_unstable();
        let mut n2_iter: Vec<NodeIdx> = graph.neighbors_iter(2).collect();
        n2_iter.sort_unstable();
        assert_eq!(n2_alloc, n2_iter);

        assert!(graph.neighbors_iter(6).next().is_none());
        let n6_iter: Vec<NodeIdx> = graph.neighbors_iter(6).collect();
        assert_eq!(n6_iter, graph.neighbors(6));

        let n3_alloc = graph.neighbors(3);
        let n3_iter: Vec<NodeIdx> = graph.neighbors_iter(3).collect();
        assert_eq!(n3_iter, n3_alloc);
    }

    #[test]
    fn test_edges_from() {
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 0), (0, 0)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);
        let edges_from_0 = graph.get_edges_from(0);
        assert!(edges_from_0.contains(&Edge {
            src: 0,
            dest: 1,
            weight: 1
        }));
        assert!(edges_from_0.contains(&Edge {
            src: 0,
            dest: 2,
            weight: 1
        }));
        assert!(edges_from_0.contains(&Edge {
            src: 0,
            dest: 0,
            weight: 1
        }));
    }

    #[test]
    fn test_neighbors_iter_equivalent_neighbors() {
        // Include a self-loop to ensure `neighbors*` still excludes the node itself.
        let edges = vec![
            Edge::new(0, 1, Some(3_u8)),
            Edge::new(1, 2, Some(5_u8)),
            Edge::new(0, 0, Some(7_u8)),
        ];
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);

        let mut alloc_neighbors = graph.neighbors(0);
        let iter_neighbors: Vec<NodeIdx> = graph.neighbors_iter(0).collect();

        alloc_neighbors.sort_unstable();
        let mut iter_neighbors_sorted = iter_neighbors;
        iter_neighbors_sorted.sort_unstable();

        assert_eq!(alloc_neighbors, iter_neighbors_sorted);
    }

    #[test]
    fn test_edges_from_iter_equivalent_get_edges_from() {
        // Include a self-loop and varying weights to validate exact values.
        let edges = vec![
            Edge::new(0, 1, Some(3_u8)),
            Edge::new(1, 2, Some(5_u8)),
            Edge::new(0, 0, Some(7_u8)),
        ];
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 3);

        let mut alloc_edges = graph.get_edges_from(0);
        let mut iter_edges: Vec<Edge<u8>> = graph.edges_from_iter(0).collect();

        let mut alloc_tuples: Vec<(NodeIdx, NodeIdx, u8)> = alloc_edges
            .drain(..)
            .map(|e| (e.src, e.dest, e.weight))
            .collect();
        let mut iter_tuples: Vec<(NodeIdx, NodeIdx, u8)> = iter_edges
            .drain(..)
            .map(|e| (e.src, e.dest, e.weight))
            .collect();

        alloc_tuples.sort_unstable();
        iter_tuples.sort_unstable();

        assert_eq!(alloc_tuples, iter_tuples);
    }

    #[test]
    fn test_edges_in_selection() {
        //      ()
        //      0
        //     / \
        //    1   4
        //    |
        //    2
        //   / \
        //  3   5
        //
        //  6 (isolated)

        let edges = edges_from_tuples(vec![(0, 0), (0, 1), (1, 2), (2, 3), (0, 4), (2, 5)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 7);

        let selection: HashSet<NodeIdx> = vec![0, 1].into_iter().collect();
        let selected_edges: HashSet<Edge<u8>> = graph.get_edges_in_selection(&selection).collect();

        let expected_edges: HashSet<Edge<u8>> =
            edges_from_tuples(vec![(0, 0), (0, 1), (1, 2), (0, 4)]).collect();

        assert_eq!(selected_edges, expected_edges);
    }

    #[test]
    fn test_connected_components_simple() {
        // Graph: 0-1  2-3  4 (isolated)
        let edges = edges_from_tuples(vec![(0, 1), (2, 3)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 5);

        let mut components: Vec<HashSet<NodeIdx>> = graph.connected_components().collect();
        components.sort_by_key(|c| c.iter().min().cloned().unwrap());

        assert_eq!(components.len(), 3);
        assert_eq!(components[0], [0, 1].into_iter().collect());
        assert_eq!(components[1], [2, 3].into_iter().collect());
        assert_eq!(components[2], [4].into_iter().collect());
    }

    #[test]
    fn test_connected_components_complex() {
        //      0
        //     / \
        //    1   4
        //    |
        //    2
        //   / \
        //  3   5
        //
        //  6 (isolated)
        let edges = edges_from_tuples(vec![(0, 1), (1, 2), (2, 3), (0, 4), (2, 5)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 7);

        let mut components: Vec<HashSet<NodeIdx>> = graph.connected_components().collect();
        components.sort_by_key(|c| c.iter().min().cloned().unwrap());

        assert_eq!(components.len(), 2);
        assert_eq!(components[0], [0, 1, 2, 3, 4, 5].into_iter().collect());
        assert_eq!(components[1], [6].into_iter().collect());
    }

    #[test]
    fn test_connected_components_by() {
        // 0(-)1
        // |   |
        // 2(-)3
        //
        // Only consider components where all node indices have the same parity

        let edges = edges_from_tuples(vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
        let graph = Graph::<u8>::from_edges(edges.into_iter(), 4);

        let mut components: Vec<HashSet<NodeIdx>> = graph
            .connected_components_by(|src_node, node| src_node % 2 == node % 2)
            .collect();

        components.sort_by_key(|c| c.iter().min().cloned().unwrap());

        assert_eq!(components.len(), 2);
        assert_eq!(components[0], [0, 2].into_iter().collect());
        assert_eq!(components[1], [1, 3].into_iter().collect());
    }
}
