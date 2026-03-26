#![cfg(test)]

use is_close;

use crate::common::graph::Graph;
use crate::common::types::{EdgeWeight, NodeIdx, edges_from_tuples};
use itertools::Itertools;
use rand::SeedableRng;
use rand::distr::{Distribution, Uniform};
use rand::rngs::StdRng;

pub fn get_example_edges() -> Vec<(NodeIdx, NodeIdx)> {
    // Using graph from:
    // https://www.figma.com/board/6JHFnKR0hScokRbVyYNg2j/Leiden?node-id=0-1&t=1LXERSVxJPoMK65N-1
    vec![
        (3, 4),
        (3, 7),
        (0, 2),
        (2, 4),
        (4, 7),
        (2, 8),
        (4, 8),
        (5, 4),
        (5, 7),
        (8, 5),
        (5, 6),
        (1, 8),
    ]
}

pub fn get_example_graph<T: EdgeWeight>() -> Graph<T> {
    let edge_iter = edges_from_tuples(get_example_edges());
    Graph::<T>::from_edges(edge_iter, 9)
}

pub fn get_2_communities_edges() -> Vec<(NodeIdx, NodeIdx)> {
    // 3 - 2   6 - 7
    // | X |   | X |
    // 0 - 1 - 4 - 5
    vec![
        (0, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
        (1, 4),
        (4, 5),
        (6, 4),
        (6, 5),
        (7, 4),
        (7, 5),
        (7, 6),
    ]
}

pub fn get_2_communities_graph<T: EdgeWeight>() -> Graph<T> {
    let num_nodes = 8;
    let edge_iter = edges_from_tuples(get_2_communities_edges());
    Graph::<T>::from_edges(edge_iter, num_nodes)
}

pub fn get_random_edges(n_nodes: usize, n_edges: usize, seed: u64) -> Vec<(NodeIdx, NodeIdx)> {
    if n_nodes == 1 {
        return vec![(0, 0); n_edges];
    }

    let mut rng = StdRng::seed_from_u64(seed);
    Uniform::new(0, n_nodes)
        .unwrap()
        .sample_iter(&mut rng)
        .chunks(2)
        .into_iter()
        .map(|chunk| {
            let node_pair = chunk.collect::<Vec<NodeIdx>>();
            (node_pair[0], node_pair[1])
        })
        .filter(|(n1, n2)| n1 != n2)
        .take(n_edges)
        .collect()
}

pub fn get_random_graph<T: EdgeWeight>(n_nodes: usize, n_edges: usize, seed: u64) -> Graph<T> {
    let edge_iter = edges_from_tuples(get_random_edges(n_nodes, n_edges, seed));
    Graph::<T>::from_edges(edge_iter, n_nodes)
}

pub fn is_close_to_zero(a: f64) -> bool {
    is_close::is_close!(a, 0.0, abs_tol = 1e-8)
}
