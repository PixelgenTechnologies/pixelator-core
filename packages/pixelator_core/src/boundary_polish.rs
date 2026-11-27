use rayon::prelude::*;
use smallvec::SmallVec;

use crate::common::graph::Graph;
use crate::common::types::PartitionId;

/// One round of boundary polish on a bipartite graph.
///
/// A node moves to a neighbouring community when strictly more of its edge weight goes to that
/// community than to its own. When several communities qualify, the one with the most edge
/// weight wins, and among equals the one with the smallest id.
///
/// Side A is updated first, reading only side B; then side B, reading only side A. All
/// neighbours of a node are on the other side, so no node reads a label that the same pass
/// writes, and the result does not depend on how the threads are scheduled.
///
/// Returns the number of moves.
pub fn boundary_polish(graph: &Graph<u8>, is_b: &[bool], communities: &mut [PartitionId]) -> usize {
    let n = graph.get_num_nodes();
    assert_eq!(
        is_b.len(),
        n,
        "Bipartite side mask length ({}) must match number of graph nodes ({})",
        is_b.len(),
        n
    );
    assert_eq!(
        communities.len(),
        n,
        "Number of community labels ({}) must match number of graph nodes ({})",
        communities.len(),
        n
    );

    let adj = graph.get_adjacency_matrix();
    let indptr_raw = adj.indptr();
    let indptr = indptr_raw
        .as_slice()
        .expect("CSR adjacency matrix indptr must be contiguous");
    let indices = adj.indices();
    let weights = adj.data();

    let mut num_moves = 0;
    for side_b in [false, true] {
        let labels: &[PartitionId] = communities;
        let moves: Vec<(usize, PartitionId)> = (0..n)
            .into_par_iter()
            .with_min_len(4096)
            .filter_map(|u| {
                if is_b[u] != side_b {
                    return None;
                }
                let own = labels[u];
                let range = indptr[u]..indptr[u + 1];
                if indices[range.clone()].iter().all(|&v| labels[v] == own) {
                    return None;
                }

                let mut neighbours: SmallVec<[(PartitionId, usize); 8]> = indices[range.clone()]
                    .iter()
                    .zip(&weights[range])
                    .map(|(&v, &weight)| (labels[v], weight as usize))
                    .collect();
                neighbours.sort_unstable_by_key(|&(community, _)| community);

                let mut own_weight = 0;
                let mut best_weight = 0;
                let mut best = own;
                let mut i = 0;
                while i < neighbours.len() {
                    let community = neighbours[i].0;
                    let mut weight = 0;
                    while i < neighbours.len() && neighbours[i].0 == community {
                        weight += neighbours[i].1;
                        i += 1;
                    }
                    if community == own {
                        own_weight = weight;
                    } else if weight > best_weight {
                        best_weight = weight;
                        best = community;
                    }
                }
                (best_weight > own_weight).then_some((u, best))
            })
            .collect();

        num_moves += moves.len();
        for (u, community) in moves {
            communities[u] = community;
        }
    }
    num_moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::edges_from_tuples;

    #[test]
    fn test_node_with_two_edges_out_and_one_in_moves() {
        // A-nodes: 0, 1. B-nodes: 2, 3, 4.
        // Node 0 is in community 0 with node 2, and has two edges to community 1 (3 and 4).
        let edges = edges_from_tuples(vec![(0, 2), (0, 3), (0, 4), (1, 3), (1, 4)]);
        let is_b = vec![false, false, true, true, true];
        let graph = Graph::<u8>::from_edges(edges, 5);
        let mut communities = vec![0, 1, 0, 1, 1];

        let num_moves = boundary_polish(&graph, &is_b, &mut communities);

        // Node 0 moves in the pass over side A. Node 2, on side B, then reads the new label of
        // its only neighbour and follows.
        assert_eq!(communities, vec![1, 1, 1, 1, 1]);
        assert_eq!(num_moves, 2);
    }

    #[test]
    fn test_node_with_as_many_edges_out_as_in_stays() {
        // Node 0 has two edges into its own community (2, 3) and two out (4, 5).
        let edges = edges_from_tuples(vec![
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 5),
            (6, 2),
            (6, 3),
        ]);
        let is_b = vec![false, false, true, true, true, true, false];
        let graph = Graph::<u8>::from_edges(edges, 7);
        let mut communities = vec![0, 1, 0, 0, 1, 1, 0];
        let before = communities.clone();

        let num_moves = boundary_polish(&graph, &is_b, &mut communities);

        assert_eq!(communities, before);
        assert_eq!(num_moves, 0);
    }

    #[test]
    fn test_node_moves_to_the_community_with_most_edges() {
        // Node 0 (community 0) has one edge in (1), two edges to community 1 (2, 3) and three
        // to community 2 (4, 5, 6). Nodes 7 and 8 keep the B-nodes in place.
        let edges = edges_from_tuples(vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (7, 2),
            (7, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (9, 1),
        ]);
        let is_b = vec![
            false, true, true, true, true, true, true, false, false, false,
        ];
        let graph = Graph::<u8>::from_edges(edges, 10);
        let mut communities = vec![0, 0, 1, 1, 2, 2, 2, 1, 2, 0];

        let num_moves = boundary_polish(&graph, &is_b, &mut communities);

        assert_eq!(communities, vec![2, 0, 1, 1, 2, 2, 2, 1, 2, 0]);
        assert_eq!(num_moves, 1);
    }
}
