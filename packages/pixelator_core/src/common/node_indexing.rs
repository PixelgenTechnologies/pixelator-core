use std::hash::{BuildHasherDefault, Hasher};

use rustc_hash::FxHasher;

use crate::common::types::{Edge, EdgeWeight, NodeIdx, UMI, UMIPair};

/// Hasher for UMI keys: FxHash followed by the murmur3 `fmix64` finalizer.
///
/// FxHash only spreads keys well for tables up to 2^26 buckets; for larger tables part of the
/// bucket index comes straight from the key's lowest bits, so biased UMIs pile up in part of the
/// table and probe sequences get very long. The finalizer lets every UMI bit affect every hash bit.
#[derive(Default)]
struct UmiHasher(FxHasher);

impl Hasher for UmiHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0.write_u64(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0.write_usize(i);
    }

    #[inline]
    fn finish(&self) -> u64 {
        let mut h = self.0.finish();
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^ (h >> 33)
    }
}

type UmiBuildHasher = BuildHasherDefault<UmiHasher>;

// The std map is only used with `UmiBuildHasher`, never with the default SipHash.
#[allow(clippy::disallowed_types)]
type UmiHashMap<V> = std::collections::HashMap<UMI, V, UmiBuildHasher>;

pub struct UmiToNodeIndexMapping {
    node_idx_to_umi: Vec<NodeIdx>,
    umi_to_node_idx: UmiHashMap<NodeIdx>,
}

impl UmiToNodeIndexMapping {
    /// Reserve room for about `node_capacity` nodes.
    pub fn with_capacity(node_capacity: usize) -> Self {
        Self {
            node_idx_to_umi: Vec::with_capacity(node_capacity),
            umi_to_node_idx: UmiHashMap::with_capacity_and_hasher(
                node_capacity,
                Default::default(),
            ),
        }
    }

    /// Return the node index for `umi`, assigning the next index if it is new.
    pub fn get_or_insert(&mut self, umi: UMI) -> NodeIdx {
        if let Some(&idx) = self.umi_to_node_idx.get(&umi) {
            return idx;
        }
        let idx = self.node_idx_to_umi.len();
        self.umi_to_node_idx.insert(umi, idx);
        self.node_idx_to_umi.push(umi);
        idx
    }

    pub fn from_umi_pairs<I>(umi_pairs: I) -> Self
    where
        I: IntoIterator<Item = UMIPair>,
    {
        let umi_pairs = umi_pairs.into_iter();
        // Guess the number of unique UMIs from the number of pairs.
        let unique_umi_ratio_guess = umi_pairs.size_hint().0 / 5;
        let mut mapping = Self::with_capacity(unique_umi_ratio_guess);

        for (src, dest) in umi_pairs {
            mapping.get_or_insert(src);
            mapping.get_or_insert(dest);
        }

        mapping
    }

    pub fn map_node_index_to_umi(&self, node_index: NodeIdx) -> UMI {
        self.node_idx_to_umi[node_index]
    }

    pub fn map_umi_to_node_index(&self, umi: UMI) -> NodeIdx {
        self.umi_to_node_idx[&umi]
    }

    pub fn map_umi_pair_iterator_to_edge<I, T>(&self, edges: I) -> impl Iterator<Item = Edge<T>>
    where
        I: IntoIterator<Item = UMIPair>,
        T: EdgeWeight,
    {
        edges.into_iter().map(|e| self.map_umi_pair_to_edge(&e))
    }

    pub fn map_umi_pair_to_edge<T>(&self, umi_pair: &UMIPair) -> Edge<T>
    where
        T: EdgeWeight,
    {
        let (src, dest) = umi_pair;
        Edge {
            src: self.umi_to_node_idx[src],
            dest: self.umi_to_node_idx[dest],
            weight: T::one(),
        }
    }

    pub fn map_edge_from_idx_to_umi<T>(&self, edge: &Edge<T>) -> UMIPair
    where
        T: EdgeWeight,
    {
        (
            self.node_idx_to_umi[edge.src],
            self.node_idx_to_umi[edge.dest],
        )
    }

    pub fn get_num_of_nodes(&self) -> usize {
        self.node_idx_to_umi.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_umi_to_node_index_mapping_roundtrip() {
        // Create dummy umi pairs
        let umis = [(11, 22), (22, 33), (33, 11)];
        let mapping = UmiToNodeIndexMapping::from_umi_pairs(umis.iter().copied());
        // Check number of nodes
        assert_eq!(mapping.get_num_of_nodes(), 3);

        assert_eq!(
            mapping.map_umi_pair_to_edge(&umis[0]),
            Edge {
                src: 0,
                dest: 1,
                weight: 1usize
            }
        );
        assert_eq!(
            mapping.map_umi_pair_to_edge(&umis[1]),
            Edge {
                src: 1,
                dest: 2,
                weight: 2usize
            }
        );
        assert_eq!(
            mapping.map_umi_pair_to_edge(&umis[2]),
            Edge {
                src: 2,
                dest: 0,
                weight: 3usize
            }
        );

        assert_eq!(
            mapping.map_edge_from_idx_to_umi(&Edge {
                src: 0,
                dest: 1,
                weight: 1usize
            }),
            umis[0]
        );
        assert_eq!(
            mapping.map_edge_from_idx_to_umi(&Edge {
                src: 1,
                dest: 2,
                weight: 2usize
            }),
            umis[1]
        );
        assert_eq!(
            mapping.map_edge_from_idx_to_umi(&Edge {
                src: 2,
                dest: 0,
                weight: 3usize
            }),
            umis[2]
        );
    }

    #[test]
    fn test_umi_hash_spreads_high_bucket_bits_for_biased_low_bits() {
        use std::hash::BuildHasher;

        // Every key has the same lowest 4 bits (the same first two nucleotides). For a table
        // with 2^30 buckets, bucket-index bits 26..30 must still take every value.
        let hasher = UmiBuildHasher::default();
        let high_bucket_bits: rustc_hash::FxHashSet<u64> = (0..10_000usize)
            .map(|i| (hasher.hash_one((i << 4) | 0b0101) >> 26) & 0xF)
            .collect();

        assert_eq!(high_bucket_bits.len(), 16);
    }

    #[test]
    fn test_incremental_insert_matches_from_edges() {
        use crate::common::graph::Graph;

        // Includes a duplicate pair and the same pair reversed, which must sum.
        let pairs = [
            (11usize, 22),
            (22, 33),
            (33, 11),
            (11, 22),
            (50, 11),
            (22, 11),
        ];
        let two_pass = UmiToNodeIndexMapping::from_umi_pairs(pairs.iter().copied());
        let mut one_pass = UmiToNodeIndexMapping::with_capacity(pairs.len());
        let triplets: Vec<(NodeIdx, NodeIdx, u8)> = pairs
            .iter()
            .copied()
            .map(|(src, dest)| {
                let (src, dest) = (one_pass.get_or_insert(src), one_pass.get_or_insert(dest));
                (src.min(dest), src.max(dest), 1)
            })
            .collect();

        assert_eq!(two_pass.get_num_of_nodes(), one_pass.get_num_of_nodes());
        for &(src, dest) in &pairs {
            assert_eq!(
                two_pass.map_umi_to_node_index(src),
                one_pass.map_umi_to_node_index(src)
            );
            assert_eq!(
                two_pass.map_umi_to_node_index(dest),
                one_pass.map_umi_to_node_index(dest)
            );
        }

        let graph_two = Graph::<u8>::from_edges(
            two_pass.map_umi_pair_iterator_to_edge(pairs.iter().copied()),
            two_pass.get_num_of_nodes(),
        );
        let graph_one =
            Graph::<u8>::from_undirected_triplets(triplets, one_pass.get_num_of_nodes());

        assert_eq!(graph_one.get_num_nodes(), graph_two.get_num_nodes());
        assert_eq!(
            graph_one.get_edge_entry_count(),
            graph_two.get_edge_entry_count()
        );
        assert_eq!(
            graph_one.get_total_edge_weight(),
            graph_two.get_total_edge_weight()
        );
        for node in 0..graph_one.get_num_nodes() {
            let mut from_one: Vec<_> = graph_one
                .edges_from_iter(node)
                .map(|edge| (edge.dest, edge.weight))
                .collect();
            let mut from_two: Vec<_> = graph_two
                .edges_from_iter(node)
                .map(|edge| (edge.dest, edge.weight))
                .collect();
            from_one.sort_unstable();
            from_two.sort_unstable();
            assert_eq!(from_one, from_two);
        }
    }
}
