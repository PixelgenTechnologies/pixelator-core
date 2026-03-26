use std::collections::hash_map::Entry;

use rustc_hash::FxHashMap as HashMap;

use crate::common::types::{Edge, EdgeWeight, NodeIdx, UMI, UMIPair};

pub struct UmiToNodeIndexMapping {
    node_idx_to_umi: Vec<NodeIdx>,
    umi_to_node_idx: HashMap<UMI, NodeIdx>,
}

impl UmiToNodeIndexMapping {
    pub fn from_umi_pairs(umi_pairs: &[UMIPair]) -> Self {
        // We can make a guess about the number of unique UMIs based on the
        // number of pairs to improve performance when allocating
        let unique_umi_ratio_guesss = umi_pairs.len() / 5;
        let mut node_idx_to_umi: Vec<NodeIdx> = Vec::with_capacity(unique_umi_ratio_guesss);
        let mut umi_to_node_idx: HashMap<UMI, NodeIdx> =
            HashMap::with_capacity_and_hasher(unique_umi_ratio_guesss, Default::default());

        for (src, dest) in umi_pairs.iter() {
            if let Entry::Vacant(e) = umi_to_node_idx.entry(*src) {
                e.insert(node_idx_to_umi.len());
                node_idx_to_umi.push(*src);
            }

            if let Entry::Vacant(e) = umi_to_node_idx.entry(*dest) {
                e.insert(node_idx_to_umi.len());
                node_idx_to_umi.push(*dest);
            }
        }

        Self {
            node_idx_to_umi,
            umi_to_node_idx,
        }
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
        let umis = vec![(11, 22), (22, 33), (33, 11)];
        let mapping = UmiToNodeIndexMapping::from_umi_pairs(&umis);
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
}
