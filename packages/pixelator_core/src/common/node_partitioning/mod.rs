use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use sprs::{CsMat, TriMat};

use crate::common::types::{NodeIdx, PartitionId};

mod leiden_node_partitioning;
pub use self::leiden_node_partitioning::LeidenNodePartitioning;

mod fast_node_partitioning;
pub use self::fast_node_partitioning::FastNodePartitioning;

pub trait NodePartitioning {
    /// Initialize partitions from a vector mapping node ids to partition ids
    fn initialize_from_partitions(partitions: Vec<PartitionId>) -> Self;
    /// Returns a reference to the vector mapping node ids to partition ids
    fn get_node_to_partition_map(&self) -> &[PartitionId];
    /// Update the partition of a specific node
    fn update_partition(&mut self, node: NodeIdx, partition: PartitionId);

    /// Normalizes partition ids to use all integers from 0 to |P| - 1
    fn normalize(&mut self);
    /// Put all nodes in their own individual partitions
    fn initialize_with_singlet_partitions(num_nodes: NodeIdx) -> Self
    where
        Self: Sized,
    {
        Self::initialize_from_partitions((0..num_nodes).collect::<Vec<PartitionId>>())
    }

    /// Put all nodes in the same unique partition
    fn initialize_with_null_partition(num_nodes: NodeIdx) -> Self
    where
        Self: Sized,
    {
        Self::initialize_from_partitions(vec![0; num_nodes])
    }

    /// Returns the total number of nodes
    fn num_nodes(&self) -> usize {
        self.get_node_to_partition_map().len()
    }

    /// Returns the partition id of a given node
    fn get_partition_for_node(&self, node: &NodeIdx) -> PartitionId {
        self.get_node_to_partition_map()[*node]
    }

    /// Generate a map from partition id to node ids
    fn get_partition_to_node_map(&self) -> HashMap<PartitionId, HashSet<NodeIdx>> {
        let mut partition_map: HashMap<PartitionId, HashSet<NodeIdx>> = HashMap::default();
        for (node, &partition) in self.get_node_to_partition_map().iter().enumerate() {
            partition_map.entry(partition).or_default().insert(node);
        }
        partition_map
    }

    /// Return the total number of partitions
    fn num_partitions(&self) -> usize {
        let unique_partitions = self
            .get_node_to_partition_map()
            .iter()
            .collect::<HashSet<_>>();
        unique_partitions.len()
    }

    /// Returns true if all partitions ids from 0 to |P| - 1 are used
    fn is_normalized(&self) -> bool {
        self.get_node_to_partition_map()
            .iter()
            .max()
            .copied()
            .unwrap()
            + 1
            == self.num_partitions()
    }

    /// Converts current NodePartitioning into another NodePartitioning implementation.
    ///
    /// For instance, this is useful to convert a FastNodePartitioning instance into a
    /// LeidenNodePartitioning instance during the transition from FLP to Leiden.
    fn into_node_partitioning<T: NodePartitioning>(self) -> T
    where
        Self: Sized,
    {
        T::initialize_from_partitions(self.get_node_to_partition_map().to_vec())
    }

    /// Generates the |P| x |N| sparse matrix matching nodes to partitions
    fn partitions_as_matrix(&self) -> CsMat<f64> {
        let num_nodes = self.num_nodes();
        let num_partitions = self.num_partitions();
        let mut tri_mat = TriMat::new((num_partitions, num_nodes));

        for (node, &partition_id) in self.get_node_to_partition_map().iter().enumerate() {
            tri_mat.add_triplet(partition_id, node, 1.0);
        }

        tri_mat.to_csr()
    }

    /// Returns the largest k partitions
    fn top_k_largest_partitions(&self, k: usize) -> Vec<HashSet<NodeIdx>> {
        let mut partitions = self
            .get_partition_to_node_map()
            .into_values()
            .collect::<Vec<HashSet<NodeIdx>>>();
        partitions.sort_by_key(|p| -(p.len() as isize));
        partitions.into_iter().take(k).collect()
    }

    /// Returns the size of the largest partition
    fn largest_partition_size(&self) -> usize {
        self.get_partition_to_node_map()
            .into_values()
            .map(|p| p.len())
            .max()
            .unwrap_or(0)
    }
}
