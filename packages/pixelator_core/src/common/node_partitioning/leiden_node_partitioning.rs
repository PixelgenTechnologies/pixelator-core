use crate::common::node_partitioning::NodePartitioning;
use crate::common::types::{NodeIdx, PartitionId};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::cmp::max;

#[derive(Debug, Clone)]
/// Implementation of node partitioning optimized to be used in the Leiden algorithm.
pub struct LeidenNodePartitioning {
    node_partition_map: Vec<PartitionId>,
    partition_to_node_map: HashMap<PartitionId, HashSet<NodeIdx>>,
    partition_to_node_map_stale: bool,
    new_partition_id: PartitionId,
}

impl LeidenNodePartitioning {
    fn new(node_partition_map: Vec<PartitionId>) -> Self {
        let mut partition_to_node_map: HashMap<PartitionId, HashSet<NodeIdx>> = HashMap::default();
        for (node, &partition) in node_partition_map.iter().enumerate() {
            partition_to_node_map
                .entry(partition)
                .or_default()
                .insert(node);
        }
        let new_partition_id = node_partition_map.iter().max().unwrap_or(&0) + 1;
        LeidenNodePartitioning {
            node_partition_map,
            partition_to_node_map,
            partition_to_node_map_stale: false,
            new_partition_id,
        }
    }

    /// Return true of the partition only contains one node
    pub fn is_singlet_partition(&self, partition: &PartitionId) -> bool {
        self.panic_if_partition_to_node_map_stale();
        self.partition_to_node_map[partition].len() == 1
    }

    /// Rebuild `partition_to_node_map` from `node_partition_map` if the cache is stale.
    ///
    /// This is intended to be called once between expensive refinement/aggregation phases,
    /// not during the local moving loop.
    pub fn rebuild_partition_to_node_map_if_stale(&mut self) {
        if !self.partition_to_node_map_stale {
            return;
        }

        self.partition_to_node_map = HashMap::default();
        for (node, &partition) in self.node_partition_map.iter().enumerate() {
            self.partition_to_node_map
                .entry(partition)
                .or_default()
                .insert(node);
        }
        self.partition_to_node_map_stale = false;
    }

    /// Fast path for updating a node's partition during the local moving phase.
    ///
    /// This avoids maintaining `partition_to_node_map` incrementally; instead, it is marked
    /// stale and rebuilt once before it is needed again (e.g. refinement/aggregation).
    pub fn update_partition_fast(&mut self, node: NodeIdx, partition: PartitionId) {
        self.node_partition_map[node] = partition;
        self.new_partition_id = max(self.new_partition_id, partition + 1);
        self.partition_to_node_map_stale = true;
    }

    /// Return a reference to the map from partition to nodes
    pub fn get_partition_to_node_map_ref(&self) -> &HashMap<PartitionId, HashSet<NodeIdx>> {
        self.panic_if_partition_to_node_map_stale();
        &(self.partition_to_node_map)
    }

    /// Generates a new, unused partition id
    pub fn get_new_partition_id(&self) -> PartitionId {
        self.new_partition_id
    }

    fn panic_if_partition_to_node_map_stale(&self) {
        if self.partition_to_node_map_stale {
            panic!(
                "Make sure you call `rebuild_partition_to_node_map_if_stale` before accessing the partition_to_node_map.\
                 If you see this error there is a bug in the code."
            );
        }
    }
}

impl NodePartitioning for LeidenNodePartitioning {
    fn initialize_from_partitions(partitions: Vec<PartitionId>) -> Self {
        LeidenNodePartitioning::new(partitions)
    }

    fn get_node_to_partition_map(&self) -> &[PartitionId] {
        &(self.node_partition_map)
    }

    fn get_partition_to_node_map(&self) -> HashMap<PartitionId, HashSet<NodeIdx>> {
        self.panic_if_partition_to_node_map_stale();
        // NOTE: this overrides the default trait implementation
        self.partition_to_node_map.clone()
    }

    /// Update the partition of a given node.
    ///
    /// This update function will maintain the expensive `partition_to_node_map` cache,
    /// and is therefore slower than `update_partition_fast`. It
    fn update_partition(&mut self, node: NodeIdx, partition: PartitionId) {
        self.panic_if_partition_to_node_map_stale();

        self.partition_to_node_map
            .entry(self.node_partition_map[node])
            .and_modify(|e| {
                e.remove(&node);
            });

        self.node_partition_map[node] = partition;
        self.partition_to_node_map
            .entry(partition)
            .or_default()
            .insert(node);
        self.partition_to_node_map_stale = false;
        self.new_partition_id = max(self.new_partition_id, partition + 1);
    }

    fn normalize(&mut self) {
        self.panic_if_partition_to_node_map_stale();

        if self.is_normalized() {
            return;
        }

        let partition_map = self
            .node_partition_map
            .iter()
            .copied()
            .collect::<HashSet<PartitionId>>()
            .into_iter()
            .enumerate()
            .map(|(new_pid, old_pid)| (old_pid, new_pid))
            .collect::<HashMap<PartitionId, PartitionId>>();

        self.node_partition_map = self
            .node_partition_map
            .iter()
            .map(|old_pid| partition_map[old_pid])
            .collect();

        self.partition_to_node_map = self
            .partition_to_node_map
            .iter()
            .filter_map(|(id, nodes)| {
                if !nodes.is_empty() {
                    Some((partition_map[id], nodes.clone()))
                } else {
                    None
                }
            })
            .collect();

        self.new_partition_id = self.node_partition_map.len();
        self.partition_to_node_map_stale = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_partition() {
        let mut np = LeidenNodePartitioning::initialize_from_partitions(vec![3, 10, 3, 10, 10]);
        np.normalize();

        assert_eq!(np.node_partition_map.len(), 5);

        assert_eq!(np.node_partition_map[0], np.node_partition_map[2]);

        assert_eq!(np.node_partition_map[1], np.node_partition_map[3]);
        assert_eq!(np.node_partition_map[3], np.node_partition_map[4]);

        assert_eq!(
            np.node_partition_map
                .into_iter()
                .collect::<HashSet<usize>>(),
            vec![0, 1].into_iter().collect::<HashSet<usize>>()
        );
    }

    #[test]
    fn test_node_partitioning_initialization() {
        let np = LeidenNodePartitioning::initialize_with_singlet_partitions(4);
        assert_eq!(np.node_partition_map, vec![0, 1, 2, 3]);

        let np_null = LeidenNodePartitioning::initialize_with_null_partition(3);
        assert_eq!(np_null.node_partition_map, vec![0, 0, 0]);

        let custom = vec![2, 2, 3];
        let np_custom = LeidenNodePartitioning::initialize_from_partitions(custom.clone());
        assert_eq!(np_custom.node_partition_map, custom);
    }

    #[test]
    fn test_node_partitioning_assign_and_get() {
        let mut np = LeidenNodePartitioning::initialize_with_null_partition(2);
        np.update_partition(1, 5);
        assert_eq!(np.get_partition_for_node(&1), 5);
        assert_eq!(np.get_partition_for_node(&0), 0);
    }

    #[test]
    fn test_partitions_as_matrix_basic() {
        // Partition: [0, 0, 1, 1, 0, 1, 2]
        let partitions =
            LeidenNodePartitioning::initialize_from_partitions(vec![0, 0, 1, 1, 0, 1, 2]);
        let mat = partitions.partitions_as_matrix();

        // Should be 3 partitions (rows) and 7 nodes (cols)
        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 7);

        // Check that the matrix is binary and matches expected structure
        // Partition 0: nodes 0, 1, 4
        let part0 = mat.outer_view(0).unwrap();
        assert_eq!(part0.indices(), &[0, 1, 4]);
        assert_eq!(part0.data(), &[1.0, 1.0, 1.0]);

        // Partition 1: nodes 2, 3, 5
        let part1 = mat.outer_view(1).unwrap();
        assert_eq!(part1.indices(), &[2, 3, 5]);
        assert_eq!(part1.data(), &[1.0, 1.0, 1.0]);

        // Partition 2: node 6
        let part2 = mat.outer_view(2).unwrap();
        assert_eq!(part2.indices(), &[6]);
        assert_eq!(part2.data(), &[1.0]);
    }

    #[test]
    fn test_partitions_as_matrix_singletons() {
        // Each node in its own partition
        let partitions = LeidenNodePartitioning::initialize_with_singlet_partitions(4);
        let mat = partitions.partitions_as_matrix();

        assert_eq!(mat.rows(), 4);
        assert_eq!(mat.cols(), 4);
        for i in 0..4 {
            let row = mat.outer_view(i).unwrap();
            assert_eq!(row.indices(), &[i]);
            assert_eq!(row.data(), &[1.0]);
        }
    }

    #[test]
    fn test_partitions_as_matrix_all_in_one_partition() {
        // All nodes in one partition
        let partitions = LeidenNodePartitioning::initialize_with_null_partition(3);
        let mat = partitions.partitions_as_matrix();

        assert_eq!(mat.rows(), 1);
        assert_eq!(mat.cols(), 3);
        let row = mat.outer_view(0).unwrap();
        assert_eq!(row.indices(), &[0, 1, 2]);
        assert_eq!(row.data(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_partitions_as_matrix_null() {
        // All nodes in one partition
        let partitions = LeidenNodePartitioning::initialize_with_null_partition(3);
        let mat = partitions.partitions_as_matrix();

        assert_eq!(mat.rows(), 1);
        assert_eq!(mat.cols(), 3);
        let row = mat.outer_view(0).unwrap();
        assert_eq!(row.indices(), &[0, 1, 2]);
        assert_eq!(row.data(), &[1.0, 1.0, 1.0]);
    }
}
