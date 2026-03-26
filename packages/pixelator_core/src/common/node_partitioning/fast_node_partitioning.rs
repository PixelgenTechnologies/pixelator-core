use crate::common::node_partitioning::NodePartitioning;
use crate::common::types::{NodeIdx, PartitionId};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

#[derive(Debug, Clone)]
/// A fast and lightweight implementation of node partitioning.
///
/// This implementation is optimized to be used with the fast label
/// propagation algorithm.
pub struct FastNodePartitioning {
    pub node_partition_map: Vec<PartitionId>,
}

impl FastNodePartitioning {
    fn new(node_partition_map: Vec<PartitionId>) -> Self {
        FastNodePartitioning { node_partition_map }
    }
}

impl NodePartitioning for FastNodePartitioning {
    fn initialize_from_partitions(partitions: Vec<PartitionId>) -> Self {
        FastNodePartitioning::new(partitions)
    }

    fn get_node_to_partition_map(&self) -> &[PartitionId] {
        &(self.node_partition_map)
    }

    fn update_partition(&mut self, node: NodeIdx, partition: PartitionId) {
        self.node_partition_map[node] = partition;
    }

    fn normalize(&mut self) {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::node_partitioning::LeidenNodePartitioning;

    #[test]
    fn test_fastnodepartitioning_normalize_partition() {
        let mut np = FastNodePartitioning::initialize_from_partitions(vec![3, 10, 3, 10, 10]);
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
    fn test_fastnodepartitioning_initialization() {
        let np = FastNodePartitioning::initialize_with_singlet_partitions(4);
        assert_eq!(np.node_partition_map, vec![0, 1, 2, 3]);

        let np_null = FastNodePartitioning::initialize_with_null_partition(3);
        assert_eq!(np_null.node_partition_map, vec![0, 0, 0]);

        let custom = vec![2, 2, 3];
        let np_custom = FastNodePartitioning::initialize_from_partitions(custom.clone());
        assert_eq!(np_custom.node_partition_map, custom);
    }

    #[test]
    fn test_fastnodepartitioning_assign_and_get() {
        let mut np = FastNodePartitioning::initialize_with_null_partition(2);
        np.update_partition(1, 5);
        assert_eq!(np.get_partition_for_node(&1), 5);
        assert_eq!(np.get_partition_for_node(&0), 0);
    }

    #[test]
    fn test_fastnodepartitioning_partitions_as_matrix_basic() {
        // Partition: [0, 0, 1, 1, 0, 1, 2]
        let partitions =
            FastNodePartitioning::initialize_from_partitions(vec![0, 0, 1, 1, 0, 1, 2]);
        let mat = partitions.partitions_as_matrix();

        // Should be 3 partitions (rows) and 7 nodes (cols)
        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 7);

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
    fn test_fastnodepartitioning_partitions_as_matrix_singletons() {
        // Each node in its own partition
        let partitions = FastNodePartitioning::initialize_with_singlet_partitions(4);
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
    fn test_fastnodepartitioning_partitions_as_matrix_all_in_one_partition() {
        // All nodes in one partition
        let partitions = FastNodePartitioning::initialize_with_null_partition(3);
        let mat = partitions.partitions_as_matrix();

        assert_eq!(mat.rows(), 1);
        assert_eq!(mat.cols(), 3);
        let row = mat.outer_view(0).unwrap();
        assert_eq!(row.indices(), &[0, 1, 2]);
        assert_eq!(row.data(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_fastnodepartitioning_partitions_as_matrix_null() {
        // All nodes in one partition
        let partitions = FastNodePartitioning::initialize_with_null_partition(3);
        let mat = partitions.partitions_as_matrix();

        assert_eq!(mat.rows(), 1);
        assert_eq!(mat.cols(), 3);
        let row = mat.outer_view(0).unwrap();
        assert_eq!(row.indices(), &[0, 1, 2]);
        assert_eq!(row.data(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_fastnodepartitioning_into_leiden_partitioning() {
        let partitions = FastNodePartitioning::initialize_with_null_partition(3);
        let partitions = partitions.into_node_partitioning::<LeidenNodePartitioning>();
        assert_eq!(partitions.get_node_to_partition_map(), &[0, 0, 0]);
    }
}
