use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::common::graph::Graph;
use crate::common::node_partitioning::NodePartitioning;
use crate::common::types::PartitionId;

/// Strategy for handling non-4-cycle nodes after merging all 4-cycles in a bipartite graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FourCycleBridgeMode {
    /// Highest precision (`3b`): non-4-cycle nodes only join a 4-cycle core if at least 2 of
    /// their neighbors belong to that same 4-cycle core (`best_count >= 2`), or if the node
    /// has degree 1. Folds virtually zero crossing edges (`~0.005%`).
    StrictMajorityOnly,
    /// Same-core bridges and ears (`3c`): propagates 4-cycle core labels along 2-core bridges
    /// and 6-cycle/8-cycle ears that connect to a single 4-cycle core, while withdrawing to a
    /// singleton whenever two different 4-cycle cores meet at a tie.
    MergeSameCoreBridges,
}

#[inline]
fn atomic_find(parents: &[AtomicUsize], mut x: usize) -> usize {
    loop {
        let p = parents[x].load(Ordering::Relaxed);
        if p == x {
            return x;
        }
        let gp = parents[p].load(Ordering::Relaxed);
        if gp != p {
            let _ = parents[x].compare_exchange_weak(p, gp, Ordering::Relaxed, Ordering::Relaxed);
        }
        x = gp;
    }
}

#[inline]
fn atomic_union(parents: &[AtomicUsize], mut u: usize, mut v: usize) {
    loop {
        u = atomic_find(parents, u);
        v = atomic_find(parents, v);
        if u == v {
            return;
        }
        let (high, low) = if u > v { (u, v) } else { (v, u) };
        if parents[high]
            .compare_exchange_weak(high, low, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            return;
        }
    }
}

#[inline]
fn mode_strict_majority_or_leaf(
    numbers: &mut [usize],
    preferred: usize,
    retain_preferred: bool,
) -> usize {
    if numbers.len() == 1 {
        return numbers[0];
    }
    numbers.sort_unstable();
    let mut best_count = 0usize;
    let mut best_n = numbers[0];
    let mut i = 0;
    let len = numbers.len();
    while i < len {
        let val = numbers[i];
        let mut j = i + 1;
        while j < len && numbers[j] == val {
            j += 1;
        }
        let count = j - i;
        if count > best_count || (retain_preferred && count == best_count && val == preferred) {
            best_count = count;
            best_n = val;
        }
        i = j;
    }
    if best_count >= 2 { best_n } else { preferred }
}

/// Parallel 4-Cycle Compression for Bipartite Graphs.
///
/// 1. Discovers all 4-cycles `(u - v1 - w - v2 - u)` using Chiba-Nishizeki degree orientation
///    (`rank(u) > rank(v)` and `rank(u) > rank(w)`) and merges their vertices via lock-free Union-Find.
/// 2. Coarsens non-4-cycle nodes according to `bridge_mode`:
///    - [`FourCycleBridgeMode::StrictMajorityOnly`]: only merges non-4-cycle nodes that have
///      at least 2 neighbors in the same 4-cycle core (or degree 1).
///    - [`FourCycleBridgeMode::MergeSameCoreBridges`]: merges same-core ears and bridges into
///      their parent 4-cycle core while withdrawing to a singleton whenever two different
///      4-cycle cores meet at a tie.
pub fn four_cycle_compression<P>(
    graph: &Graph<u8>,
    is_b: &[bool],
    bridge_mode: FourCycleBridgeMode,
    max_rounds: usize,
) -> P
where
    P: NodePartitioning,
{
    let n = graph.get_num_nodes();
    assert_eq!(
        is_b.len(),
        n,
        "Bipartite side mask length ({}) must match number of graph nodes ({})",
        is_b.len(),
        n
    );

    let mut labels: Vec<PartitionId> = (0..n).collect();
    let adj = graph.get_adjacency_matrix();
    let indptr_raw = adj.indptr();
    let indptr = indptr_raw
        .as_slice()
        .expect("CSR adjacency matrix indptr must be contiguous");
    let indices = adj.indices();
    let degrees = graph.get_degrees();

    {
        let parents: &[AtomicUsize] =
            unsafe { std::slice::from_raw_parts(labels.as_mut_ptr() as *const AtomicUsize, n) };

        // Step 1: Parallel Chiba-Nishizeki degree-oriented 4-cycle discovery & Union-Find
        (0..n).into_par_iter().with_min_len(1024).for_each(|u| {
            let deg_u = degrees[u];
            if deg_u < 2 {
                return;
            }
            let rank_u = (deg_u, u);
            let start_u = indptr[u];
            let end_u = indptr[u + 1];

            let mut low_neighbors = SmallVec::<[usize; 16]>::new();
            for &v in &indices[start_u..end_u] {
                let deg_v = degrees[v];
                if deg_v >= 2 && (deg_v, v) < rank_u {
                    low_neighbors.push(v);
                }
            }
            if low_neighbors.len() < 2 {
                return;
            }

            let mut wedges = SmallVec::<[(usize, usize); 64]>::new();
            for &v in &low_neighbors {
                for &w in &indices[indptr[v]..indptr[v + 1]] {
                    let deg_w = degrees[w];
                    if deg_w >= 2 && (deg_w, w) < rank_u {
                        wedges.push((w, v));
                    }
                }
            }

            if wedges.len() < 2 {
                return;
            }

            wedges.sort_unstable_by_key(|&(w, _)| w);
            let mut i = 0;
            while i < wedges.len() {
                let w = wedges[i].0;
                let mut j = i + 1;
                while j < wedges.len() && wedges[j].0 == w {
                    j += 1;
                }
                if j - i >= 2 {
                    atomic_union(parents, u, w);
                    for k in i..j {
                        atomic_union(parents, u, wedges[k].1);
                    }
                }
                i = j;
            }
        });

        // Flatten Union-Find trees and mark 4-cycle core membership in a single parallel pass
        let in_four_cycle = vec![false; n];
        {
            let in_fc_atomic: &[AtomicBool] = unsafe {
                std::slice::from_raw_parts(in_four_cycle.as_ptr() as *const AtomicBool, n)
            };
            (0..n).into_par_iter().with_min_len(4096).for_each(|u| {
                let r = atomic_find(parents, u);
                parents[u].store(r, Ordering::Relaxed);
                if r != u {
                    in_fc_atomic[u].store(true, Ordering::Relaxed);
                    in_fc_atomic[r].store(true, Ordering::Relaxed);
                }
            });
        }

        // Step 2: Handle non-4-cycle nodes according to `bridge_mode`
        match bridge_mode {
            FourCycleBridgeMode::StrictMajorityOnly => {
                for round in 0..max_rounds {
                    let b_chg: usize = (0..n)
                        .into_par_iter()
                        .with_min_len(4096)
                        .map(|b| {
                            if !is_b[b] || in_four_cycle[b] {
                                return 0usize;
                            }
                            let start = indptr[b];
                            let end = indptr[b + 1];
                            if start == end {
                                return 0usize;
                            }
                            let cur = parents[b].load(Ordering::Relaxed);
                            let mut parts = SmallVec::<[usize; 8]>::new();
                            for &a in &indices[start..end] {
                                parts.push(parents[a].load(Ordering::Relaxed));
                            }
                            let chosen =
                                mode_strict_majority_or_leaf(parts.as_mut_slice(), cur, round > 0);
                            if chosen != cur {
                                parents[b].store(chosen, Ordering::Relaxed);
                                1usize
                            } else {
                                0usize
                            }
                        })
                        .sum();

                    let a_chg: usize = (0..n)
                        .into_par_iter()
                        .with_min_len(4096)
                        .map(|a| {
                            if is_b[a] || in_four_cycle[a] {
                                return 0usize;
                            }
                            let start = indptr[a];
                            let end = indptr[a + 1];
                            if start == end {
                                return 0usize;
                            }
                            let cur = parents[a].load(Ordering::Relaxed);
                            let mut parts = SmallVec::<[usize; 8]>::new();
                            for &b in &indices[start..end] {
                                parts.push(parents[b].load(Ordering::Relaxed));
                            }
                            let chosen =
                                mode_strict_majority_or_leaf(parts.as_mut_slice(), cur, round > 0);
                            if chosen != cur {
                                parents[a].store(chosen, Ordering::Relaxed);
                                1usize
                            } else {
                                0usize
                            }
                        })
                        .sum();

                    if (b_chg + a_chg) * 100 <= n {
                        break;
                    }
                }
            }
            FourCycleBridgeMode::MergeSameCoreBridges => {
                const UNSET: usize = usize::MAX;
                (0..n).into_par_iter().with_min_len(4096).for_each(|u| {
                    if !in_four_cycle[u] {
                        parents[u].store(UNSET, Ordering::Relaxed);
                    }
                });

                for _round in 0..max_rounds {
                    for target_is_b in [true, false] {
                        (0..n).into_par_iter().with_min_len(4096).for_each(|u| {
                            if is_b[u] != target_is_b || in_four_cycle[u] {
                                return;
                            }
                            let mut parts = SmallVec::<[usize; 8]>::new();
                            for &nb in &indices[indptr[u]..indptr[u + 1]] {
                                let l_nb = parents[nb].load(Ordering::Relaxed);
                                if l_nb != UNSET {
                                    parts.push(l_nb);
                                }
                            }
                            if parts.is_empty() {
                                parents[u].store(UNSET, Ordering::Relaxed);
                                return;
                            }
                            parts.sort_unstable();
                            if parts[0] == parts[parts.len() - 1] {
                                parents[u].store(parts[0], Ordering::Relaxed);
                                return;
                            }
                            let mut best_count = 0usize;
                            let mut second_best = 0usize;
                            let mut best_c = UNSET;
                            let mut i = 0;
                            while i < parts.len() {
                                let val = parts[i];
                                let mut j = i + 1;
                                while j < parts.len() && parts[j] == val {
                                    j += 1;
                                }
                                let c = j - i;
                                if c > best_count {
                                    second_best = best_count;
                                    best_count = c;
                                    best_c = val;
                                } else if c > second_best {
                                    second_best = c;
                                }
                                i = j;
                            }
                            if best_count > second_best {
                                parents[u].store(best_c, Ordering::Relaxed);
                            } else {
                                parents[u].store(UNSET, Ordering::Relaxed);
                            }
                        });
                    }
                }

                (0..n).into_par_iter().with_min_len(4096).for_each(|u| {
                    if parents[u].load(Ordering::Relaxed) == UNSET {
                        parents[u].store(u, Ordering::Relaxed);
                    }
                });
            }
        }

        // Mark all nodes belonging to multi-node partitions (`p != u`) in `in_four_cycle`
        // so singleton partitions (`!in_four_cycle[start]`) can skip CSR neighbor scans in O(1).
        let in_fc_atomic: &[AtomicBool] =
            unsafe { std::slice::from_raw_parts(in_four_cycle.as_ptr() as *const AtomicBool, n) };
        (0..n).into_par_iter().with_min_len(4096).for_each(|u| {
            let p = parents[u].load(Ordering::Relaxed);
            if p != u {
                in_fc_atomic[u].store(true, Ordering::Relaxed);
                in_fc_atomic[p].store(true, Ordering::Relaxed);
            }
        });

        const UNVISITED: usize = usize::MAX;
        let mut comp_labels = vec![UNVISITED; n];
        let mut queue = std::collections::VecDeque::new();
        let mut component_id = 0usize;

        for start in 0..n {
            if !in_four_cycle[start] {
                comp_labels[start] = component_id;
                component_id += 1;
                continue;
            }
            if comp_labels[start] != UNVISITED {
                continue;
            }
            let id = component_id;
            component_id += 1;
            let target_label = labels[start];
            comp_labels[start] = id;
            for &neighbor in &indices[indptr[start]..indptr[start + 1]] {
                if comp_labels[neighbor] == UNVISITED && labels[neighbor] == target_label {
                    comp_labels[neighbor] = id;
                    queue.push_back(neighbor);
                }
            }
            while let Some(node) = queue.pop_front() {
                for &neighbor in &indices[indptr[node]..indptr[node + 1]] {
                    if comp_labels[neighbor] == UNVISITED && labels[neighbor] == target_label {
                        comp_labels[neighbor] = id;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        drop(in_four_cycle);
        drop(labels);
        P::initialize_from_partitions(comp_labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::graph::Graph;
    use crate::common::node_partitioning::{FastNodePartitioning, NodePartitioning};
    use crate::common::types::edges_from_tuples;

    #[test]
    fn test_four_cycle_compression_strict_and_same_core_bridges() {
        // Construct two 4-cycle cores (Cell 1 and Cell 2), a same-core 6-cycle ear on Cell 1,
        // and a cross-cell bridge node between Cell 1 and Cell 2:
        // A-nodes: 0, 1, 2 (Cell 1), 3, 4 (Cell 2)
        // B-nodes: 5, 6, 7, 8 (Cell 1), 9, 10 (Cell 2), 11 (Cross-cell bridge)
        // Cell 1 4-cycle: 0-5-1-6-0
        // Cell 1 6-cycle ear anchored to {0, 1}: 0-7-2-8-1
        // Cell 2 4-cycle: 3-9-4-10-3
        // Cross-cell bridge B-node 11 connected to 1 (Cell 1) and 3 (Cell 2)
        let edges = edges_from_tuples(vec![
            // Cell 1 4-cycle
            (0, 5),
            (0, 6),
            (1, 5),
            (1, 6),
            // Cell 1 6-cycle ear: 0 - 7 - 2 - 8 - 1
            (0, 7),
            (2, 7),
            (2, 8),
            (1, 8),
            // Cell 2 4-cycle
            (3, 9),
            (3, 10),
            (4, 9),
            (4, 10),
            // Cross-cell bridge: 1 - 11 - 3
            (1, 11),
            (3, 11),
        ]);
        let is_b = vec![
            false, false, false, false, false, // 0..=4 in A
            true, true, true, true, true, true, true, // 5..=11 in B
        ];
        let graph = Graph::<u8>::from_edges(edges, 12).with_bipartite_side_mask(is_b.clone());

        // 1. StrictMajorityOnly (3b):
        // - Merges 4-cycle {0, 1, 5, 6} and 4-cycle {3, 4, 9, 10}
        // - Keeps cross-cell bridge 11 AND 6-cycle ear {2, 7, 8} unmerged from the 4-cycle cores
        let p_strict: FastNodePartitioning =
            four_cycle_compression(&graph, &is_b, FourCycleBridgeMode::StrictMajorityOnly, 4);
        let m_strict = p_strict.get_node_to_partition_map();
        assert_eq!(m_strict[0], m_strict[1]);
        assert_eq!(m_strict[0], m_strict[5]);
        assert_eq!(m_strict[0], m_strict[6]);
        assert_eq!(m_strict[3], m_strict[4]);
        assert_eq!(m_strict[3], m_strict[9]);
        assert_eq!(m_strict[3], m_strict[10]);
        assert_ne!(m_strict[0], m_strict[3]);
        assert_ne!(m_strict[11], m_strict[0]);
        assert_ne!(m_strict[11], m_strict[3]);

        // 2. MergeSameCoreBridges (3c):
        // - Merges 4-cycle {0, 1, 5, 6} AND the same-core 6-cycle ear {2, 7, 8} into the same Cell 1 partition!
        // - Still keeps the cross-cell bridge node 11 separate from both Cell 1 and Cell 2!
        let p_same_core: FastNodePartitioning =
            four_cycle_compression(&graph, &is_b, FourCycleBridgeMode::MergeSameCoreBridges, 4);
        let m_same = p_same_core.get_node_to_partition_map();
        assert_eq!(m_same[0], m_same[1]);
        assert_eq!(m_same[0], m_same[5]);
        assert_eq!(m_same[0], m_same[6]);
        assert_eq!(m_same[0], m_same[2]);
        assert_eq!(m_same[0], m_same[7]);
        assert_eq!(m_same[0], m_same[8]);
        assert_eq!(m_same[3], m_same[4]);
        assert_eq!(m_same[3], m_same[9]);
        assert_eq!(m_same[3], m_same[10]);
        assert_ne!(m_same[0], m_same[3]);
        assert_ne!(m_same[11], m_same[0]);
        assert_ne!(m_same[11], m_same[3]);
    }
}
