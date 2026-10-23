use std::alloc::{GlobalAlloc, Layout};
use std::cmp::Ordering;
use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

use itertools::Itertools;
use mimalloc::MiMalloc;
use rand::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use smallvec::SmallVec;

use pixelator_core::common::constants::MIN_PNA_COMPONENT_SIZE;
use pixelator_core::common::graph::Graph;
use pixelator_core::common::io::{
    ParquetUMIPairIter, create_graph_and_umi_mapping_from_parquet_file,
};
use pixelator_core::common::node_indexing::UmiToNodeIndexMapping;
use pixelator_core::common::node_partitioning::{
    FastNodePartitioning, LeidenNodePartitioning, NodePartitioning,
};
use pixelator_core::common::types::{NodeIdx, PartitionId};
use pixelator_core::common::utils::{NodeQueue, mode_retain_smallest, mode_smallest};
use pixelator_core::fast_label_propagation::algorithm::{
    DEFAULT_BIPARTITE_FLP_MAX_ROUNDS, DEFAULT_BIPARTITE_FLP_MIN_CHANGED_FRACTION,
    alternating_bipartite_flp, clean_partitions,
};
use pixelator_core::leiden::algorithm::leiden;
use pixelator_core::leiden::quality::modularity::Modularity;
use pixelator_core::leiden::weighted_partitioned_graph::{
    AggregateOptions, WeightedPartitionedGraph,
};

struct TrackingAllocator;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { MiMalloc.alloc(layout) };
        if !ptr.is_null() {
            let size = layout.size();
            let prev = LIVE_BYTES.fetch_add(size, AtomicOrdering::Relaxed);
            PEAK_BYTES.fetch_max(prev + size, AtomicOrdering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), AtomicOrdering::Relaxed);
        unsafe { MiMalloc.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator;

fn reset_peak_memory() -> usize {
    let cur = LIVE_BYTES.load(AtomicOrdering::Relaxed);
    PEAK_BYTES.store(cur, AtomicOrdering::Relaxed);
    cur
}

fn peak_delta_mb(base_live: usize) -> f64 {
    let peak = PEAK_BYTES.load(AtomicOrdering::Relaxed);
    peak.saturating_sub(base_live) as f64 / (1024.0 * 1024.0)
}

/// Load a Parquet edge list and optionally shuffle the `(umi1, umi2)` edge rows BEFORE
/// inserting into `UmiToNodeIndexMapping`, simulating realistic unsorted sequencing data
/// where node indices `0..N` are not sorted by ground-truth cell!
fn load_graph_from_parquet_with_edge_order(
    parquet_file: &str,
    shuffle_edges_seed: Option<u64>,
) -> (UmiToNodeIndexMapping, Graph<u8>) {
    if shuffle_edges_seed.is_none() {
        return create_graph_and_umi_mapping_from_parquet_file::<u8>(parquet_file);
    }

    let pairs_iter =
        ParquetUMIPairIter::new(parquet_file).expect("Failed to create ParquetUMIPairIter");
    let mut raw_pairs: Vec<_> = pairs_iter.collect();
    let mut rng = StdRng::seed_from_u64(shuffle_edges_seed.unwrap());
    raw_pairs.shuffle(&mut rng);

    let num_pairs = raw_pairs.len();
    let est_nodes = num_pairs.saturating_mul(2) / 3;
    let mut umi_mapping = UmiToNodeIndexMapping::with_capacity(est_nodes);
    let mut is_b = Vec::with_capacity(est_nodes);

    let triplets: Vec<(NodeIdx, NodeIdx, u8)> = raw_pairs
        .into_iter()
        .map(|(src, dest)| {
            let u = umi_mapping.get_or_insert(src);
            if u == is_b.len() {
                is_b.push(false);
            }
            let v = umi_mapping.get_or_insert(dest);
            if v == is_b.len() {
                is_b.push(true);
            }
            (u.min(v), u.max(v), 1u8)
        })
        .collect();

    let num_nodes = umi_mapping.get_num_of_nodes();
    let graph =
        Graph::<u8>::from_undirected_triplets(triplets, num_nodes).with_bipartite_side_mask(is_b);
    (umi_mapping, graph)
}

/// Legacy random tie-breaking mode (matching `origin/dev` before deterministic tie-breaking).
fn mode_random(numbers: &mut [usize], rng: &mut impl Rng) -> usize {
    numbers.sort_unstable();
    let ((_, best_n), _) = numbers.iter().dedup_with_count().fold(
        ((0usize, 0usize), 0usize),
        |((best_count, best_n), ties), (c, &n)| match c.cmp(&best_count) {
            Ordering::Less => ((best_count, best_n), ties),
            Ordering::Greater => ((c, n), 1),
            Ordering::Equal => {
                let ties = ties + 1;
                if rng.random_bool(1.0 / ties as f64) {
                    ((c, n), ties)
                } else {
                    ((best_count, best_n), ties)
                }
            }
        },
    );
    best_n
}

/// PR #33 mode: retain `preferred` if tied for max count; otherwise break ties randomly.
fn mode_retain_random(numbers: &mut [usize], preferred: usize, rng: &mut impl Rng) -> usize {
    numbers.sort_unstable();
    let ((_, best_n), _) = numbers.iter().dedup_with_count().fold(
        ((0usize, 0usize), 0usize),
        |((best_count, best_n), ties), (c, &n)| match c.cmp(&best_count) {
            Ordering::Less => ((best_count, best_n), ties),
            Ordering::Greater => ((c, n), 1),
            Ordering::Equal => {
                if best_n == preferred {
                    ((best_count, best_n), ties)
                } else if n == preferred {
                    ((c, n), 1)
                } else {
                    let ties = ties + 1;
                    if rng.random_bool(1.0 / ties as f64) {
                        ((c, n), ties)
                    } else {
                        ((best_count, best_n), ties)
                    }
                }
            }
        },
    );
    best_n
}

/// Mode that requires `best_count >= 2` when node has degree >= 2 (leaves of degree 1 still join
/// their sole neighbor). Ties among modes with count >= 2 are broken by `mode_retain_smallest`
/// (or `mode_smallest` when `retain_preferred` is false).
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

#[derive(Clone, Copy)]
enum BipartiteModification {
    /// 2a. Round 0 (A -> B): B picks highest-degree A-neighbor `max_by_key((deg[a], usize::MAX - a))`
    ///     Round 0 (B -> A): `mode_smallest`, Round 1+: `mode_retain_smallest`
    HubAttractorR0,
    /// 2b. Hub Attractor in Round 0 (A -> B) + Strict Majority (`count >= 2` for deg >= 2) on A in Round 0 (B -> A),
    ///     followed by standard `mode_retain_smallest` in Round 1+
    HubAttractorR0StrictMajR0,
    /// 2c. Standard `indices[start]` in Round 0 (A -> B) + Strict Majority (`count >= 2` for deg >= 2) on A in Round 0 (B -> A),
    ///     followed by standard `mode_retain_smallest` in Round 1+
    SmallestR0StrictMajR0,
    /// 2d. Hub Attractor in Round 0 (A -> B) + Strict Majority (`count >= 2` for deg >= 2) on A in ALL rounds
    HubAttractorR0StrictMajAllRounds,
    /// 2e. Mutual 2-hop / 4-cycle support in Round 0 (A -> B) + mode_smallest in R0
    FourCycleSupportedR0,
    /// 2f. Mutual 2-hop / 4-cycle support in Round 0 (A -> B) + Strict Majority on A in Round 0 (B -> A),
    ///     followed by standard `mode_retain_smallest` in Round 1+
    FourCycleSupportedR0StrictMajR0,
}

/// Lock-free parallel Disjoint-Set (Union-Find) over `AtomicUsize` parent pointers (`parent[x] <= x`).
#[inline]
fn atomic_find(parents: &[AtomicUsize], mut x: usize) -> usize {
    loop {
        let p = parents[x].load(AtomicOrdering::Relaxed);
        if p == x {
            return x;
        }
        let gp = parents[p].load(AtomicOrdering::Relaxed);
        if gp != p {
            let _ = parents[x].compare_exchange_weak(
                p,
                gp,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            );
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
            .compare_exchange_weak(high, low, AtomicOrdering::Relaxed, AtomicOrdering::Relaxed)
            .is_ok()
        {
            return;
        }
    }
}

/// Approach 3: Find all 4-cycles `(a1 - b1 - a2 - b2 - a1)` in the bipartite graph and merge
/// all nodes participating in each 4-cycle (plus optionally contract degree-1 leaf nodes into
/// their sole neighbor).
///
/// How 4-cycle detection works on CSR:
/// For each A-node `a1`, we inspect its B-neighbors `b in N(a1)`, and for each `b` we look at
/// `a2 in N(b)` with `a2 > a1` (so each pair `(a1, a2)` is only visited in one direction).
/// If a pair `(a1, a2)` appears via two or more distinct B-neighbors `b1, b2`, then
/// `(a1, b1, a2, b2)` is a 4-cycle! We union `(a1, a2, b1, b2)` in a lock-free Union-Find.
#[derive(Clone, Copy)]
enum FourCycleMode {
    /// 3a. Merge all 4-cycles (Union-Find) + 3 rounds of Bipartite FLP on non-4-cycle tree tendrils
    UnionFind4CyclesPlusTreeFlp,
    /// 3b. Highest Precision: Merge all 4-cycles (Union-Find) + Strict-Majority Bipartite FLP
    ///     (nodes only join a 4-cycle core if >= 2 of their neighbors belong to that same core)
    UnionFind4CyclesPlusStrictTreeFlp,
    /// 3c. Same-Core Ear / Bridge Merging: Merge all 4-cycles (Union-Find), then propagate
    ///     "core reachability" across non-4-cycle nodes: any node/ear/bridge whose paths to
    ///     4-cycle cores all lead to the SAME 4-cycle core `C` is merged into `C`, while any
    ///     bridge node that can reach TWO OR MORE DIFFERENT 4-cycle cores (`C1 != C2`) is
    ///     marked `AMBIGUOUS` and left as a singleton for Leiden to cut!
    UnionFind4CyclesPlusSameCoreEars,
    /// 3d. Local 4-Cycle Only R0 + StrictMaj Bipartite FLP
    Local4CycleSeededBipartiteFlp,
}

fn run_four_cycle_merge(
    graph: &Graph<u8>,
    is_b: &[bool],
    mode: FourCycleMode,
) -> (FastNodePartitioning, f64, f64, f64) {
    let n = graph.get_num_nodes();
    let base_mem = reset_peak_memory();
    let t0 = Instant::now();

    let mut labels: Vec<PartitionId> = (0..n).collect();
    let adj = graph.get_adjacency_matrix();
    let indptr_raw = adj.indptr();
    let indptr = indptr_raw.as_slice().unwrap();
    let indices = adj.indices();
    let degrees = graph.get_degrees();

    if matches!(mode, FourCycleMode::Local4CycleSeededBipartiteFlp) {
        // Local 4-Cycle Seeded Bipartite FLP:
        // Step 1: Find every edge `(u, v)` that participates in at least one 4-cycle!
        // Specifically, for each B-node `b`, if an A-neighbor `a` shares at least one 4-cycle with `b`,
        // we mark `(b, a)` as a 4-cycle edge.
        // In Round 0 (A -> B), `b` ONLY considers A-neighbors connected to `b` by a 4-cycle edge
        // (unless `b` has degree 1, in which case it joins its sole A-neighbor; if `b` has degree >= 2
        // and ZERO 4-cycle edges, `b` stays a singleton in Round 0!).
        let atomic_labels: &[AtomicUsize] =
            unsafe { std::slice::from_raw_parts(labels.as_mut_ptr() as *const AtomicUsize, n) };

        for round in 0..DEFAULT_BIPARTITE_FLP_MAX_ROUNDS {
            let b_chg: usize = (0..n)
                .into_par_iter()
                .with_min_len(4096)
                .map(|b| {
                    if !is_b[b] {
                        return 0usize;
                    }
                    let start = indptr[b];
                    let end = indptr[b + 1];
                    if start == end {
                        return 0usize;
                    }
                    let cur = atomic_labels[b].load(AtomicOrdering::Relaxed);
                    let chosen = if round == 0 {
                        let b_neighbors = &indices[start..end];
                        if b_neighbors.len() == 1 {
                            b_neighbors[0]
                        } else {
                            // Find A-neighbor with most 4-cycles with `b`
                            let mut best_a = cur;
                            let mut best_key = (0usize, 0usize, 0usize);
                            for &a in b_neighbors {
                                let mut fc = 0usize;
                                for &bp in &indices[indptr[a]..indptr[a + 1]] {
                                    if bp == b {
                                        continue;
                                    }
                                    let bp_nb = &indices[indptr[bp]..indptr[bp + 1]];
                                    for &a2 in b_neighbors {
                                        if a2 != a && bp_nb.binary_search(&a2).is_ok() {
                                            fc += 1;
                                        }
                                    }
                                }
                                if fc > 0 {
                                    let key = (fc, degrees[a], usize::MAX - a);
                                    if key > best_key {
                                        best_key = key;
                                        best_a = a;
                                    }
                                }
                            }
                            best_a
                        }
                    } else {
                        let mut parts = SmallVec::<[usize; 8]>::new();
                        for &a in &indices[start..end] {
                            parts.push(atomic_labels[a].load(AtomicOrdering::Relaxed));
                        }
                        mode_strict_majority_or_leaf(parts.as_mut_slice(), cur, true)
                    };
                    if chosen != cur {
                        atomic_labels[b].store(chosen, AtomicOrdering::Relaxed);
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
                    if is_b[a] {
                        return 0usize;
                    }
                    let start = indptr[a];
                    let end = indptr[a + 1];
                    if start == end {
                        return 0usize;
                    }
                    let mut parts = SmallVec::<[usize; 8]>::new();
                    for &b in &indices[start..end] {
                        parts.push(atomic_labels[b].load(AtomicOrdering::Relaxed));
                    }
                    let cur = atomic_labels[a].load(AtomicOrdering::Relaxed);
                    let chosen = mode_strict_majority_or_leaf(parts.as_mut_slice(), cur, round > 0);
                    if chosen != cur {
                        atomic_labels[a].store(chosen, AtomicOrdering::Relaxed);
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

        let loop_mem = peak_delta_mb(base_mem);
        let cleaned = clean_partitions(
            graph,
            FastNodePartitioning::initialize_from_partitions(labels),
        );
        let total_sec = t0.elapsed().as_secs_f64();
        let total_mem = peak_delta_mb(base_mem);
        return (cleaned, total_sec, loop_mem, total_mem);
    }

    {
        let parents: &[AtomicUsize] =
            unsafe { std::slice::from_raw_parts(labels.as_mut_ptr() as *const AtomicUsize, n) };

        (0..n).into_par_iter().with_min_len(1024).for_each(|u| {
            let deg_u = degrees[u];
            if deg_u < 2 {
                return;
            }
            let rank_u = (deg_u, u);
            let start_u = indptr[u];
            let end_u = indptr[u + 1];

            let mut wedges = SmallVec::<[(usize, usize); 64]>::new();
            for &v in &indices[start_u..end_u] {
                if (degrees[v], v) >= rank_u {
                    continue;
                }
                for &w in &indices[indptr[v]..indptr[v + 1]] {
                    if (degrees[w], w) < rank_u {
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

        (0..n).into_par_iter().with_min_len(4096).for_each(|u| {
            let r = atomic_find(parents, u);
            parents[u].store(r, AtomicOrdering::Relaxed);
        });

        let mut in_four_cycle = vec![false; n];
        for u in 0..n {
            let r = parents[u].load(AtomicOrdering::Relaxed);
            if r != u {
                in_four_cycle[u] = true;
                in_four_cycle[r] = true;
            }
        }

        // Diagnose why 850k nodes are NOT in 4-cycles:
        // How many distinct 4-cycle cores exist? And how many non-4-cycle nodes exist?
        if matches!(mode, FourCycleMode::UnionFind4CyclesPlusSameCoreEars) {
            let num_in_fc = in_four_cycle.iter().filter(|&&b| b).count();
            let mut core_sizes: HashMap<usize, usize> = HashMap::default();
            for u in 0..n {
                if in_four_cycle[u] {
                    *core_sizes
                        .entry(parents[u].load(AtomicOrdering::Relaxed))
                        .or_insert(0) += 1;
                }
            }
            let big_cores = core_sizes.values().filter(|&&sz| sz >= 100).count();
            eprintln!(
                "  [4-cycle core stats] in_4cycle={}/{} ({:.2}%), total_4cycle_clusters={}, big_cores(>=100)={}",
                num_in_fc,
                n,
                100.0 * num_in_fc as f64 / n as f64,
                core_sizes.len(),
                big_cores
            );
        }

        if matches!(mode, FourCycleMode::UnionFind4CyclesPlusSameCoreEars) {
            // Look at the 4-cycle statistics:
            // Out of 3,102,130 nodes, 1,341,514 (43.24%) are in 4-cycles, forming:
            //   - ~298 large 4-cycle cores (`size >= 100`), plus
            //   - ~51,200 tiny "4-cycle islands" (`size 4..20`) in the periphery of the cell,
            //     connected to the main 4-cycle core of the SAME cell by degree-2 paths/bridges!
            //
            // If we run Bipartite FLP (`3a`, where non-4-cycle nodes adopt their neighbor's label
            // via `mode_smallest` / `mode_retain_smallest`), why did `3a` fold `21,570` crossing edges,
            // whereas `3b` folded only `10`?
            // Because in `3a`, a non-4-cycle node `u` that sits on a bridge between Cell X and Cell Y
            // (where one neighbor has a label from Cell X and another neighbor has a label from Cell Y)
            // broke the 1-vs-1 tie by picking `min(label_X, label_Y)`, which glued `u` into one of the
            // cells across the crossing edge!
            //
            // What if we do the exact complement:
            // 1) Start with the 4-cycle Union-Find clusters (`parents[u]`), AND let non-4-cycle nodes
            //    propagate ONLY labels that originated from a 4-cycle (`in_four_cycle`)!
            // 2) Whenever a non-4-cycle node `u` sees that ALL of its 4-cycle-labeled neighbors have the
            //    **SAME** label `c` (even if only 1 neighbor is currently labeled!), `u` Tentatively adopts `c`!
            //    WAIT: if `u` adopts `c` when only 1 neighbor is labeled, what happens if in the NEXT half-step
            //    another neighbor `v` of `u` becomes labeled with a DIFFERENT 4-cycle core `c2 != c`?
            //    Then `u` now sees TWO DIFFERENT 4-cycle cores (`c` and `c2`) among its neighbors!
            //    So in each half-step, we re-evaluate **ALL** non-4-cycle nodes (`!in_four_cycle[u]`):
            //    - If a non-4-cycle node `u` sees that all of its labeled neighbors agree on the SAME
            //      4-cycle core `c` (or `c` has a strict majority `count >= 2` over any other core), `u`
            //      takes label `c`!
            //    - If a non-4-cycle node `u` sees a TIE between two or more DIFFERENT 4-cycle cores
            //      (e.g. 1 neighbor in `Core_X` and 1 neighbor in `Core_Y`!), `u` **WITHDRAWS / UN-MERGES**
            //      back to `UNSET` (singleton)!
            //
            // Think about what that "Tentative Propagation + Tie-Withdrawal" rule does:
            // - Along any same-core ear or bridge `Core_X - u1 - u2 - u3 - Core_X`:
            //   `u1` sees `Core_X` from the left; `u3` sees `Core_X` from the right; when they meet at `u2`,
            //   both sides carry the **SAME** label `Core_X` (`Core_X == Core_X`)! There is NO tie, so the
            //   entire ear/bridge `u1, u2, u3` stays merged into `Core_X`!
            // - Along any dead-end internal branch `Core_X - u1 - u2` (inside the 2-core, e.g. leading to a
            //   6-cycle or 8-cycle on `Core_X`):
            //   `u1` and `u2` only ever see `Core_X`, so they all merge into `Core_X`!
            // - Along a true crossing bridge `Core_X - u_X - v_Y - Core_Y` between two different cells:
            //   In Round 1, `u_X` tentatively takes `Core_X` and `v_Y` tentatively takes `Core_Y`.
            //   In Round 2, `u_X` looks at its neighbors and sees `{Core_X, Core_Y}` (1 vs 1 tie!), and `v_Y`
            //   looks at its neighbors and sees `{Core_X, Core_Y}` (1 vs 1 tie!) — so BOTH `u_X` and `v_Y`
            //   immediately **withdraw** to `UNSET` (or keep `u_X != v_Y`) and NEVER fold the crossing edge `(u_X, v_Y)`!
            const UNSET: usize = usize::MAX;
            (0..n).into_par_iter().with_min_len(4096).for_each(|u| {
                if !in_four_cycle[u] {
                    parents[u].store(UNSET, AtomicOrdering::Relaxed);
                }
            });

            // Alternating Bipartite half-steps (A -> B, then B -> A) directly in `parents` (ZERO extra Vec allocations!):
            for _round in 0..4 {
                for target_is_b in [true, false] {
                    (0..n).into_par_iter().with_min_len(4096).for_each(|u| {
                        if is_b[u] != target_is_b || in_four_cycle[u] {
                            return;
                        }
                        let mut parts = SmallVec::<[usize; 8]>::new();
                        for &nb in &indices[indptr[u]..indptr[u + 1]] {
                            let l_nb = parents[nb].load(AtomicOrdering::Relaxed);
                            if l_nb != UNSET {
                                parts.push(l_nb);
                            }
                        }
                        if parts.is_empty() {
                            parents[u].store(UNSET, AtomicOrdering::Relaxed);
                            return;
                        }
                        parts.sort_unstable();
                        if parts[0] == parts[parts.len() - 1] {
                            // 100% of labeled neighbors agree on the SAME 4-cycle core `parts[0]`!
                            parents[u].store(parts[0], AtomicOrdering::Relaxed);
                            return;
                        }
                        // Multiple 4-cycle cores meet at `u`: only join if one core has a STRICT MAJORITY
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
                            parents[u].store(best_c, AtomicOrdering::Relaxed);
                        } else {
                            // Tied between two different 4-cycle cores -> withdraw to UNSET!
                            parents[u].store(UNSET, AtomicOrdering::Relaxed);
                        }
                    });
                }
            }

            (0..n).into_par_iter().with_min_len(4096).for_each(|u| {
                if parents[u].load(AtomicOrdering::Relaxed) == UNSET {
                    parents[u].store(u, AtomicOrdering::Relaxed);
                }
            });
        } else {
            let strict_tree = matches!(mode, FourCycleMode::UnionFind4CyclesPlusStrictTreeFlp);

            for round in 0..4 {
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
                        let cur = parents[b].load(AtomicOrdering::Relaxed);
                        let mut parts = SmallVec::<[usize; 8]>::new();
                        for &a in &indices[start..end] {
                            parts.push(parents[a].load(AtomicOrdering::Relaxed));
                        }
                        let chosen = if strict_tree {
                            mode_strict_majority_or_leaf(parts.as_mut_slice(), cur, round > 0)
                        } else if round == 0 {
                            mode_smallest(parts.as_mut_slice())
                        } else {
                            mode_retain_smallest(parts.as_mut_slice(), cur)
                        };
                        if chosen != cur {
                            parents[b].store(chosen, AtomicOrdering::Relaxed);
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
                        let cur = parents[a].load(AtomicOrdering::Relaxed);
                        let mut parts = SmallVec::<[usize; 8]>::new();
                        for &b in &indices[start..end] {
                            parts.push(parents[b].load(AtomicOrdering::Relaxed));
                        }
                        let chosen = if strict_tree {
                            mode_strict_majority_or_leaf(parts.as_mut_slice(), cur, round > 0)
                        } else if round == 0 {
                            mode_smallest(parts.as_mut_slice())
                        } else {
                            mode_retain_smallest(parts.as_mut_slice(), cur)
                        };
                        if chosen != cur {
                            parents[a].store(chosen, AtomicOrdering::Relaxed);
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
    }

    let loop_mem = peak_delta_mb(base_mem);
    let cleaned = clean_partitions(
        graph,
        FastNodePartitioning::initialize_from_partitions(labels),
    );
    let total_sec = t0.elapsed().as_secs_f64();
    let total_mem = peak_delta_mb(base_mem);
    (cleaned, total_sec, loop_mem, total_mem)
}

fn run_bipartite_modification(
    graph: &Graph<u8>,
    is_b: &[bool],
    modification: BipartiteModification,
    min_changed_fraction: f64,
    max_rounds: usize,
) -> (FastNodePartitioning, f64, f64) {
    let n = graph.get_num_nodes();
    let base_mem = reset_peak_memory();
    let t0 = Instant::now();

    let threshold_nodes = (n as f64 * min_changed_fraction.clamp(0.0, 1.0)) as usize;
    let mut labels: Vec<PartitionId> = (0..n).collect();

    let adj = graph.get_adjacency_matrix();
    let indptr_raw = adj.indptr();
    let indptr = indptr_raw.as_slice().unwrap();
    let indices = adj.indices();
    let degrees = graph.get_degrees();

    {
        let atomic_labels: &[AtomicUsize] =
            unsafe { std::slice::from_raw_parts(labels.as_mut_ptr() as *const AtomicUsize, n) };

        for round in 0..max_rounds {
            // Half-step 1 (A -> B)
            let b_changed: usize = atomic_labels
                .par_iter()
                .enumerate()
                .with_min_len(4096)
                .map(|(b, label_b)| {
                    if !is_b[b] {
                        return 0usize;
                    }
                    let start = indptr[b];
                    let end = indptr[b + 1];
                    if start == end {
                        return 0usize;
                    }
                    let cur = label_b.load(AtomicOrdering::Relaxed);
                    let chosen = match (modification, round) {
                        (BipartiteModification::SmallestR0StrictMajR0, 0) => indices[start],
                        (
                            BipartiteModification::HubAttractorR0
                            | BipartiteModification::HubAttractorR0StrictMajR0
                            | BipartiteModification::HubAttractorR0StrictMajAllRounds,
                            0,
                        ) => indices[start..end]
                            .iter()
                            .copied()
                            .max_by_key(|&a| (degrees[a], usize::MAX - a))
                            .unwrap(),
                        (
                            BipartiteModification::FourCycleSupportedR0
                            | BipartiteModification::FourCycleSupportedR0StrictMajR0,
                            0,
                        ) => {
                            let b_neighbors = &indices[start..end];
                            if b_neighbors.len() == 1 {
                                b_neighbors[0]
                            } else {
                                // Fast two-pointer intersection count over sorted CSR slices
                                b_neighbors
                                    .iter()
                                    .copied()
                                    .max_by_key(|&a| {
                                        let mut four_cycles = 0usize;
                                        for &b_prime in &indices[indptr[a]..indptr[a + 1]] {
                                            if b_prime == b {
                                                continue;
                                            }
                                            let bp_neighbors =
                                                &indices[indptr[b_prime]..indptr[b_prime + 1]];
                                            for &a2 in b_neighbors {
                                                if a2 != a
                                                    && bp_neighbors.binary_search(&a2).is_ok()
                                                {
                                                    four_cycles += 1;
                                                }
                                            }
                                        }
                                        (four_cycles, degrees[a], usize::MAX - a)
                                    })
                                    .unwrap()
                            }
                        }
                        _ => {
                            let mut parts = SmallVec::<[usize; 8]>::new();
                            for &a in &indices[start..end] {
                                parts.push(atomic_labels[a].load(AtomicOrdering::Relaxed));
                            }
                            mode_retain_smallest(parts.as_mut_slice(), cur)
                        }
                    };

                    if chosen != cur {
                        label_b.store(chosen, AtomicOrdering::Relaxed);
                        1usize
                    } else {
                        0usize
                    }
                })
                .sum();

            // Half-step 2 (B -> A)
            let a_changed: usize = atomic_labels
                .par_iter()
                .enumerate()
                .with_min_len(4096)
                .map(|(a, label_a)| {
                    if is_b[a] {
                        return 0usize;
                    }
                    let start = indptr[a];
                    let end = indptr[a + 1];
                    if start == end {
                        return 0usize;
                    }
                    let mut parts = SmallVec::<[usize; 8]>::new();
                    for &b in &indices[start..end] {
                        parts.push(atomic_labels[b].load(AtomicOrdering::Relaxed));
                    }
                    let cur = label_a.load(AtomicOrdering::Relaxed);
                    let chosen = match (modification, round) {
                        (
                            BipartiteModification::HubAttractorR0
                            | BipartiteModification::FourCycleSupportedR0,
                            0,
                        ) => mode_smallest(parts.as_mut_slice()),
                        (
                            BipartiteModification::HubAttractorR0StrictMajR0
                            | BipartiteModification::SmallestR0StrictMajR0
                            | BipartiteModification::FourCycleSupportedR0StrictMajR0,
                            0,
                        ) => mode_strict_majority_or_leaf(parts.as_mut_slice(), cur, false),
                        (BipartiteModification::HubAttractorR0StrictMajAllRounds, r) => {
                            mode_strict_majority_or_leaf(parts.as_mut_slice(), cur, r > 0)
                        }
                        _ => mode_retain_smallest(parts.as_mut_slice(), cur),
                    };

                    if chosen != cur {
                        label_a.store(chosen, AtomicOrdering::Relaxed);
                        1usize
                    } else {
                        0usize
                    }
                })
                .sum();

            if b_changed + a_changed <= threshold_nodes {
                break;
            }
        }
    }

    let cleaned = clean_partitions(
        graph,
        FastNodePartitioning::initialize_from_partitions(labels),
    );
    let sec = t0.elapsed().as_secs_f64();
    let mem = peak_delta_mb(base_mem);
    (cleaned, sec, mem)
}

#[derive(Clone, Copy)]
enum QueueTieVariant {
    BaselineRandom,
    RetainRandom,
    SmallestOnly,
    RetainSmallest,
}

fn run_queue_flp_variant(
    graph: &Graph<u8>,
    variant: QueueTieVariant,
    seed: u64,
) -> (FastNodePartitioning, usize, f64, f64, f64, f64) {
    let n = graph.get_num_nodes();
    let base_mem = reset_peak_memory();

    let mut node_partition = FastNodePartitioning::initialize_with_singlet_partitions(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut nodes: Vec<NodeIdx> = (0..n).collect();
    nodes.shuffle(&mut rng);

    let t0 = Instant::now();
    let mut node_queue = NodeQueue::from(nodes);
    let mut iterations = 0usize;
    let adj = graph.get_adjacency_matrix();

    while let Some(node_i) = node_queue.pop_front() {
        iterations += 1;
        let neighbors_view = adj.outer_view(node_i).unwrap();
        if neighbors_view.nnz() == 0 {
            continue;
        }
        let neighbors = neighbors_view.indices();
        let mut partitions = SmallVec::<[usize; 8]>::new();
        partitions.extend(
            neighbors
                .iter()
                .map(|x| node_partition.get_partition_for_node(x)),
        );

        let cur = node_partition.get_partition_for_node(&node_i);
        let chosen = match variant {
            QueueTieVariant::BaselineRandom => mode_random(partitions.as_mut_slice(), &mut rng),
            QueueTieVariant::RetainRandom => {
                mode_retain_random(partitions.as_mut_slice(), cur, &mut rng)
            }
            QueueTieVariant::SmallestOnly => mode_smallest(partitions.as_mut_slice()),
            QueueTieVariant::RetainSmallest => mode_retain_smallest(partitions.as_mut_slice(), cur),
        };

        if chosen != cur {
            node_partition.update_partition(node_i, chosen);
            for &nb in neighbors {
                if node_partition.get_partition_for_node(&nb) != chosen {
                    node_queue.push_back(nb);
                }
            }
        }
    }
    let flp_loop_sec = t0.elapsed().as_secs_f64();
    let flp_loop_mem_mb = peak_delta_mb(base_mem);
    drop(node_queue);

    let t_clean = Instant::now();
    let cleaned = clean_partitions(graph, node_partition);
    let clean_sec = t_clean.elapsed().as_secs_f64();
    let flp_total_mem_mb = peak_delta_mb(base_mem);
    (
        cleaned,
        iterations,
        flp_loop_sec,
        clean_sec,
        flp_loop_mem_mb,
        flp_total_mem_mb,
    )
}

/// Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) between two labelings.
fn compute_ari_and_nmi(a: &[usize], b: &[usize]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let nf = n as f64;

    let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::default();
    let mut a_counts: HashMap<usize, usize> = HashMap::default();
    let mut b_counts: HashMap<usize, usize> = HashMap::default();

    for (&la, &lb) in a.iter().zip(b.iter()) {
        *pair_counts.entry((la, lb)).or_insert(0) += 1;
        *a_counts.entry(la).or_insert(0) += 1;
        *b_counts.entry(lb).or_insert(0) += 1;
    }

    let choose2 = |k: usize| -> f64 {
        let kf = k as f64;
        kf * (kf - 1.0) * 0.5
    };

    let sum_nij: f64 = pair_counts.values().map(|&c| choose2(c)).sum();
    let sum_ai: f64 = a_counts.values().map(|&c| choose2(c)).sum();
    let sum_bj: f64 = b_counts.values().map(|&c| choose2(c)).sum();
    let total_pairs = choose2(n);
    let expected = (sum_ai * sum_bj) / total_pairs;
    let max_idx = 0.5 * (sum_ai + sum_bj);
    let ari = if (max_idx - expected).abs() < 1e-12 {
        1.0
    } else {
        (sum_nij - expected) / (max_idx - expected)
    };

    let mut mi = 0.0f64;
    for (&(la, lb), &nij) in &pair_counts {
        if nij > 0 {
            let p_ij = nij as f64 / nf;
            let p_i = a_counts[&la] as f64 / nf;
            let p_j = b_counts[&lb] as f64 / nf;
            mi += p_ij * (p_ij / (p_i * p_j)).ln();
        }
    }
    let h_a: f64 = a_counts
        .values()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / nf;
            -p * p.ln()
        })
        .sum();
    let h_b: f64 = b_counts
        .values()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / nf;
            -p * p.ln()
        })
        .sum();
    let nmi = if h_a + h_b < 1e-12 {
        1.0
    } else {
        2.0 * mi / (h_a + h_b)
    };

    (ari, nmi)
}

struct GroundTruthData {
    gt_cell_of_1pc_node: Vec<usize>,
    true_crossing_edges: Vec<(NodeIdx, NodeIdx)>,
    true_intra_edges: Vec<(NodeIdx, NodeIdx)>,
    gt_cells_ge_8k: usize,
    gt_total_components: usize,
}

fn build_ground_truth(
    path_0pc: &str,
    mapping_1pc: &UmiToNodeIndexMapping,
    graph_1pc: &Graph<u8>,
) -> GroundTruthData {
    let (mapping_0pc, graph_0pc) = create_graph_and_umi_mapping_from_parquet_file::<u8>(path_0pc);
    let comp_0pc = graph_0pc.connected_components();

    let mut comp_sizes: HashMap<usize, usize> = HashMap::default();
    for &c in &comp_0pc {
        *comp_sizes.entry(c).or_insert(0) += 1;
    }
    let gt_total_components = comp_sizes.len();
    let gt_cells_ge_8k = comp_sizes
        .values()
        .filter(|&&sz| sz >= MIN_PNA_COMPONENT_SIZE)
        .count();

    let n_1pc = graph_1pc.get_num_nodes();
    let mut gt_cell_of_1pc_node = Vec::with_capacity(n_1pc);
    for u_1pc in 0..n_1pc {
        let umi = mapping_1pc.map_node_index_to_umi(u_1pc);
        let u_0pc = mapping_0pc.map_umi_to_node_index(umi);
        gt_cell_of_1pc_node.push(comp_0pc[u_0pc]);
    }

    let adj = graph_1pc.get_adjacency_matrix();
    let indptr_raw = adj.indptr();
    let indptr = indptr_raw.as_slice().unwrap();
    let indices = adj.indices();

    let mut true_crossing_edges = Vec::new();
    let mut true_intra_edges = Vec::new();
    for u in 0..n_1pc {
        let cu = gt_cell_of_1pc_node[u];
        for &v in &indices[indptr[u]..indptr[u + 1]] {
            if u < v {
                if cu != gt_cell_of_1pc_node[v] {
                    true_crossing_edges.push((u, v));
                } else {
                    true_intra_edges.push((u, v));
                }
            }
        }
    }

    GroundTruthData {
        gt_cell_of_1pc_node,
        true_crossing_edges,
        true_intra_edges,
        gt_cells_ge_8k,
        gt_total_components,
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_variant(
    name: &str,
    graph_1pc: &Graph<u8>,
    gt: &GroundTruthData,
    flp_partition: FastNodePartitioning,
    flp_sec: f64,
    clean_sec: f64,
    flp_loop_mem_mb: f64,
    flp_total_mem_mb: f64,
    baseline_final_components: Option<&[usize]>,
) -> Vec<usize> {
    let flp_map = flp_partition.get_node_to_partition_map();
    let flp_num_partitions = flp_partition.num_partitions();

    let folded_crossing_edges = gt
        .true_crossing_edges
        .iter()
        .filter(|&&(u, v)| flp_map[u] == flp_map[v])
        .count();
    let total_true_crossing = gt.true_crossing_edges.len();
    let folded_crossing_pct = 100.0 * folded_crossing_edges as f64 / total_true_crossing as f64;

    let flp_coarsened_intra = gt
        .true_intra_edges
        .iter()
        .filter(|&&(u, v)| flp_map[u] == flp_map[v])
        .count();
    let flp_coarsened_intra_pct =
        100.0 * flp_coarsened_intra as f64 / gt.true_intra_edges.len() as f64;

    let base_leiden_mem = reset_peak_memory();
    let t_leiden = Instant::now();
    let quality_function = Modularity::new(1.0, graph_1pc.get_total_edge_weight());
    let graph_usize = Graph::<usize>::from(graph_1pc.clone());
    let partition: LeidenNodePartitioning = NodePartitioning::into_node_partitioning(flp_partition);

    let mut wp_graph =
        WeightedPartitionedGraph::new(graph_usize, partition, quality_function, None, Some(0));
    wp_graph.aggregate(AggregateOptions::All);
    leiden(&mut wp_graph, 0.1, None, None);
    let post_leiden_stats = wp_graph.get_statistics();
    let leiden_sec = t_leiden.elapsed().as_secs_f64();
    let leiden_mem_mb = peak_delta_mb(base_leiden_mem);

    let final_partition_map = wp_graph.get_ancestor_to_partition_map();
    let final_components =
        graph_1pc.connected_components_by(|u, v| final_partition_map[u] == final_partition_map[v]);

    let mut comp_sizes: HashMap<usize, usize> = HashMap::default();
    for &c in &final_components {
        *comp_sizes.entry(c).or_insert(0) += 1;
    }
    let total_final_components = comp_sizes.len();
    let cells_ge_8k = comp_sizes
        .values()
        .filter(|&&sz| sz >= MIN_PNA_COMPONENT_SIZE)
        .count();
    let stranded_nodes: usize = comp_sizes
        .values()
        .filter(|&&sz| sz < MIN_PNA_COMPONENT_SIZE)
        .sum();

    let leaked_crossing_edges = gt
        .true_crossing_edges
        .iter()
        .filter(|&&(u, v)| final_components[u] == final_components[v])
        .count();
    let removed_crossing_edges = total_true_crossing - leaked_crossing_edges;
    let removed_crossing_pct = 100.0 * removed_crossing_edges as f64 / total_true_crossing as f64;

    let preserved_intra_edges = gt
        .true_intra_edges
        .iter()
        .filter(|&&(u, v)| final_components[u] == final_components[v])
        .count();
    let preserved_intra_pct =
        100.0 * preserved_intra_edges as f64 / gt.true_intra_edges.len() as f64;

    let (ari_vs_gt, nmi_vs_gt) = compute_ari_and_nmi(&gt.gt_cell_of_1pc_node, &final_components);
    let (ari_vs_base, nmi_vs_base) = baseline_final_components
        .map(|base| compute_ari_and_nmi(base, &final_components))
        .unwrap_or((1.0, 1.0));

    println!("--------------------------------------------------------------------------------");
    println!("Variant: {}", name);
    println!(
        "  [1. Performance]   FLP={:.2}s | Clean={:.2}s | Leiden={:.2}s | Total={:.2}s | FLP_Loop_Heap={:.1} MB | FLP+Clean_Heap={:.1} MB | Leiden_Heap={:.1} MB",
        flp_sec,
        clean_sec,
        leiden_sec,
        flp_sec + clean_sec + leiden_sec,
        flp_loop_mem_mb,
        flp_total_mem_mb,
        leiden_mem_mb
    );
    println!(
        "  [2. FLP Folding]   FLP_Partitions={} | True_Crossing_Folded={} / {} ({:.3}%) | Intra_Cell_Coarsened_in_FLP={:.2}%",
        flp_num_partitions,
        folded_crossing_edges,
        total_true_crossing,
        folded_crossing_pct,
        flp_coarsened_intra_pct
    );
    println!(
        "  [3. Components]    ARI_vs_GT={:.6} | NMI_vs_GT={:.6} | ARI_vs_Base={:.6} | NMI_vs_Base={:.6}",
        ari_vs_gt, nmi_vs_gt, ari_vs_base, nmi_vs_base
    );
    println!(
        "                     True_Crossing_Removed={}/{} ({:.3}%) [Leaked={} ({} from FLP, {} from Leiden)] | True_Intra_Preserved={:.4}%",
        removed_crossing_edges,
        total_true_crossing,
        removed_crossing_pct,
        leaked_crossing_edges,
        folded_crossing_edges,
        leaked_crossing_edges.saturating_sub(folded_crossing_edges),
        preserved_intra_pct
    );
    println!(
        "                     Final_Components={} | Cells(>=8k)={} (GT={}) | Stranded_Nodes={} | Modularity_Q={:.2}",
        total_final_components,
        cells_ge_8k,
        gt.gt_cells_ge_8k,
        stranded_nodes,
        post_leiden_stats.quality
    );

    final_components
}

fn run_dataset_evaluation(
    dataset_prefix: &str,
    shuffle_edges_seed: Option<u64>,
    only_4cycles: bool,
) {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path_0pc = manifest_dir
        .join("test_data")
        .join(format!("{}_0pc.parquet", dataset_prefix));
    let path_1pc = manifest_dir
        .join("test_data")
        .join(format!("{}_1pc.parquet", dataset_prefix));

    println!("\n================================================================================");
    println!(
        "EVALUATING DATASET PAIR: {} (0pc vs 1pc) | shuffle_edges_seed={:?}",
        dataset_prefix, shuffle_edges_seed
    );
    println!("================================================================================");

    let base_load_mem = reset_peak_memory();
    let (mapping_1pc, graph_1pc) =
        load_graph_from_parquet_with_edge_order(path_1pc.to_str().unwrap(), shuffle_edges_seed);
    let graph_load_live_mb = LIVE_BYTES
        .load(AtomicOrdering::Relaxed)
        .saturating_sub(base_load_mem) as f64
        / (1024.0 * 1024.0);

    let gt = build_ground_truth(path_0pc.to_str().unwrap(), &mapping_1pc, &graph_1pc);
    drop(mapping_1pc);

    println!(
        "Graph (1pc): N={} | E_total={} | E_intra_gt={} | E_crossing_gt={} ({:.2}%)",
        graph_1pc.get_num_nodes(),
        graph_1pc.get_edge_entry_count(),
        gt.true_intra_edges.len(),
        gt.true_crossing_edges.len(),
        100.0 * gt.true_crossing_edges.len() as f64 / graph_1pc.get_edge_entry_count() as f64
    );
    println!(
        "Ground Truth (0pc): Total_Components={} | Cells(>=8k)={} | Graph+Mask_Live_Heap={:.1} MB",
        gt.gt_total_components, gt.gt_cells_ge_8k, graph_load_live_mb
    );

    let is_b = graph_1pc
        .get_bipartite_side_mask()
        .expect("Graph loaded from parquet must have bipartite side mask");
    let bip_loop_only_mb =
        (graph_1pc.get_num_nodes() * std::mem::size_of::<usize>()) as f64 / (1024.0 * 1024.0);

    let base_components_opt: Option<Vec<usize>> = if !only_4cycles {
        // 0. Baseline (random tie-breaking)
        let (p_base, _, l_base, c_base, ml_base, mt_base) =
            run_queue_flp_variant(&graph_1pc, QueueTieVariant::BaselineRandom, 42);
        let base_components = evaluate_variant(
            "0. Queue Baseline (random tie)",
            &graph_1pc,
            &gt,
            p_base,
            l_base,
            c_base,
            ml_base,
            mt_base,
            None,
        );

        // PR #33 (retain + random tie-breaking)
        let (p_pr33, _, l_pr33, c_pr33, ml_pr33, mt_pr33) =
            run_queue_flp_variant(&graph_1pc, QueueTieVariant::RetainRandom, 42);
        let _ = evaluate_variant(
            "PR#33 Queue (retain + random tie)",
            &graph_1pc,
            &gt,
            p_pr33,
            l_pr33,
            c_pr33,
            ml_pr33,
            mt_pr33,
            Some(&base_components),
        );

        // 1a. Queue: Smallest ID on tie
        let (p_1a, _, l_1a, c_1a, ml_1a, mt_1a) =
            run_queue_flp_variant(&graph_1pc, QueueTieVariant::SmallestOnly, 42);
        let _ = evaluate_variant(
            "1a. Queue (Smallest ID on tie)",
            &graph_1pc,
            &gt,
            p_1a,
            l_1a,
            c_1a,
            ml_1a,
            mt_1a,
            Some(&base_components),
        );

        // 1b. Queue: Retain + Smallest ID on tie
        let (p_1b, _, l_1b, c_1b, ml_1b, mt_1b) =
            run_queue_flp_variant(&graph_1pc, QueueTieVariant::RetainSmallest, 42);
        let _ = evaluate_variant(
            "1b. Queue (Retain + Smallest ID)",
            &graph_1pc,
            &gt,
            p_1b,
            l_1b,
            c_1b,
            ml_1b,
            mt_1b,
            Some(&base_components),
        );

        let base_bip_mem = reset_peak_memory();
        let t_bip = Instant::now();
        let p_bip: FastNodePartitioning = alternating_bipartite_flp(
            &graph_1pc,
            is_b,
            DEFAULT_BIPARTITE_FLP_MIN_CHANGED_FRACTION,
            DEFAULT_BIPARTITE_FLP_MAX_ROUNDS,
        );
        let bip_total_flp_sec = t_bip.elapsed().as_secs_f64();
        let bip_peak_mem_mb = peak_delta_mb(base_bip_mem);

        let _ = evaluate_variant(
            "2. Alternating Bipartite FLP (<1% changed, Smallest ID)",
            &graph_1pc,
            &gt,
            p_bip,
            bip_total_flp_sec,
            0.0,
            bip_loop_only_mb,
            bip_peak_mem_mb,
            Some(&base_components),
        );

        // Test Bipartite modifications on the randomized-edge graph (NO hashing!)
        for (label, modif) in [
            (
                "2a. Bipartite FLP: Hub-Attractor R0 (max deg_A, tie min ID)",
                BipartiteModification::HubAttractorR0,
            ),
            (
                "2b. Bipartite FLP: Hub-Attractor R0 + StrictMaj A in R0",
                BipartiteModification::HubAttractorR0StrictMajR0,
            ),
            (
                "2c. Bipartite FLP: Smallest R0 + StrictMaj A in R0",
                BipartiteModification::SmallestR0StrictMajR0,
            ),
            (
                "2d. Bipartite FLP: Hub-Attractor R0 + StrictMaj A (All Rounds)",
                BipartiteModification::HubAttractorR0StrictMajAllRounds,
            ),
            (
                "2e. Bipartite FLP: 4-Cycle Supported R0 + Hub + Smallest ID",
                BipartiteModification::FourCycleSupportedR0,
            ),
            (
                "2f. Bipartite FLP: 4-Cycle Supported R0 + StrictMaj A in R0",
                BipartiteModification::FourCycleSupportedR0StrictMajR0,
            ),
        ] {
            let (p_mod, sec_mod, mem_mod) = run_bipartite_modification(
                &graph_1pc,
                is_b,
                modif,
                DEFAULT_BIPARTITE_FLP_MIN_CHANGED_FRACTION,
                DEFAULT_BIPARTITE_FLP_MAX_ROUNDS,
            );
            let _ = evaluate_variant(
                label,
                &graph_1pc,
                &gt,
                p_mod,
                sec_mod,
                0.0,
                bip_loop_only_mb,
                mem_mod,
                Some(&base_components),
            );
        }
        Some(base_components)
    } else {
        let (p_base, _, l_base, c_base, ml_base, mt_base) =
            run_queue_flp_variant(&graph_1pc, QueueTieVariant::BaselineRandom, 42);
        let base_components = evaluate_variant(
            "0. Queue Baseline (random tie)",
            &graph_1pc,
            &gt,
            p_base,
            l_base,
            c_base,
            ml_base,
            mt_base,
            None,
        );
        Some(base_components)
    };

    // Approach 3: Direct 4-Cycle Merging (Union-Find over all 4-cycles in the bipartite graph)
    for (label, mode) in [
        (
            "3a. Merge All 4-Cycles (Union-Find) + Tree FLP",
            FourCycleMode::UnionFind4CyclesPlusTreeFlp,
        ),
        (
            "3b. Highest Precision: 4-Cycles + StrictMaj Tree FLP",
            FourCycleMode::UnionFind4CyclesPlusStrictTreeFlp,
        ),
        (
            "3c. Merge All 4-Cycles + Same-Core Ears/Bridges",
            FourCycleMode::UnionFind4CyclesPlusSameCoreEars,
        ),
        (
            "3d. Local 4-Cycle Only R0 + StrictMaj Bipartite FLP",
            FourCycleMode::Local4CycleSeededBipartiteFlp,
        ),
    ] {
        let (p_fc, sec_fc, loop_mem_fc, tot_mem_fc) = run_four_cycle_merge(&graph_1pc, is_b, mode);
        let _ = evaluate_variant(
            label,
            &graph_1pc,
            &gt,
            p_fc,
            sec_fc,
            0.0,
            loop_mem_fc,
            tot_mem_fc,
            base_components_opt.as_deref(),
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut target = "all".to_string();
    let mut only_4cycles = false;
    let mut shuffle_seed: Option<u64> = Some(42); // Default: randomize edge order before building graph!
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--dataset" && i + 1 < args.len() {
            target = args[i + 1].clone();
            i += 2;
        } else if args[i] == "--only-4cycles" {
            only_4cycles = true;
            i += 1;
        } else if args[i] == "--no-shuffle" {
            shuffle_seed = None;
            i += 1;
        } else if args[i] == "--shuffle-seed" && i + 1 < args.len() {
            shuffle_seed = Some(args[i + 1].parse().expect("Invalid seed"));
            i += 2;
        } else {
            target = args[i].clone();
            i += 1;
        }
    }

    match target.as_str() {
        "40" | "40cells" | "mix_40cells" => {
            run_dataset_evaluation("mix_40cells", shuffle_seed, only_4cycles);
        }
        "200" | "200cells" | "mix_200cells" => {
            run_dataset_evaluation("mix_200cells", shuffle_seed, only_4cycles);
        }
        "all" => {
            run_dataset_evaluation("mix_40cells", shuffle_seed, only_4cycles);
            run_dataset_evaluation("mix_200cells", shuffle_seed, only_4cycles);
        }
        other => {
            eprintln!(
                "Unknown dataset '{}'. Use --dataset 40cells, --dataset 200cells, or --dataset all.",
                other
            );
            std::process::exit(1);
        }
    }
}
