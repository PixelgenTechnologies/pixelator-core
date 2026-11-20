use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use pixelator_core::common::graph::Graph;
use pixelator_core::common::node_indexing::UmiToNodeIndexMapping;
use pixelator_core::common::types::UMI;

use pixelator_core::graph_collapse::{graph_collapse, CollapseStats};

fn to_py_err(err: impl std::fmt::Debug) -> PyErr {
    PyRuntimeError::new_err(format!("{err:?}"))
}

/// One edgelist row: `(component, umi1, umi2)`.
type EdgelistRow = (u32, u64, u64);

/// One collapsed UMI: `(component, original, corrected)`.
type Rename = (u32, u64, u64);

/// Collapse likely sequencing-error UMIs in an edgelist.
///
/// The three arrays are the columns of the edgelist, one entry per edge. All
/// edges are held in memory, so the caller bounds memory by passing one batch
/// of whole components at a time. Rows may come in any order: they are grouped
/// by `component` and sorted before each graph is built, so the result does not
/// depend on the order. Components are searched in parallel.
///
/// Args:
///     umi1: 1-D uint64 numpy array of 2-bit-encoded `umi1` values.
///     umi2: 1-D uint64 numpy array of 2-bit-encoded `umi2` values.
///     component: 1-D uint32 numpy array with the component of each edge. It
///         only has to tell components apart; the caller maps it back to a name.
///     n_threads: Number of worker threads for the search.
///     bfs_threshold: Maximum graph distance searched for a correction target.
///     hamming_threshold: Maximum Hamming distance between an investigated UMI
///         and its correction target.
///     max_degree: Nodes with degree above this value are not searched. The
///         default, 1, investigates only leaves.
///
/// Returns:
///     A `(renames, stats)` tuple.
///
///     `renames` is a dict of three equally long 1-D numpy arrays, one entry per
///     collapsed UMI, sorted by `component` and then `original`:
///
///     - `component` (uint32): the component the UMI was collapsed in.
///     - `original` (uint64): the collapsed UMI.
///     - `corrected` (uint64): the UMI it is collapsed into.
///
///     `stats` is a dict of what the search counted, over all components:
///
///     - `total_components`: components processed.
///     - `nodes_considered`: nodes with degree at most `max_degree`.
///     - `targets_found`: kept matches. A match dropped because it would close
///       a loop is not counted, and neither is a pair that the caller later
///       discards (for example for differing markers).
///     - `bfs_distance_distribution`: `{hops: count}` over kept matches.
///     - `hamming_distance_distribution`: `{distance: count}` over kept matches.
///
///     Both distributions list only distances that occurred, in ascending
///     order. Each distance is that of the match as found: for a chain
///     `A -> B -> H` it is the distance from A to B and from B to H, not from
///     A to H.
///
/// Raises:
///     ValueError: If the arrays differ in length.
#[pyfunction(
    name = "graph_collapse",
    signature = (
        umi1,
        umi2,
        component,
        n_threads,
        bfs_threshold=6,
        hamming_threshold=2,
        max_degree=1,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn py_graph_collapse<'py>(
    py: Python<'py>,
    umi1: PyReadonlyArray1<'py, u64>,
    umi2: PyReadonlyArray1<'py, u64>,
    component: PyReadonlyArray1<'py, u32>,
    n_threads: usize,
    bfs_threshold: u64,
    hamming_threshold: u8,
    max_degree: u64,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let (umi1, umi2, component) = (umi1.as_array(), umi2.as_array(), component.as_array());
    if umi1.len() != umi2.len() || umi1.len() != component.len() {
        return Err(PyValueError::new_err(format!(
            "umi1, umi2 and component must have the same length, got {}, {} and {}",
            umi1.len(),
            umi2.len(),
            component.len()
        )));
    }
    // Copied while the GIL is held, so Python cannot change the arrays while
    // the search runs without it.
    let rows: Vec<EdgelistRow> = component
        .iter()
        .zip(umi1.iter())
        .zip(umi2.iter())
        .map(|((&component, &umi1), &umi2)| (component, umi1, umi2))
        .collect();

    // `graph_collapse` parallelizes on the current Rayon pool. A local pool
    // keeps that at `n_threads` instead of the CPU count.
    let pool = ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(to_py_err)?;
    let (renames, stats) = py.detach(move || {
        pool.install(|| run_graph_collapse(rows, bfs_threshold, hamming_threshold, max_degree))
    });

    Ok((renames_to_dict(py, renames)?, stats_to_dict(py, &stats)?))
}

/// Run graph collapse on every component of the edgelist.
///
/// Each component is loaded into its own graph and collapsed independently,
/// in parallel.
///
/// Returns the collapsed UMIs, sorted by component and then original UMI,
/// together with the stats summed over components.
fn run_graph_collapse(
    mut rows: Vec<EdgelistRow>,
    bfs_threshold: u64,
    hamming_threshold: u8,
    max_degree: u64,
) -> (Vec<Rename>, CollapseStats) {
    // Node indices and neighbor order follow the edge order, and they decide
    // which match the search finds first. Sorting groups each component into
    // one run and makes the result independent of the row order.
    rows.par_sort_unstable();

    let per_component: Vec<(Vec<Rename>, CollapseStats)> = rows
        .par_chunk_by(|a, b| a.0 == b.0)
        .map(|component_rows| {
            collapse_component(component_rows, bfs_threshold, hamming_threshold, max_degree)
        })
        .collect();

    let mut stats = CollapseStats::default();
    let mut renames = Vec::new();
    for (component_renames, component_stats) in per_component {
        stats.add_components(1);
        stats += component_stats;
        renames.extend(component_renames);
    }
    (renames, stats)
}

/// Collapse one component, given its rows sorted by `(umi1, umi2)`.
///
/// Returns the component's renames, sorted by original UMI, and its stats.
fn collapse_component(
    rows: &[EdgelistRow],
    bfs_threshold: u64,
    hamming_threshold: u8,
    max_degree: u64,
) -> (Vec<Rename>, CollapseStats) {
    let component = rows[0].0;
    let umi_pairs = rows
        .iter()
        .map(|&(_, umi1, umi2)| (umi1 as UMI, umi2 as UMI));
    let umi_mapping = UmiToNodeIndexMapping::from_umi_pairs(umi_pairs.clone());
    let num_nodes = umi_mapping.get_num_of_nodes();
    let edges = umi_mapping.map_umi_pair_iterator_to_edge(umi_pairs);
    let graph = Graph::<u8>::from_edges(edges, num_nodes);

    let (targets, stats) = graph_collapse(
        &graph,
        &umi_mapping,
        bfs_threshold as usize,
        hamming_threshold,
        max_degree as usize,
    );
    let mut renames: Vec<Rename> = targets
        .into_iter()
        .map(|(original, corrected)| (component, original as u64, corrected as u64))
        .collect();
    renames.sort_unstable();
    (renames, stats)
}

/// The renames as a dict of `component`, `original` and `corrected` numpy arrays.
fn renames_to_dict<'py>(py: Python<'py>, renames: Vec<Rename>) -> PyResult<Bound<'py, PyDict>> {
    let mut component = Vec::with_capacity(renames.len());
    let mut original = Vec::with_capacity(renames.len());
    let mut corrected = Vec::with_capacity(renames.len());
    for (c, o, r) in renames {
        component.push(c);
        original.push(o);
        corrected.push(r);
    }
    let out = PyDict::new(py);
    out.set_item("component", component.into_pyarray(py))?;
    out.set_item("original", original.into_pyarray(py))?;
    out.set_item("corrected", corrected.into_pyarray(py))?;
    Ok(out)
}

/// Mirrors [`CollapseStats`] as a plain dict, with the keys listed in the
/// docstring of [`py_graph_collapse`].
fn stats_to_dict<'py>(py: Python<'py>, stats: &CollapseStats) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("total_components", stats.total_components)?;
    out.set_item("nodes_considered", stats.nodes_considered)?;
    out.set_item("targets_found", stats.targets_found)?;

    let bfs_distance_dict = PyDict::new(py);
    for (d, c) in stats.bfs_distance_counts() {
        bfs_distance_dict.set_item(d, c)?;
    }
    out.set_item("bfs_distance_distribution", bfs_distance_dict)?;

    let hamming_distance_dict = PyDict::new(py);
    for (d, c) in stats.hamming_distance_counts() {
        hamming_distance_dict.set_item(d, c)?;
    }
    out.set_item("hamming_distance_distribution", hamming_distance_dict)?;

    Ok(out)
}

#[cfg(test)]
mod tests {
    use pixelator_core::encoding::encode_dna_2bit;

    use super::*;

    fn umi(seq: &str) -> u64 {
        encode_dna_2bit(seq.as_bytes())
    }

    #[test]
    fn chain_collapses_within_its_component_only() {
        // A chain A -> B -> H in component 7. Its rows are interleaved with
        // component 3, so grouping cannot rely on the row order. CCAA is one
        // symbol from CAAA, and CAAA is one symbol from both CCAA and AAAA.
        // Rows are sorted before the search, so CAAA meets AAAA, the chain's
        // end, first. Component 3 is one edge between two distant UMIs, which
        // has no collapse target. M and N are umi1 and every other UMI is
        // umi2, so none sits on both sides.
        let umi_m = umi("GGGG");
        let umi_h = umi("AAAA");
        let umi_a = umi("CCAA");
        let umi_b = umi("CAAA");
        let umi_n = umi("TTTT");
        let rows = vec![
            (7, umi_m, umi_h),
            (3, umi("ACAC"), umi("GTGT")),
            (7, umi_m, umi_b),
            (7, umi_n, umi_h),
            (7, umi_m, umi_a),
        ];

        let (renames, stats) = run_graph_collapse(rows, 2, 1, 1);

        let mut expected = vec![(7, umi_a, umi_h), (7, umi_b, umi_h)];
        expected.sort_unstable();
        assert_eq!(renames, expected);
        // Component 7 has three leaves (A, B and N) and component 3 has two.
        // Only the chain in component 7 finds matches.
        assert_eq!(stats.total_components, 2);
        assert_eq!(stats.nodes_considered, 5);
        assert_eq!(stats.targets_found, 2);
        assert_eq!(stats.bfs_distance_counts(), [(2, 2)]);
        assert_eq!(stats.hamming_distance_counts(), [(1, 2)]);
    }

    #[test]
    fn same_umis_in_two_components_collapse_separately() {
        // The same leaf chain appears in components 1 and 2. Each component is
        // its own graph, so each reports its own renames.
        let umi_m = umi("GGGG");
        let umi_h = umi("AAAA");
        let umi_a = umi("CAAA");
        let umi_n = umi("TTTT");
        let rows = vec![
            (2, umi_m, umi_h),
            (1, umi_m, umi_h),
            (2, umi_m, umi_a),
            (1, umi_m, umi_a),
            (2, umi_n, umi_h),
            (1, umi_n, umi_h),
        ];

        let (renames, stats) = run_graph_collapse(rows, 2, 1, 1);

        assert_eq!(renames, vec![(1, umi_a, umi_h), (2, umi_a, umi_h)]);
        assert_eq!(stats.total_components, 2);
        assert_eq!(stats.targets_found, 2);
    }

    #[test]
    fn empty_edgelist_returns_no_renames() {
        let (renames, stats) = run_graph_collapse(Vec::new(), 2, 1, 1);

        assert!(renames.is_empty());
        assert_eq!(stats.total_components, 0);
        assert_eq!(stats.nodes_considered, 0);
        assert_eq!(stats.targets_found, 0);
    }
}
