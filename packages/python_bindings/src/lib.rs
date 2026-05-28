use pixelator_core::common::io::{
    create_graph_and_umi_mapping_from_parquet_file, filter_out_crossing_edges_from_edge_list,
    write_node_partitions_to_parquet,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyTuple};

use pixelator_core::common::graph::{Graph, GraphProperties};
use pixelator_core::common::node_partitioning::{
    FastNodePartitioning, LeidenNodePartitioning, NodePartitioning,
};
use pixelator_core::common::types::Edge;
use pixelator_core::fast_label_propagation::algorithm::fast_label_propagation;
use pixelator_core::fast_label_propagation::strategies::{
    AssignmentStrategy, DefaultAssignmentStrategy,
};
use pixelator_core::hybrid_community_detection::algorithm::hybrid_community_detection;
use pixelator_core::leiden::algorithm::{leiden, ThresholdOptions};
use pixelator_core::leiden::quality::modularity::Modularity;
use pixelator_core::leiden::weighted_partitioned_graph::WeightedPartitionedGraph;

use log::debug;

use rustc_hash::FxHashMap as HashMap;

use paste::paste;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[pymodule(name = "pixelator_core")]
fn pixelator_core_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(find_graph_statistics, m)?)?;
    m.add_function(wrap_pyfunction!(run_label_propagation, m)?)?;
    m.add_function(wrap_pyfunction!(run_label_propagation_networkx, m)?)?;
    m.add_function(wrap_pyfunction!(run_leiden, m)?)?;
    m.add_function(wrap_pyfunction!(run_leiden_networkx, m)?)?;
    m.add_function(wrap_pyfunction!(run_hybrid_community_detection, m)?)?;
    m.add_class::<PyGraphProperties>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

///
/// Finds graph statistics from an edge list stored in a Parquet file.
///
/// # Arguments
/// * `parquet_file` - Path to the Parquet file containing the edge list.
///
/// # Returns
/// A tuple containing:
/// * Number of nodes
/// * Number of edges
/// * Number of connected components
/// * Fraction of nodes in the largest connected component
///
#[pyfunction]
pub fn find_graph_statistics(parquet_file: &str) -> PyResult<(usize, usize, usize, f64)> {
    let (_, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(parquet_file);

    let graph_properties = GraphProperties::new(&graph);

    Ok((
        graph_properties.node_count,
        graph_properties.edge_weight_sum,
        graph_properties.n_connected_components,
        graph_properties.fraction_in_largest_component,
    ))
}

/// Generate a Python equivalent of a Rust dataclass, including a `from` method to convert between
/// the two
///
/// NB: unfortunately you need to input all the fields and their type, since macros cannot fetch
/// data other than what is provided to them. This macro prevents you from having to write the
/// fields again in the from function.
macro_rules! py_dataclass {
    ($name:ident { $($field:ident: $ty:ty),* $(,)? }) => {
        paste! {
            #[pyo3::pyclass(get_all)]
            pub struct [<Py $name>] {
                $(
                    pub $field: $ty
                ),*
            }

            impl From<$name> for [<Py $name>] {
                fn from($name { $($field),* }: $name) -> Self {
                    Self { $($field),* }
                }
            }
        }
    };
}

py_dataclass!(GraphProperties {
    edge_weight_sum: usize,
    node_count: usize,
    n_connected_components: usize,
    fraction_in_largest_component: f64,
    stranded_nodes: usize,
    component_size_distribution: HashMap<usize, usize>,
});

/// Finds community partitioning by combining Fast Label Propagation (FLP), graph aggregation,
/// and optionally the Leiden algorithm for multiplet recovery.
///
/// # Arguments
/// * `parquet_file` - Path to the Parquet file containing the edge list.
/// * `resolution` - Resolution parameter for the modularity quality function used in the Leiden
/// algorithm. Larger values tend to yield smaller communities.
/// * `output` - Path to the filtered edge-list Parquet file to write. Default is
///   `filtered_edge_list.parquet`.
/// * `flp_epochs` - Number of full passes of FLP on the original graph before aggregation.
///   Default is 1.
/// * `randomness` - Randomness of Leiden node moves when `multiplet_recovery` is true. Low values
///   favor moves that improve quality; higher values allow suboptimal moves and can help escape
///   local minima at the cost of convergence speed. Default is 0.1.
/// * `seed` - Seed for the random number generator in the weighted partitioned graph (aggregation
///   and Leiden). If `None`, the implementation uses a default seed of 0.
/// * `max_iteration` - Maximum number of Leiden iterations when `multiplet_recovery` is true.
///   In most cases the algorithm stops earlier once converged.
/// * `multiplet_recovery` - If true, runs Leiden on the aggregated graph after FLP. If false,
///   the pipeline stops after aggregation and the last statistics tuple matches the aggregated
///   state without a Leiden refinement pass.
///
/// # Returns
/// A tuple containing:
/// * The path to the written Parquet file (same as `output`, or the default path).
/// * Graph statistics on the input graph before multiplet recovery (pre-FLP).
/// * Graph statistics after FLP and aggregation.
/// * Graph statistics after Leiden when `multiplet_recovery` is true, otherwise graph
///   statistics after aggregation.
///
/// The node partitioning is written in a Parquet file as specified by the `output` parameter.
#[pyfunction(signature = (
    parquet_file,
    resolution,
    multiplet_recovery,
    output="filtered_edge_list.parquet",
    flp_epochs=1,
    randomness=0.1,
    seed=None,
    max_iteration=None
))]
#[allow(clippy::too_many_arguments)]
pub fn run_hybrid_community_detection(
    parquet_file: &str,
    resolution: f64,
    multiplet_recovery: bool,
    output: &str,
    flp_epochs: u64,
    randomness: f64,
    seed: Option<u64>,
    max_iteration: Option<usize>,
) -> PyResult<(
    String,
    PyGraphProperties,
    PyGraphProperties,
    PyGraphProperties,
)> {
    let (umi_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(parquet_file);
    let quality_function = Modularity::new(resolution, graph.get_total_edge_weight());

    let pre_recovery_properties = GraphProperties::new(&graph);

    let (node_partition, post_flp_statistics, post_leiden_statistics) = hybrid_community_detection(
        graph,
        quality_function,
        randomness,
        seed,
        max_iteration,
        flp_epochs,
        false,
        multiplet_recovery,
    );

    // NB: technically, crossing edges are not removed from the graph until the edge list is saved
    // to parquet. The statistics below are adjusted to exclude such crossing edges.
    let post_flp_properties = GraphProperties::from(post_flp_statistics);
    let post_recovery_properties = GraphProperties::from(post_leiden_statistics);

    debug!("Writing data to parquet {}", output);
    filter_out_crossing_edges_from_edge_list(&parquet_file, &output, &node_partition, &umi_mapping)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;
    Ok((
        output.to_string(),
        PyGraphProperties::from(pre_recovery_properties),
        PyGraphProperties::from(post_flp_properties),
        PyGraphProperties::from(post_recovery_properties),
    ))
}

/// Finds community partitioning using the Fast Label Propagation (FLP) algorithm.
///
/// # Arguments
/// * `parquet_file` - Path to the Parquet file containing the edge list.
/// * `epochs` - Number of times FLP is run. Default is once.
/// * `output` - Path to the output parquet file. Default is `node_partitions.parquet`.
///
/// # Returns
/// * The number of partitions
///
/// The node partitioning is written in a Parquet file as specified by the `output` parameter.
#[pyfunction(signature = (
    parquet_file,
    epochs=1,
    output="node_partitions.parquet"
))]
pub fn run_label_propagation(parquet_file: &str, epochs: u64, output: &str) -> PyResult<usize> {
    let (umi_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(parquet_file);

    let node_partition = run_flp_core(&graph, epochs);

    write_node_partitions_to_parquet(output, &node_partition, &umi_mapping, None)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;

    Ok(node_partition.num_partitions())
}

fn validate_networkx_graph(graph: &Bound<'_, PyAny>) -> PyResult<()> {
    if !graph.hasattr("nodes")? || !graph.hasattr("edges")? {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected a NetworkX graph object with `nodes` and `edges` methods",
        ));
    }
    Ok(())
}

fn extract_node_index(
    node_to_index: &Bound<'_, PyDict>,
    node: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    node_to_index
        .get_item(node)?
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Encountered edge with node not present in graph.nodes()",
            )
        })?
        .extract::<usize>()
}

fn extract_non_negative_integral_weight(weight: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(value) = weight.extract::<usize>() {
        return Ok(value);
    }

    if let Ok(value) = weight.extract::<f64>() {
        if value.is_finite() && value >= 0.0 && value.fract() == 0.0 {
            return Ok(value as usize);
        }
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Expected NetworkX edge weights to be non-negative integer-like values",
    ))
}

fn networkx_to_weighted_edges(
    graph: &Bound<'_, PyAny>,
) -> PyResult<(Vec<Py<PyAny>>, Vec<(usize, usize, usize)>)> {
    validate_networkx_graph(graph)?;
    let py = graph.py();

    let node_to_index = PyDict::new(py);
    let mut node_labels: Vec<Py<PyAny>> = Vec::new();
    for (idx, node_item) in graph.call_method0("nodes")?.try_iter()?.enumerate() {
        let node = node_item?;
        node_to_index.set_item(&node, idx)?;
        node_labels.push(node.unbind());
    }

    let edge_kwargs = PyDict::new(py);
    edge_kwargs.set_item("data", "weight")?;
    edge_kwargs.set_item("default", 1)?;

    let mut weighted_edges: Vec<(usize, usize, usize)> = Vec::new();
    for edge_item in graph
        .call_method("edges", (), Some(&edge_kwargs))?
        .try_iter()?
    {
        let edge_any = edge_item?;
        let edge_tuple = edge_any
            .downcast::<PyTuple>()
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Expected NetworkX edges(data='weight') to yield (u, v, weight) tuples",
                )
            })?;
        if edge_tuple.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected NetworkX edges(data='weight') to yield 3-tuples",
            ));
        }

        let source = edge_tuple.get_item(0)?;
        let destination = edge_tuple.get_item(1)?;
        let weight = edge_tuple.get_item(2)?;
        let source_idx = extract_node_index(&node_to_index, &source)?;
        let destination_idx = extract_node_index(&node_to_index, &destination)?;
        let edge_weight = extract_non_negative_integral_weight(&weight)?;
        weighted_edges.push((source_idx, destination_idx, edge_weight));
    }

    Ok((node_labels, weighted_edges))
}

fn partition_to_python_communities<P: NodePartitioning>(
    py: Python<'_>,
    node_partition: &P,
    node_labels: &[Py<PyAny>],
) -> PyResult<Vec<Py<PyAny>>> {
    let mut communities: Vec<(usize, Py<PyAny>)> = Vec::new();
    for nodes in node_partition.get_partition_to_node_map().into_values() {
        let mut members: Vec<Py<PyAny>> = Vec::with_capacity(nodes.len());
        let mut min_index = usize::MAX;
        for node_index in nodes {
            min_index = min_index.min(node_index);
            members.push(node_labels[node_index].clone_ref(py));
        }
        let community = PySet::new(py, members)?;
        communities.push((min_index, community.into_any().unbind()));
    }
    communities.sort_by_key(|(min_index, _)| *min_index);
    Ok(communities.into_iter().map(|(_, community)| community).collect())
}

fn get_merge_threshold(
    merge_edge_threshold: Option<usize>,
    merge_edge_threshold_relative: Option<f64>,
) -> PyResult<Option<ThresholdOptions>> {
    match (merge_edge_threshold, merge_edge_threshold_relative) {
        (Some(_), Some(_)) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "`merge_edge_threshold` and `merge_edge_threshold_relative` cannot both be used at the same time",
        )),
        (Some(value), None) => Ok(Some(ThresholdOptions::Absolute(value))),
        (None, Some(value)) => Ok(Some(ThresholdOptions::Relative(value))),
        (None, None) => Ok(None),
    }
}

fn run_flp_core(graph: &Graph<u8>, epochs: u64) -> FastNodePartitioning {
    let node_partition = FastNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
    let assignment_strategy: &dyn AssignmentStrategy<_> = &DefaultAssignmentStrategy;
    fast_label_propagation(graph, epochs, assignment_strategy, node_partition)
}

fn build_leiden_partition(
    partition: Option<Vec<usize>>,
    node_count: usize,
    mismatch_err: Option<&'static str>,
) -> PyResult<LeidenNodePartitioning> {
    if let Some(node_partition_map) = partition {
        if let Some(mismatch_err) = mismatch_err {
            if node_partition_map.len() != node_count {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(mismatch_err));
            }
        }
        Ok(LeidenNodePartitioning::initialize_from_partitions(
            node_partition_map,
        ))
    } else {
        Ok(LeidenNodePartitioning::initialize_with_singlet_partitions(
            node_count,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn run_leiden_core(
    graph: Graph<usize>,
    resolution: f64,
    max_iteration: Option<usize>,
    partition: Option<Vec<usize>>,
    randomness: f64,
    seed: Option<u64>,
    merge_threshold: Option<ThresholdOptions>,
    partition_mismatch_err: Option<&'static str>,
) -> PyResult<(FastNodePartitioning, f64)> {
    let partition = build_leiden_partition(partition, graph.get_num_nodes(), partition_mismatch_err)?;
    let quality_function = Modularity::new(resolution, graph.get_total_edge_weight());
    let mut wp_graph = WeightedPartitionedGraph::new(graph, partition, quality_function, None, seed);
    leiden(&mut wp_graph, randomness, max_iteration, merge_threshold);

    let node_partition =
        FastNodePartitioning::initialize_from_partitions(wp_graph.get_ancestor_to_partition_map());
    Ok((node_partition, wp_graph.quality()))
}

/// Finds community partitioning using Fast Label Propagation (FLP) on a NetworkX graph.
///
/// # Arguments
/// * `graph` - A NetworkX graph object.
/// * `epochs` - Number of times FLP is run. Default is once.
///
/// # Returns
/// A list of disjoint sets representing communities of the graph.
#[pyfunction(signature = (graph, epochs=1))]
pub fn run_label_propagation_networkx(
    py: Python<'_>,
    graph: Py<PyAny>,
    epochs: u64,
) -> PyResult<Vec<Py<PyAny>>> {
    let graph = graph.bind(py);
    let (node_labels, weighted_edges) = networkx_to_weighted_edges(graph)?;
    let edges = weighted_edges
        .into_iter()
        .map(|(src, dest, weight)| {
            let converted_weight = u8::try_from(weight).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "NetworkX edge weight exceeds FLP u8 weight range",
                )
            })?;
            Ok(Edge::new(src, dest, Some(converted_weight)))
        })
        .collect::<PyResult<Vec<Edge<u8>>>>()?;
    let rust_graph = Graph::<u8>::from_edges(edges.into_iter(), node_labels.len());

    let node_partition = run_flp_core(&rust_graph, epochs);

    partition_to_python_communities(py, &node_partition, &node_labels)
}

/// Finds community partitioning using the Leiden algorithm
///
/// # Arguments
/// * `parquet_file` - Path to the Parquet file containing the edge list.
/// * `max_iteration` - maximum number of iterations to perform. NB: in most cases the algorithm
/// should converge and stop by itself.
/// * `partition` - initial node partitioning to use. If not provided, singlet partitions will be
/// used
/// * `resolution` - resolution to use in the quality function. The larger the resolution, the
/// smaller the resulting communities will be.
/// * `output` - Path to the output parquet file. Default is `node_partitions.parquet`.
/// * `randomness` - Randomness of node transitions. Low values will favor moves maximizing the
/// quality, while higher values will allow suboptimal moves, making it easier to avoid local
/// minima at the cost of convergence speed. Default value is 0.1
/// * `seed` - seed to use in the random generator
/// * `merge_edge_threshold` - Merge communities when their connecting edge count exceeds this
///   absolute threshold. Cannot be used together with `merge_edge_threshold_relative`.
/// * `merge_edge_threshold_relative` - Merge communities when their connecting edge relative to the
/// number of nodes in the smallest community exceeds this threshold. Cannot be used together with
/// `merge_edge_threshold`.
///
/// # Returns
/// A tuple containing:
/// * The number of partitions
/// * The overall quality of the partitioning
///
/// The node partitioning is written in a Parquet file as specified by the `output` parameter.
#[pyfunction(signature = (
    parquet_file,
    resolution,
    max_iteration=None,
    partition=None,
    output="node_partitions.parquet",
    randomness=0.1,
    seed=None,
    merge_edge_threshold=None,
    merge_edge_threshold_relative=None
))]
#[allow(clippy::too_many_arguments)]
pub fn run_leiden(
    parquet_file: &str,
    resolution: f64,
    max_iteration: Option<usize>,
    partition: Option<Vec<usize>>,
    output: &str,
    randomness: f64,
    seed: Option<u64>,
    merge_edge_threshold: Option<usize>,
    merge_edge_threshold_relative: Option<f64>,
) -> PyResult<(usize, f64)> {
    let merge_threshold = get_merge_threshold(merge_edge_threshold, merge_edge_threshold_relative)?;

    let (umi_mapping, graph) =
        create_graph_and_umi_mapping_from_parquet_file::<usize>(parquet_file);

    let (node_partition, quality) = run_leiden_core(
        graph,
        resolution,
        max_iteration,
        partition,
        randomness,
        seed,
        merge_threshold,
        None,
    )?;
    write_node_partitions_to_parquet(output, &node_partition, &umi_mapping, None)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;

    Ok((node_partition.num_partitions(), quality))
}

/// Finds community partitioning using the Leiden algorithm on a NetworkX graph.
///
/// # Arguments
/// * `graph` - A NetworkX graph object.
/// * `max_iteration` - Maximum number of Leiden iterations to perform.
/// * `partition` - Optional initial node partition labels aligned with graph node iteration order.
/// * `resolution` - Resolution parameter used in the modularity quality function.
/// * `randomness` - Randomness of node transitions in Leiden.
/// * `seed` - Seed for the random generator.
/// * `merge_edge_threshold` - Absolute merge threshold.
/// * `merge_edge_threshold_relative` - Relative merge threshold.
///
/// # Returns
/// A list of disjoint sets representing communities of the graph.
#[pyfunction(signature = (
    graph,
    resolution,
    max_iteration=None,
    partition=None,
    randomness=0.1,
    seed=None,
    merge_edge_threshold=None,
    merge_edge_threshold_relative=None
))]
#[allow(clippy::too_many_arguments)]
pub fn run_leiden_networkx(
    py: Python<'_>,
    graph: Py<PyAny>,
    resolution: f64,
    max_iteration: Option<usize>,
    partition: Option<Vec<usize>>,
    randomness: f64,
    seed: Option<u64>,
    merge_edge_threshold: Option<usize>,
    merge_edge_threshold_relative: Option<f64>,
) -> PyResult<Vec<Py<PyAny>>> {
    let graph = graph.bind(py);
    let merge_threshold = get_merge_threshold(merge_edge_threshold, merge_edge_threshold_relative)?;
    let (node_labels, weighted_edges) = networkx_to_weighted_edges(graph)?;
    let edges = weighted_edges
        .into_iter()
        .map(|(src, dest, weight)| Edge::new(src, dest, Some(weight)));
    let rust_graph = Graph::<usize>::from_edges(edges, node_labels.len());

    let (node_partition, _) = run_leiden_core(
        rust_graph,
        resolution,
        max_iteration,
        partition,
        randomness,
        seed,
        merge_threshold,
        Some("Length of `partition` must match the number of nodes in the NetworkX graph"),
    )?;
    partition_to_python_communities(py, &node_partition, &node_labels)
}
