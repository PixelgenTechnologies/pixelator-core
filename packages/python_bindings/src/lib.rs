use pixelator_core::common::io::{
    create_graph_and_umi_mapping_from_parquet_file, filter_out_crossing_edges_from_edge_list,
    write_node_partitions_to_parquet,
};
use pyo3::prelude::*;

use pixelator_core::common::graph::GraphProperties;
use pixelator_core::common::node_partitioning::{
    FastNodePartitioning, LeidenNodePartitioning, NodePartitioning,
};
use pixelator_core::fast_label_propagation::algorithm::fast_label_propagation;
use pixelator_core::fast_label_propagation::strategies::{
    AssignmentStrategy, DefaultAssignmentStrategy,
};
use pixelator_core::leiden::algorithm::leiden;
use pixelator_core::leiden::quality::modularity::Modularity;
use pixelator_core::leiden::weighted_partitioned_graph::WeightedPartitionedGraph;
use pixelator_core::new_graph::algorithm::hybrid_community_detection;

use log::debug;

use rustc_hash::FxHashMap as HashMap;

use paste::paste;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[pymodule(name = "pixelator_core")]
mod pixelator_core_py {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::find_graph_statistics;

    #[pymodule_export]
    use super::run_label_propagation;

    #[pymodule_export]
    use super::run_leiden;

    #[pymodule_export]
    use super::run_hybrid_community_detection;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        pyo3_log::init();
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
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
pub fn find_graph_statistics(parquet_file: String) -> PyResult<(usize, usize, usize, f64)> {
    let (_, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(&parquet_file);

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
/// * Graph statistics after Leiden.
///
/// The node partitioning is written in a Parquet file as specified by the `output` parameter.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn run_hybrid_community_detection(
    parquet_file: String,
    resolution: f64,
    output: Option<String>,
    flp_epochs: Option<u64>,
    randomness: Option<f64>,
    seed: Option<u64>,
    max_iteration: Option<usize>,
    multiplet_recovery: bool,
) -> PyResult<(
    String,
    PyGraphProperties,
    PyGraphProperties,
    PyGraphProperties,
)> {
    let output_file = output.unwrap_or_else(|| "filtered_edge_list.parquet".to_string());
    let (umi_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(&parquet_file);
    let quality_function = Modularity::new(resolution, graph.get_total_edge_weight());

    let pre_recovery_properties = GraphProperties::new(&graph);

    let (node_partition, post_flp_statistics, post_leiden_statistics) = hybrid_community_detection(
        graph,
        quality_function,
        randomness.unwrap_or(0.1),
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

    debug!("Writing data to parquet {}", output_file);
    filter_out_crossing_edges_from_edge_list(
        &parquet_file,
        &output_file,
        &node_partition,
        &umi_mapping,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;
    Ok((
        output_file,
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
#[pyfunction]
pub fn run_label_propagation(
    parquet_file: String,
    epochs: Option<u64>,
    output: Option<String>,
) -> PyResult<usize> {
    let epochs = epochs.unwrap_or(1);
    let output_file = output.unwrap_or_else(|| "node_partitions.parquet".to_string());

    let (umi_mapping, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(&parquet_file);

    let node_partition =
        FastNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
    let assignment_strategy: &dyn AssignmentStrategy<_> = &DefaultAssignmentStrategy;

    let node_partition =
        fast_label_propagation(&graph, epochs, assignment_strategy, node_partition);

    write_node_partitions_to_parquet(output_file, &node_partition, &umi_mapping, None)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;

    Ok(node_partition.num_partitions())
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
///
/// # Returns
/// A tuple containing:
/// * The number of partitions
/// * The overall quality of the partitioning
///
/// The node partitioning is written in a Parquet file as specified by the `output` parameter.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn run_leiden(
    parquet_file: String,
    max_iteration: Option<usize>,
    partition: Option<Vec<usize>>,
    resolution: f64,
    output: Option<String>,
    randomness: Option<f64>,
    seed: Option<u64>,
) -> PyResult<(usize, f64)> {
    let output_file = output.unwrap_or_else(|| "node_partitions.parquet".to_string());
    let (umi_mapping, graph) =
        create_graph_and_umi_mapping_from_parquet_file::<usize>(&parquet_file);

    let partition = if let Some(node_partition_map) = partition {
        LeidenNodePartitioning::initialize_from_partitions(node_partition_map)
    } else {
        LeidenNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes())
    };
    let quality_function = Modularity::new(resolution, graph.get_total_edge_weight());

    let mut wp_graph =
        WeightedPartitionedGraph::new(graph, partition, quality_function, None, seed);

    leiden(
        &mut wp_graph,
        randomness.unwrap_or(0.1),
        max_iteration,
        None,
    );

    let node_partition = wp_graph.get_partitioning();
    write_node_partitions_to_parquet(output_file, node_partition, &umi_mapping, None)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;

    Ok((node_partition.num_partitions(), wp_graph.quality()))
}
