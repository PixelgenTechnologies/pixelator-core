use std::time::Instant;

use clap::Parser;
use clap::Subcommand;
use env_logger::{Builder, Env};
use log::{debug, info};

use pixelator_core::common::graph::GraphProperties;
use pixelator_core::common::io::create_graph_and_umi_mapping_from_parquet_file;
use pixelator_core::common::io::write_node_partitions_to_parquet;
use pixelator_core::common::node_partitioning::{FastNodePartitioning, NodePartitioning};
use pixelator_core::fast_label_propagation::algorithm::fast_label_propagation;
use pixelator_core::fast_label_propagation::strategies::{
    AssignmentStrategy, DefaultAssignmentStrategy,
};
use pixelator_core::leiden::algorithm::leiden;
use pixelator_core::leiden::quality::modularity::Modularity;
use pixelator_core::leiden::weighted_partitioned_graph::WeightedPartitionedGraph;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run fast label propagation
    Flp {
        /// Input Parquet file
        #[arg()]
        parquet_file: String,

        /// Optional limit to how many epochs to run (1 is default)
        #[arg(long)]
        epochs: Option<u64>,

        /// Output Parquet file for node partitions
        #[arg(long, default_value = "node_partitions.parquet")]
        output: String,
    },
    /// Run Leiden
    Leiden {
        /// Input Parquet file
        #[arg()]
        parquet_file: String,

        #[arg(long, default_value_t = 1.0)]
        resolution: f64,

        /// Output Parquet file for node partitions
        #[arg(long, default_value = "node_partitions.parquet")]
        output: String,
    },

    /// Run Stats
    Stats {
        /// Input Parquet file
        #[arg()]
        parquet_file: String,
    },
}

fn main() {
    let cli = Cli::parse();

    let log_level = match cli.debug {
        0 => "info",
        _ => "debug",
    };
    let env = Env::new().filter_or("RUST_LOG", log_level);
    Builder::from_env(env).init();

    match &cli.command {
        Some(Commands::Flp {
            parquet_file,
            epochs,
            output,
        }) => {
            let res = run_flp(parquet_file, epochs, output);
            if res.is_err() {
                std::process::exit(1);
            }
        }
        Some(Commands::Leiden {
            parquet_file,
            resolution,
            output,
        }) => {
            let res = run_leiden(parquet_file, *resolution, 1.0, Some(1), None, output);
            if res.is_err() {
                std::process::exit(1);
            }
        }
        Some(Commands::Stats { parquet_file }) => {
            info!("Running stats");
            info!("Input arguments were:");
            info!("  Parquet file: {}", parquet_file);

            let (_, graph) = create_graph_and_umi_mapping_from_parquet_file::<u8>(parquet_file);

            let graph_properties = GraphProperties::new(&graph);
            info!("Graph properties:");
            info!("  Number of nodes: {}", graph_properties.node_count);
            info!(
                "  Total edge weight (undirected): {}",
                graph_properties.edge_weight_sum
            );
            info!(
                "  Number of connected components: {}",
                graph_properties.n_connected_components
            );
            info!(
                "  Percent in largest component: {:.4}%",
                graph_properties.fraction_in_largest_component * 100.0
            );
        }
        None => {}
    }
}

fn run_flp(
    parquet_file: &String,
    epochs: &Option<u64>,
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Running FLP");
    info!("Input arguments were:");
    info!("  Parquet file: {}", parquet_file);
    info!("  Epochs: {:?}", epochs);
    info!("  Output file: {}", output);
    let start_time = Instant::now();

    let epochs = epochs.unwrap_or(1);

    info!("Starting label propagation adjacency matrix construction...");
    let (mapping, graph) = create_graph_and_umi_mapping_from_parquet_file(parquet_file);

    let node_partition =
        FastNodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());

    let assignment_strategy: &dyn AssignmentStrategy<FastNodePartitioning> =
        &DefaultAssignmentStrategy;
    let node_partition =
        fast_label_propagation(&graph, epochs, assignment_strategy, node_partition);

    info!(
        "Number of communities found: {}",
        node_partition.num_partitions()
    );

    write_node_partitions_to_parquet(output, &node_partition, &mapping, None)?;

    info!(
        "FLP finished in {:.2?} seconds",
        Instant::now().duration_since(start_time)
    );

    Ok(())
}

fn run_leiden(
    parquet_file: &String,
    resolution: f64,
    randomness: f64,
    seed: Option<u64>,
    max_iterations: Option<usize>,
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Running leiden");
    info!("Input arguments were:");
    info!("  Parquet file: {}", parquet_file);

    let start_time = Instant::now();
    let (mapping, graph) = create_graph_and_umi_mapping_from_parquet_file(parquet_file);

    let node_partition =
        NodePartitioning::initialize_with_singlet_partitions(graph.get_num_nodes());
    let quality_function = Modularity::new(resolution, graph.get_total_edge_weight());
    let mut wp_graph =
        WeightedPartitionedGraph::new(graph, node_partition, quality_function, None, seed);

    leiden(&mut wp_graph, randomness, max_iterations, None);
    debug!(
        "{} partitions found",
        wp_graph.get_partitioning().num_partitions()
    );
    debug!("mod={}", wp_graph.quality());

    write_node_partitions_to_parquet(output, wp_graph.get_partitioning(), &mapping, Some(2048))?;

    info!(
        "Leiden finished in {:.2?} seconds",
        Instant::now().duration_since(start_time)
    );
    Ok(())
}
