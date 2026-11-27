pub mod common;
pub mod edgelist;
pub mod encoding;
pub mod fast_label_propagation;
pub mod four_cycle_compression;
pub mod graph_collapse;
pub mod hybrid_community_detection;
pub mod leiden;

#[cfg(any(test, feature = "test-fixtures"))]
pub mod test_fixtures;
