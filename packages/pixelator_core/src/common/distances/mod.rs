mod bounded_levenshtein;
mod hamming;

pub use bounded_levenshtein::within_bounded_levenshtein;
pub use hamming::{hamming_distance_2bit, within_hamming_distance};
