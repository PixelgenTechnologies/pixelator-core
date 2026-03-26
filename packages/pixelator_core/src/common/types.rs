use std::hash::{Hash, Hasher};
use std::{iter::Sum, ops::Add};

use num_traits::Num;

/// The UMIS are DNA sequences in 2-bit encoding and
/// packed into 4-bit unsigned integers.
pub type UMI = usize;

/// A pair of UMIs.
pub type UMIPair = (UMI, UMI);

/// Type alias for node indices in a graph.
/// These are arbitrary, but need to be consecutive integers
/// starting from 0.
pub type NodeIdx = usize;
/// Type alias for partition or community identifiers.
pub type PartitionId = usize;

/// Trait for types that can be used as edge weights in a graph.
/// Must support numeric operations, conversion, and basic traits.
pub trait EdgeWeight:
    Into<usize> + Num + Clone + Copy + Default + Add + Sum + std::fmt::Debug
{
}
impl EdgeWeight for u8 {}
impl EdgeWeight for usize {}

/// Represents a weighted edge in a graph.
///
/// # Type Parameters
/// * `T` - The type of the edge weight, must implement `EdgeWeight`.
#[derive(Debug, Clone)]
pub struct Edge<T: EdgeWeight> {
    /// Source node index.
    pub src: NodeIdx,
    /// Destination node index.
    pub dest: NodeIdx,
    /// Weight of the edge.
    pub weight: T,
}

impl<T: EdgeWeight> PartialEq for Edge<T> {
    fn eq(&self, other: &Self) -> bool {
        self.src == other.src && self.dest == other.dest
    }
}

impl<T: EdgeWeight> Eq for Edge<T> {}

impl<T: EdgeWeight> Hash for Edge<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.src.hash(state);
        self.dest.hash(state);
    }
}

impl<T: EdgeWeight> Edge<T> {
    /// Creates a new edge with the given source, destination, and optional weight.
    /// If weight is `None`, uses `T::one()` as the default.
    pub fn new(src: NodeIdx, dest: NodeIdx, weight: Option<T>) -> Self {
        Self {
            src,
            dest,
            weight: weight.unwrap_or_else(T::one),
        }
    }
}

/// Converts an iterator of (src, dest) tuples into an iterator of `Edge<T>` with default weights.
///
/// # Arguments
/// * `edges` - An iterator of (NodeIdx, NodeIdx) tuples representing edges.
pub fn edges_from_tuples<I, T: EdgeWeight>(edges: I) -> impl Iterator<Item = Edge<T>>
where
    I: IntoIterator<Item = (NodeIdx, NodeIdx)>,
{
    edges
        .into_iter()
        .map(|(src, dest)| Edge::new(src, dest, None))
}
