/// A validated edgelist row, ready to be pushed into the
/// [`RowBuffer`](super::super::row_buffer::RowBuffer).
///
/// Marker fields are panel indices (`i32`), not sequences or ids: the panel
/// lookup resolves each pid to its dictionary index, which the buffer stores
/// directly and dictionary-encodes at flush.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub marker_idx1: i32,
    pub marker_idx2: i32,
    pub umi1: u64,
    pub umi2: u64,
}
