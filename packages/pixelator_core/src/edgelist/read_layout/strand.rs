use crate::common::Assay;
use crate::encoding::{encode_dna_2bit, encode_dna_2bit_rc};

/// The orientation of a read relative to the amplicon.
///
/// A `Forward` read runs 5'->3' along the amplicon (e.g. R1); a `Reverse` read
/// is what the sequencer emits from the other end, i.e. the reverse-complement
/// of the amplicon (e.g. R2). `Strand` lets the extraction primitives read a
/// region from either orientation with the same code path.
#[derive(Clone, Copy, Debug)]
pub enum Strand {
    /// Read in amplicon orientation (5'->3').
    Forward,
    /// Read is the reverse-complement of the amplicon.
    Reverse,
}

impl Strand {
    /// Byte offsets `(start, end)` of `region_id` within a read of this
    /// orientation, or `None` if the assay has no such region.
    ///
    /// Forward reads index the assay's forward bounds; reverse reads index the
    /// reverse bounds (measured from the read's own 5' end), so the returned
    /// range always slices directly into the raw read.
    pub fn bounds(&self, assay: &Assay, region_id: &str) -> Option<(usize, usize)> {
        match self {
            Self::Forward => assay.forward_bounds(region_id),
            Self::Reverse => assay.reverse_bounds(region_id),
        }
    }

    /// 2-bit-encodes a region slice into its amplicon-forward encoding.
    ///
    /// Forward reads encode the bases as-is; reverse reads are
    /// reverse-complemented first, so both orientations yield the same code for
    /// the same molecule and panel lookups stay orientation-agnostic.
    pub fn encode(&self, dna: &[u8]) -> u64 {
        match self {
            Self::Forward => encode_dna_2bit(dna),
            Self::Reverse => encode_dna_2bit_rc(dna),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::proxiome_v2_assay;

    #[test]
    fn strand_bounds_have_expected_orientation() {
        let assay = proxiome_v2_assay();
        let f_strand = Strand::Forward;
        let r_strand = Strand::Reverse;

        assert_eq!(f_strand.bounds(&assay, "umi-1"), Some((0, 28)));
        assert_eq!(r_strand.bounds(&assay, "umi-1"), Some((105, 133)));
    }

    #[test]
    fn strand_encoding_has_expected_orientation() {
        let f_strand = Strand::Forward;
        let r_strand = Strand::Reverse;

        let dna = b"AACC";
        assert_eq!(f_strand.encode(dna), 80);
        assert_eq!(r_strand.encode(dna), 250);
    }
}
