use crate::common::Assay;

use super::{ConjugateSide, Strand};

/// Byte offsets of one conjugate's `pid` and `umi` regions within a raw read.
///
/// Both ranges slice directly into the read as the sequencer emitted it, so a
/// reverse read needs no reverse-complementing before indexing (see
/// [`Strand::bounds`]).
#[derive(Clone, Copy, Default)]
pub(crate) struct SideBounds {
    pub pid: (usize, usize),
    pub umi: (usize, usize),
}

/// Region offsets for every (conjugate, orientation) pair, resolved once.
///
/// Where a region sits depends on *both* axes: [`ConjugateSide`] picks which
/// region ids to look up (`pid-1`/`umi-1` vs `pid-2`/`umi-2`), and [`Strand`]
/// picks whether they are measured from the amplicon's 5' end or from the raw
/// reverse read's. Neither varies per read, so all four combinations are
/// precomputed here and the hot path indexes this table instead of doing a
/// string-keyed assay lookup for every region of every read.
///
/// Cells are addressed by the enums' discriminants; both enums have exactly two
/// variants, so adding one means growing `bounds` to match.
#[derive(Clone)]
pub(crate) struct RegionLayout {
    bounds: [[SideBounds; 2]; 2],
}

impl RegionLayout {
    /// Resolves the `pid`/`umi` offsets of both conjugates in both orientations.
    ///
    /// # Panics
    ///
    /// Panics if the assay lacks one of the `pid`/`umi` regions. [`Assay`]
    /// validation requires all four, so a successfully loaded assay cannot
    /// trigger this; resolving eagerly keeps the per-read path free of a
    /// missing-region branch.
    pub(crate) fn new(assay: &Assay) -> Self {
        let build = |side: ConjugateSide, strand: Strand| SideBounds {
            pid: {
                let pid_region = side.pid_region();
                strand
                    .bounds(assay, pid_region)
                    .unwrap_or_else(|| panic!("{pid_region} region not in assay"))
            },
            umi: {
                let umi_region = side.umi_region();
                strand
                    .bounds(assay, umi_region)
                    .unwrap_or_else(|| panic!("{umi_region} region not in assay"))
            },
        };

        let mut bounds = [[SideBounds::default(); 2]; 2];
        for side in [ConjugateSide::A, ConjugateSide::B] {
            for strand in [Strand::Forward, Strand::Reverse] {
                bounds[side as usize][strand as usize] = build(side, strand);
            }
        }

        Self { bounds }
    }

    /// Offsets of `side`'s regions within a read of orientation `strand`.
    pub(crate) fn get(&self, side: ConjugateSide, strand: Strand) -> SideBounds {
        self.bounds[side as usize][strand as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::proxiome_v2_assay;

    #[test]
    fn every_cell_matches_the_assay_bounds_for_its_region() {
        let assay = proxiome_v2_assay();
        let layout = RegionLayout::new(&assay);

        // The region ids each (side, strand) pair must resolve to.
        let cases = [
            (ConjugateSide::A, Strand::Forward, "pid-1", "umi-1"),
            (ConjugateSide::A, Strand::Reverse, "pid-1", "umi-1"),
            (ConjugateSide::B, Strand::Forward, "pid-2", "umi-2"),
            (ConjugateSide::B, Strand::Reverse, "pid-2", "umi-2"),
        ];

        for (side, strand, pid_id, umi_id) in cases {
            let expected = |region_id: &str| {
                match strand {
                    Strand::Forward => assay.forward_bounds(region_id),
                    Strand::Reverse => assay.reverse_bounds(region_id),
                }
                .expect("region in fixture assay")
            };

            let got = layout.get(side, strand);
            assert_eq!(got.pid, expected(pid_id), "{side:?}/{strand:?} pid");
            assert_eq!(got.umi, expected(umi_id), "{side:?}/{strand:?} umi");
        }
    }

    /// A table addressed by two discriminants is easy to get wrong (every cell
    /// aliased, or one axis silently ignored), so both axes must move the offsets.
    #[test]
    fn cells_are_keyed_by_both_side_and_strand() {
        let assay = proxiome_v2_assay();
        let layout = RegionLayout::new(&assay);

        let a_fwd = layout.get(ConjugateSide::A, Strand::Forward);
        let a_rev = layout.get(ConjugateSide::A, Strand::Reverse);
        let b_fwd = layout.get(ConjugateSide::B, Strand::Forward);

        assert_ne!(a_fwd.pid, a_rev.pid, "strand must change the offsets");
        assert_ne!(a_fwd.pid, b_fwd.pid, "side must change the offsets");
    }
}
