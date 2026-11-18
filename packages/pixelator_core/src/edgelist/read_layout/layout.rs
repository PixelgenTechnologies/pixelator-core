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

impl SideBounds {
    /// Offset of the furthest region end from the read start.
    fn end(&self) -> usize {
        self.pid.1.max(self.umi.1)
    }

    /// Offset of the nearest region start from the read start.
    fn start(&self) -> usize {
        self.pid.0.min(self.umi.0)
    }

    /// These bounds moved by `end - amplicon_len`, i.e. re-measured from an
    /// amplicon that ends at read position `end`. Callers ensure `end` is
    /// large enough that no region moves before the read start.
    fn ending_at(&self, amplicon_len: usize, end: usize) -> Self {
        let shift = |(s, e): (usize, usize)| (s + end - amplicon_len, e + end - amplicon_len);
        Self {
            pid: shift(self.pid),
            umi: shift(self.umi),
        }
    }
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
    amplicon_len: usize,
    /// Per strand: the shortest read whose two conjugates' regions fit without
    /// overlapping when one is read from each end.
    min_single_read_len: [usize; 2],
}

impl RegionLayout {
    /// Resolves the `pid`/`umi` offsets of both conjugates in both orientations.
    ///
    /// # Panics
    ///
    /// Panics if the assay lacks the `amplicon` or one of the `pid`/`umi`
    /// regions. [`Assay`] validation requires all five, so a successfully loaded assay cannot
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

        let amplicon_len = assay
            .forward_bounds("amplicon")
            .map(|(_, end)| end)
            .expect("amplicon region in assay");
        let mut min_single_read_len = [0; 2];
        for strand in [Strand::Forward, Strand::Reverse] {
            let near = bounds[Self::near_side(strand) as usize][strand as usize];
            let far = bounds[Self::far_side(strand) as usize][strand as usize];
            min_single_read_len[strand as usize] = near.end() + amplicon_len - far.start();
        }

        Self {
            bounds,
            amplicon_len,
            min_single_read_len,
        }
    }

    /// The conjugate at the 5' end of a single read of orientation `strand`.
    pub(crate) fn near_side(strand: Strand) -> ConjugateSide {
        match strand {
            Strand::Forward => ConjugateSide::A,
            Strand::Reverse => ConjugateSide::B,
        }
    }

    /// The conjugate at the 3' end of a single read of orientation `strand`.
    pub(crate) fn far_side(strand: Strand) -> ConjugateSide {
        match strand {
            Strand::Forward => ConjugateSide::B,
            Strand::Reverse => ConjugateSide::A,
        }
    }

    /// Design length of the amplicon.
    pub(crate) fn amplicon_len(&self) -> usize {
        self.amplicon_len
    }

    /// Offsets of `side`'s regions within a read of orientation `strand`.
    pub(crate) fn get(&self, side: ConjugateSide, strand: Strand) -> SideBounds {
        self.bounds[side as usize][strand as usize]
    }

    /// Offsets of `side`'s regions in a single read of orientation `strand`
    /// whose amplicon ends at position `end` instead of at its design length.
    ///
    /// Meant for the conjugate at the read's 3' end (see
    /// [`RegionLayout::far_side`]), which an indel between the conjugates
    /// shifts relative to the read start but not relative to the amplicon end.
    /// Returns `None` if `end` is too short for both conjugates to fit.
    pub(crate) fn get_ending_at(
        &self,
        side: ConjugateSide,
        strand: Strand,
        end: usize,
    ) -> Option<SideBounds> {
        (end >= self.min_single_read_len[strand as usize])
            .then(|| self.get(side, strand).ending_at(self.amplicon_len, end))
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

    #[test]
    fn min_single_read_len_fits_one_pid_and_umi_per_end() {
        let assay = proxiome_v2_assay();
        let layout = RegionLayout::new(&assay);
        let (pid_start, pid_end) = assay.forward_bounds("pid-1").expect("pid-1 bounds");
        let (umi_start, umi_end) = assay.forward_bounds("umi-1").expect("umi-1 bounds");
        let per_side = (pid_end - pid_start) + (umi_end - umi_start);

        for strand in [Strand::Forward, Strand::Reverse] {
            let far = RegionLayout::far_side(strand);
            assert!(
                layout
                    .get_ending_at(far, strand, 2 * per_side - 1)
                    .is_none()
            );
            assert!(layout.get_ending_at(far, strand, 2 * per_side).is_some());
        }
    }

    /// At the design length the end-anchored offsets are the fixed ones, and
    /// any other end moves both regions by the same amount.
    #[test]
    fn ending_at_shifts_far_side_with_the_amplicon_end() {
        let assay = proxiome_v2_assay();
        let layout = RegionLayout::new(&assay);
        let amplicon_len = layout.amplicon_len();

        for strand in [Strand::Forward, Strand::Reverse] {
            let far = RegionLayout::far_side(strand);
            let fixed = layout.get(far, strand);

            let at_design = layout
                .get_ending_at(far, strand, amplicon_len)
                .expect("design length fits");
            assert_eq!(at_design.pid, fixed.pid);
            assert_eq!(at_design.umi, fixed.umi);

            let shorter = layout
                .get_ending_at(far, strand, amplicon_len - 3)
                .expect("3 bases short still fits");
            assert_eq!(shorter.pid, (fixed.pid.0 - 3, fixed.pid.1 - 3));
            assert_eq!(shorter.umi, (fixed.umi.0 - 3, fixed.umi.1 - 3));
        }
    }
}
