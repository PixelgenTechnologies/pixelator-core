/// Why a read pair was dropped during extraction.
///
/// Only the drop points that exist today are listed; future filters add
/// variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DropReason {
    /// Either read contained an `N` base in the informative regions.
    HasN,
    /// The `pid-1` barcode was not found in the panel.
    Pid1NotInPanel,
    /// The `pid-2` barcode was not found in the panel.
    Pid2NotInPanel,
    /// The `pid-1` barcode matched two or more markers ambiguously.
    Pid1Ambiguous,
    /// The `pid-2` barcode matched two or more markers ambiguously.
    Pid2Ambiguous,
    /// The read is too short to contain all necessary regions
    TooShort,
    /// The read's average Phred quality is below the configured threshold in the informative regions.
    LowQuality,
    /// The UMI is low-complexity: one base accounts for more than the configured
    /// fraction of its length.
    LowComplexity,
    /// The UMI aligns too closely to a linker binding site (LBS) sequence,
    /// suggesting LBS contamination of the UMI region.
    LbsOverlap,
}

impl DropReason {
    /// Every variant, so a report can list the reasons that never fired too.
    pub const ALL: [DropReason; 9] = [
        DropReason::HasN,
        DropReason::Pid1NotInPanel,
        DropReason::Pid2NotInPanel,
        DropReason::Pid1Ambiguous,
        DropReason::Pid2Ambiguous,
        DropReason::TooShort,
        DropReason::LowQuality,
        DropReason::LowComplexity,
        DropReason::LbsOverlap,
    ];

    /// The reason's snake_case name
    pub fn as_str(&self) -> &'static str {
        // Renaming one changes the report schema, so treat these as fixed.
        match self {
            DropReason::HasN => "has_n",
            DropReason::Pid1NotInPanel => "pid1_not_in_panel",
            DropReason::Pid2NotInPanel => "pid2_not_in_panel",
            DropReason::Pid1Ambiguous => "pid1_ambiguous",
            DropReason::Pid2Ambiguous => "pid2_ambiguous",
            DropReason::TooShort => "too_short",
            DropReason::LowQuality => "low_quality",
            DropReason::LowComplexity => "low_complexity",
            DropReason::LbsOverlap => "lbs_overlap",
        }
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashSet as HashSet;

    use super::*;

    #[test]
    fn every_variant_reports_its_snake_case_name() {
        for reason in DropReason::ALL {
            let expected = match reason {
                DropReason::HasN => "has_n",
                DropReason::Pid1NotInPanel => "pid1_not_in_panel",
                DropReason::Pid2NotInPanel => "pid2_not_in_panel",
                DropReason::Pid1Ambiguous => "pid1_ambiguous",
                DropReason::Pid2Ambiguous => "pid2_ambiguous",
                DropReason::TooShort => "too_short",
                DropReason::LowQuality => "low_quality",
                DropReason::LowComplexity => "low_complexity",
                DropReason::LbsOverlap => "lbs_overlap",
            };
            assert_eq!(reason.as_str(), expected);
        }
    }

    #[test]
    fn names_are_unique() {
        let names: HashSet<&str> = DropReason::ALL.iter().map(|r| r.as_str()).collect();

        assert_eq!(names.len(), DropReason::ALL.len());
    }

    /// Number of variants in the DropReason enum. Must be updated each time a
    /// variant is added or removed from DropReason.
    ///
    /// In practice, this should always be the same as `DropReason::ALL.len()`,
    /// as long as one remembers to update ALL when variants are added or removed.
    ///
    /// If one forgets, the test below will trigger either a compilation error or
    /// fail.
    const N_VARIANTS: usize = 9;

    /// The column claimed by a single variant.
    const fn column(index: usize) -> [usize; N_VARIANTS] {
        let mut claimed = [0; N_VARIANTS];
        claimed[index] = 1;
        claimed
    }

    #[test]
    fn all_lists_every_variant_exactly_once() {
        // Each arm must claim a column of its own, and each index must stay
        // inside a `const` block: that is what turns a new variant into a
        // compile error until `N_VARIANTS` grows, and a `N_VARIANTS` that outgrows
        // `ALL` into the failure below. Calling `column` outside `const` moves
        // the bounds check to runtime and might never happen if the variant is
        // missing from `ALL`.
        let mut tally = [0usize; N_VARIANTS];
        for reason in DropReason::ALL {
            let claimed = match reason {
                DropReason::HasN => const { column(0) },
                DropReason::Pid1NotInPanel => const { column(1) },
                DropReason::Pid2NotInPanel => const { column(2) },
                DropReason::Pid1Ambiguous => const { column(3) },
                DropReason::Pid2Ambiguous => const { column(4) },
                DropReason::TooShort => const { column(5) },
                DropReason::LowQuality => const { column(6) },
                DropReason::LowComplexity => const { column(7) },
                DropReason::LbsOverlap => const { column(8) },
            };
            for (total, hit) in tally.iter_mut().zip(claimed) {
                *total += hit;
            }
        }

        assert!(
            tally.iter().all(|&count| count == 1),
            "every variant must appear in DropReason::ALL exactly once, got {tally:?}"
        );
    }
}
