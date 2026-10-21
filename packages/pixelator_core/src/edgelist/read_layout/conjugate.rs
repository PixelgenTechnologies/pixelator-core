/// Which of a molecule's two conjugates a region belongs to.
///
/// A molecule pairs two antibody-oligo conjugates, each with its own `pid`/`umi`
/// regions in the assay and its own sequence column in the panel.
/// `ConjugateSide` selects the matching region ids so one primitive can address
/// either conjugate.
///
/// Resolving a pid against the panel is deliberately *not* here: that needs the
/// panel and the extraction failure vocabulary, so it lives with extraction (see
/// `ConjugateSide::lookup`) and this module stays free of both.
#[derive(Clone, Copy, Debug)]
pub enum ConjugateSide {
    /// The first conjugate (`pid-1`/`umi-1`, panel `sequence_1`).
    A,
    /// The second conjugate (`pid-2`/`umi-2`, panel `sequence_2`).
    B,
}

impl ConjugateSide {
    /// Assay region id of this conjugate's protein-id (marker) barcode.
    pub fn pid_region(&self) -> &'static str {
        match self {
            ConjugateSide::A => "pid-1",
            ConjugateSide::B => "pid-2",
        }
    }

    /// Assay region id of this conjugate's UMI.
    pub fn umi_region(&self) -> &'static str {
        match self {
            ConjugateSide::A => "umi-1",
            ConjugateSide::B => "umi-2",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conjugate_side_maps_to_expected_region_ids() {
        let side_a = ConjugateSide::A;
        let side_b = ConjugateSide::B;

        assert_eq!(side_a.pid_region(), "pid-1");
        assert_eq!(side_b.pid_region(), "pid-2");
        assert_eq!(side_a.umi_region(), "umi-1");
        assert_eq!(side_b.umi_region(), "umi-2");
    }
}
