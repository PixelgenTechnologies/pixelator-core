use crate::common::panel::{MarkerMatch, PanelSearcher};
use crate::edgelist::read_layout::ConjugateSide;

use super::drop_reason::DropReason;

// Split from the [`ConjugateSide`] definition because it needs the panel and
// the extraction failure vocabulary, neither of which `read_layout` may depend
// on. Rustdoc still lists this method on the type, so the split costs nothing
// in discoverability.

/// Panel matching for a conjugate.
impl ConjugateSide {
    /// Resolves an encoded pid `seq` to its panel marker index for this conjugate.
    ///
    /// Conjugate A searches the panel's `sequence_1` column, conjugate B
    /// `sequence_2`. A pid absent from the panel yields this conjugate's miss
    /// reason ([`DropReason::Pid1NotInPanel`] / [`DropReason::Pid2NotInPanel`]);
    /// a tie between two or more markers yields the ambiguous reason
    /// ([`DropReason::Pid1Ambiguous`] / [`DropReason::Pid2Ambiguous`]).
    pub fn lookup(&self, searcher: &PanelSearcher, seq: u64) -> Result<i32, DropReason> {
        let (result, miss, ambiguous) = match self {
            ConjugateSide::A => (
                searcher.search_seq1(seq),
                DropReason::Pid1NotInPanel,
                DropReason::Pid1Ambiguous,
            ),
            ConjugateSide::B => (
                searcher.search_seq2(seq),
                DropReason::Pid2NotInPanel,
                DropReason::Pid2Ambiguous,
            ),
        };
        match result {
            MarkerMatch::Hit(index) => Ok(index),
            MarkerMatch::Miss => Err(miss),
            MarkerMatch::Ambiguous => Err(ambiguous),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Panel;
    use crate::common::panel::PanelMarker;
    use crate::encoding::encode_dna_2bit;

    #[test]
    fn conjugate_side_looks_up_marker_index_by_sequence() {
        let seq1 = "ACGTACGTAC";
        let seq2 = "TGCATGCATG";

        let markers = vec![PanelMarker {
            marker_id: String::from("SYNTH"),
            sequence_1: seq1.to_string(),
            sequence_2: seq2.to_string(),
        }];
        let panel = Panel::from_markers(markers).unwrap();
        let searcher = PanelSearcher::new(panel, 0);

        let side_a = ConjugateSide::A;
        let side_b = ConjugateSide::B;

        assert_eq!(
            side_a.lookup(&searcher, encode_dna_2bit(seq1.as_bytes())),
            Ok(0)
        );
        assert_eq!(
            side_a.lookup(&searcher, encode_dna_2bit(seq2.as_bytes())),
            Err(DropReason::Pid1NotInPanel)
        );

        assert_eq!(
            side_b.lookup(&searcher, encode_dna_2bit(seq2.as_bytes())),
            Ok(0)
        );
        assert_eq!(
            side_b.lookup(&searcher, encode_dna_2bit(seq1.as_bytes())),
            Err(DropReason::Pid2NotInPanel)
        );
    }

    #[test]
    fn ambiguous_pid_drops_as_pid1_ambiguous() {
        // Two markers at Hamming distance 4 on sequence_1 so a midpoint query
        // is distance 2 from both; under threshold 2 that is Ambiguous.
        let panel = Panel::from_markers(vec![
            PanelMarker {
                marker_id: String::from("A"),
                sequence_1: String::from("AAAAAAAAAA"),
                sequence_2: String::from("GGGGGGGGGG"),
            },
            PanelMarker {
                marker_id: String::from("B"),
                sequence_1: String::from("TTTTAAAAAA"),
                sequence_2: String::from("CCCCCCCCCC"),
            },
        ])
        .expect("build panel from markers");
        let searcher = PanelSearcher::new(panel, 2);

        let query = encode_dna_2bit(b"TTAAAAAAAA");
        assert_eq!(
            ConjugateSide::A.lookup(&searcher, query),
            Err(DropReason::Pid1Ambiguous)
        );
    }

    #[test]
    fn ambiguous_pid_drops_as_pid2_ambiguous() {
        // Same midpoint setup, but on sequence_2 for conjugate B.
        let panel = Panel::from_markers(vec![
            PanelMarker {
                marker_id: String::from("A"),
                sequence_1: String::from("GGGGGGGGGG"),
                sequence_2: String::from("AAAAAAAAAA"),
            },
            PanelMarker {
                marker_id: String::from("B"),
                sequence_1: String::from("CCCCCCCCCC"),
                sequence_2: String::from("TTTTAAAAAA"),
            },
        ])
        .expect("build panel from markers");
        let searcher = PanelSearcher::new(panel, 2);

        let query = encode_dna_2bit(b"TTAAAAAAAA");
        assert_eq!(
            ConjugateSide::B.lookup(&searcher, query),
            Err(DropReason::Pid2Ambiguous)
        );
    }
}
