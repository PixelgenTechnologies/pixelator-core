use crate::common::Assay;
use crate::common::assay::RegionType;
use crate::common::distances::within_bounded_levenshtein;
use crate::encoding::{encode_dna_2bit, encode_dna_2bit_rc};

/// Flags UMIs contaminated by a linker binding site (LBS).
///
/// A UMI is contaminated when the whole (canonical, 2-bit-packed) UMI aligns to
/// any substring of an LBS reference — or its reverse complement — within
/// `max_edits` edits. Each retained LBS region contributes two packed
/// references `(bits, len)`, one per orientation.
///
/// When the filter is disabled (`max_edits` is `None`) or the assay has no
/// usable LBS references, `references` is empty and [`search`](Self::search)
/// always returns `false`.
#[derive(Clone)]
pub struct LbsSearcher {
    max_edits: u8,
    umi_len: usize,
    /// Packed LBS sequences (`encode_dna_2bit` / `_rc`) with lengths, one entry
    /// per retained orientation.
    references: Vec<(u64, usize)>,
}

impl LbsSearcher {
    /// Builds an LBS searcher from the assay's LBS regions.
    ///
    /// `None` for `max_edits` disables the filter: no references are packed and
    /// [`search`](Self::search) always returns `false`.
    ///
    /// References shorter than `umi_len - max_edits` are skipped: a UMI of
    /// `umi_len` bases needs at least `umi_len - ref.len()` deletions to fit
    /// inside a shorter reference, which already exceeds the budget.
    ///
    /// When the filter is enabled but no usable LBS references remain (none in
    /// the assay, or all too short), a warning is logged and the searcher stays
    /// empty.
    ///
    /// Panics if `umi_len` or a retained LBS length exceeds 32 (the 2-bit `u64`
    /// packing limit).
    pub fn new(assay: &Assay, max_edits: Option<u8>) -> Self {
        let umi_len = assay
            .region_sequences_by_type(RegionType::Umi)
            .iter()
            .map(|seq| seq.len())
            .max()
            .expect("No UMI sequences found in assay");
        assert!(
            umi_len <= 32,
            "UMI length {umi_len} exceeds the 32-base 2-bit packing limit"
        );

        let Some(max_edits) = max_edits else {
            return Self {
                max_edits: 0,
                umi_len,
                references: Vec::new(),
            };
        };

        let min_ref_len = umi_len.saturating_sub(max_edits as usize);
        let mut references = Vec::new();
        for lbs in assay.region_sequences_by_type(RegionType::Lbs) {
            if lbs.len() < min_ref_len {
                continue;
            }
            assert!(
                lbs.len() <= 32,
                "LBS length {} exceeds the 32-base 2-bit packing limit",
                lbs.len()
            );
            let lbs_bytes = lbs.as_bytes();
            references.push((encode_dna_2bit(lbs_bytes), lbs.len()));
            // The reverse complement reference is used here since it has also
            // been observed as a potential contaminant in the UMIs.
            references.push((encode_dna_2bit_rc(lbs_bytes), lbs.len()));
        }

        if references.is_empty() {
            log::warn!(
                "LBS-overlap filter is enabled (max_edits={max_edits}) but the assay \
                 has no LBS regions long enough to host a {umi_len}-base UMI \
                 (need at least {min_ref_len} bases); the filter will never drop a read"
            );
        }

        Self {
            max_edits,
            umi_len,
            references,
        }
    }

    /// Returns whether the canonical, 2-bit-packed `umi` is within `max_edits` of
    /// any retained LBS reference or its reverse complement.
    pub fn search(&self, umi: u64) -> bool {
        self.references.iter().any(|&(text, text_len)| {
            within_bounded_levenshtein(umi, self.umi_len, text, text_len, u32::from(self.max_edits))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;
    use crate::encoding::{encode_dna_2bit, encode_dna_2bit_rc};
    use crate::test_fixtures::proxiome_v2_assay;

    const LBS1: &[u8] = b"CAAGTGACGCTGGTCAAACACTCATGTC";

    fn searcher(max_edits: u8) -> LbsSearcher {
        LbsSearcher::new(&proxiome_v2_assay(), Some(max_edits))
    }

    /// Reverse-complements an ACGT sequence.
    fn revcomp(seq: &[u8]) -> Vec<u8> {
        seq.iter()
            .rev()
            .map(|&b| match b {
                b'A' => b'T',
                b'C' => b'G',
                b'G' => b'C',
                b'T' => b'A',
                other => other,
            })
            .collect()
    }

    #[test]
    fn search_hits_exact_forward_lbs1() {
        assert!(searcher(6).search(encode_dna_2bit(LBS1)));
    }

    #[test]
    fn search_hits_reverse_complement_lbs1() {
        // A UMI that is the reverse complement of lbs-1 matches the RC reference
        // exactly, so its forward (canonical) encoding must be caught.
        let umi = revcomp(LBS1);
        assert!(searcher(6).search(encode_dna_2bit(&umi)));
        // The same molecule read on the reverse strand encodes RC-first; both
        // canonical encodings are contaminated.
        assert!(searcher(6).search(encode_dna_2bit_rc(LBS1)));
    }

    #[test]
    fn search_misses_near_miss_at_budget_plus_one() {
        // lbs-1 with seven spread-out substitutions: infix distance exactly 7 to
        // lbs-1 (and further to its RC), so it is caught at k=7 but not k=6.
        let umi = b"GAAGAGACTCTGTTCACACAGTCAAGTC";
        assert!(!searcher(6).search(encode_dna_2bit(umi)));
        assert!(searcher(7).search(encode_dna_2bit(umi)));
    }

    #[test]
    fn search_misses_random_umi() {
        let umi = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        assert!(!searcher(6).search(encode_dna_2bit(umi)));
    }

    #[test]
    fn short_references_are_skipped() {
        // proxiome-v2 has lbs-1 (28 bp, kept) and lbs-2 (14 bp, too short for a
        // 28 bp UMI within 6 edits), so only lbs-1's two orientations remain.
        assert_eq!(searcher(6).references.len(), 2);
    }

    #[test]
    fn new_returns_empty_for_lbs_less_assay() {
        let yaml = "\
name: no-lbs
assay_spec:
  - region_id: amplicon
    region_type: amplicon
    name: amplicon
    sequence_type: joined
    regions:
      - region_id: umi-1
        region_type: umi
        name: umi 1
        sequence_type: random
        sequence: NNNNNNNNNNNNNNNNNNNNNNNNNNNN
        min_len: 28
        max_len: 28
      - region_id: pid-1
        region_type: pid
        name: pid 1
        sequence_type: random
        sequence: NNNNNNNNNN
        min_len: 10
        max_len: 10
      - region_id: pid-2
        region_type: pid
        name: pid 2
        sequence_type: random
        sequence: NNNNNNNNNN
        min_len: 10
        max_len: 10
      - region_id: umi-2
        region_type: umi
        name: umi 2
        sequence_type: random
        sequence: NNNNNNNNNNNNNNNNNNNNNNNNNNNN
        min_len: 28
        max_len: 28
";
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(yaml.as_bytes()).expect("write yaml");
        file.flush().expect("flush");
        let assay = Assay::from_path(file.path()).expect("load assay");

        let s = LbsSearcher::new(&assay, Some(6));
        assert!(s.references.is_empty());
        assert!(!s.search(encode_dna_2bit(LBS1)));
    }

    #[test]
    fn disabled_filter_never_matches() {
        let s = LbsSearcher::new(&proxiome_v2_assay(), None);
        assert!(s.references.is_empty());
        assert!(!s.search(encode_dna_2bit(LBS1)));
    }

    /// Fidelity against the known contaminant list. That list is itself the
    /// output of a k=6 infix search against this same lbs-1 reference (684 of 818
    /// sit exactly at distance 6, none beyond), so this pins the reimplementation
    /// to the prior rule — it is a regression guard, not independent biological
    /// ground truth, of which none exists today.
    #[test]
    fn all_known_contaminant_umis_are_flagged_at_k6() {
        let s = searcher(6);
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/test_fixtures/contaminant_umis.txt"
        ));

        let mut count = 0usize;
        for line in fixture.lines() {
            let umi = line.trim();
            if umi.is_empty() {
                continue;
            }
            assert_eq!(umi.len(), 28, "expected a 28 bp UMI, got {umi:?}");
            assert!(
                s.search(encode_dna_2bit(umi.as_bytes())),
                "known contaminant UMI {umi} should be flagged at k=6"
            );
            count += 1;
        }
        assert_eq!(count, 818, "expected 818 unique contaminant UMIs");
    }

    /// The flip side of the fidelity test: a large seeded pool of random UMIs is
    /// essentially never flagged at k=6, so the filter is specific to the linker.
    #[test]
    fn seeded_random_umis_are_not_flagged_at_k6() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        const BASES: [u8; 4] = *b"ACGT";
        let s = searcher(6);
        let mut rng = StdRng::seed_from_u64(0x50D_C0FF_EE00_1234);

        let mut hits = 0usize;
        for _ in 0..10_000 {
            let umi: Vec<u8> = (0..28).map(|_| BASES[rng.random_range(0..4)]).collect();
            if s.search(encode_dna_2bit(&umi)) {
                hits += 1;
            }
        }
        assert_eq!(
            hits, 0,
            "no random UMI should be flagged at k=6, got {hits}"
        );
    }
}
