mod drop_reason;
mod filters;
mod lbs_searcher;
mod lookup;
mod row;

use paraseq::Record;

use crate::common::panel::PanelSearcher;
use crate::common::{Assay, Panel};

use super::read_layout::{ConjugateSide, Read, RegionLayout, Strand};

pub use drop_reason::DropReason;
pub use filters::{DEFAULT_LBS_OVERLAP_MAX_EDITS, FilterConfig};
pub use lbs_searcher::LbsSearcher;
pub use row::Row;

use filters::{avg_phred_below, has_n, umi_base_freq_exceeds};

/// Decides whether a read yields an edgelist row, and builds the row if it does.
///
/// This is the pipeline's keep-or-drop gate. [`extract`](Extractor::extract)
/// takes one raw read (single-end) or read pair (paired-end) and returns either
/// a [`Row`] holding both conjugates' marker indices and UMIs, or the
/// [`DropReason`] that disqualified it. Checks run in a fixed order and the
/// first failure wins, so a read breaking several rules always reports the same
/// reason.
#[derive(Clone)]
pub struct Extractor {
    layout: RegionLayout,
    panel_searcher: PanelSearcher,
    filters: FilterConfig,
    lbs_searcher: LbsSearcher,
}

impl Extractor {
    pub fn new(assay: &Assay, panel: Panel, filters: FilterConfig) -> Self {
        Self {
            layout: RegionLayout::new(assay),
            panel_searcher: PanelSearcher::new(panel, filters.mismatch_threshold.unwrap_or(0)),
            lbs_searcher: LbsSearcher::new(assay, filters.lbs_overlap_max_edits),
            filters,
        }
    }

    /// Validates and extracts a read (single-end) or read pair (paired-end) into
    /// a [`Row`], or reports the [`DropReason`] for which it is discarded.
    ///
    /// Each conjugate is resolved by [`Extractor::extract_side`]: paired reads
    /// take conjugate A from R1 (forward strand) and conjugate B from R2 (reverse
    /// strand); single-end reads take both conjugates from the one read at its
    /// strand.
    pub fn extract<R: Record>(&self, read: &Read<'_, R>) -> Result<Row, DropReason> {
        // Matching on `*read` binds the inner `&R`s directly; matching on `read`
        // would bind `&&R`, which `extract_side` cannot take as a record.
        let ((marker_idx1, umi1), (marker_idx2, umi2)) = match *read {
            Read::SingleForward(r) => (
                self.extract_side(r, Strand::Forward, ConjugateSide::A)?,
                self.extract_side(r, Strand::Forward, ConjugateSide::B)?,
            ),
            Read::SingleReverse(r) => (
                self.extract_side(r, Strand::Reverse, ConjugateSide::A)?,
                self.extract_side(r, Strand::Reverse, ConjugateSide::B)?,
            ),
            Read::Paired { r1, r2 } => (
                self.extract_side(r1, Strand::Forward, ConjugateSide::A)?,
                self.extract_side(r2, Strand::Reverse, ConjugateSide::B)?,
            ),
        };

        Ok(Row {
            marker_idx1,
            marker_idx2,
            umi1,
            umi2,
        })
    }

    /// Extracts one conjugate (marker index + UMI) from `r`, reading its
    /// `pid`/`umi` regions at `strand`'s bounds and orientation.
    ///
    /// Filters run in an order that tries to minimize work before dropping a
    /// read.
    fn extract_side<Rf: Record>(
        &self,
        r: &Rf,
        strand: Strand,
        side: ConjugateSide,
    ) -> Result<(i32, u64), DropReason> {
        // No quality metrics are reported here. This returns at the first problem
        // it finds, so it never sees enough of the read to measure one; that is
        // `QcProfiler`'s job, and it covers the reads dropped here too.
        let seq = r.seq();

        let (pid_start, pid_end) = self.layout.get(side, strand).pid;
        let pid_slice = seq.get(pid_start..pid_end).ok_or(DropReason::TooShort)?;

        let (umi_start, umi_end) = self.layout.get(side, strand).umi;
        let umi_slice = seq.get(umi_start..umi_end).ok_or(DropReason::TooShort)?;

        // Only the informative regions (pid + umi) are checked for `N`; linker/uei
        // bases are ignored. Checked before the quality gate so a read that is both
        // N-containing and low-quality still drops as `HasN`.
        if has_n(pid_slice) || has_n(umi_slice) {
            return Err(DropReason::HasN);
        }
        if let Some(min_q) = self.filters.min_avg_quality
            && let Some(qual) = r.qual()
            && avg_phred_below(
                &[
                    qual.get(pid_start..pid_end).unwrap_or(&[]),
                    qual.get(umi_start..umi_end).unwrap_or(&[]),
                ],
                min_q,
            )
        {
            return Err(DropReason::LowQuality);
        }
        let marker_idx = side.lookup(&self.panel_searcher, strand.encode(pid_slice))?;

        if let Some(max_freq) = self.filters.max_umi_base_freq
            && umi_base_freq_exceeds(umi_slice, max_freq)
        {
            return Err(DropReason::LowComplexity);
        }

        let umi = strand.encode(umi_slice);

        if self.lbs_searcher.search(umi) {
            return Err(DropReason::LbsOverlap);
        }

        Ok((marker_idx, umi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{
        OwnedRecord, ReadMixConfig, generate_read_pairs, generate_single_reads, proxiome_v2_assay,
        proxiome_v2_marker_sequences, proxiome_v2_panel,
    };

    fn extractor() -> Extractor {
        Extractor::new(
            &proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig::default(),
        )
    }

    /// Reverse-complements an ACGT sequence. Test-local so the cross-consistency
    /// test can synthesize the raw reverse read (what a sequencer emits from the
    /// amplicon 3' end) as the exact reverse-complement of the forward read.
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

    /// A fully valid generated pair is extracted into a `Row` whose marker
    /// indices round-trip back to real panel markers.
    #[test]
    fn paired_read_extracts_row() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_read_pairs(&assay, &markers, 100, mix, 1);

        let ex = extractor();
        let n = ex.panel_searcher.panel().markers().len() as i32;
        for (r1, r2) in &records {
            let read = Read::Paired { r1: &r1, r2: &r2 };
            let row = ex.extract(&read).expect("valid pair should extract");
            // Indices must be valid positions in the panel's marker list.
            assert!((0..n).contains(&row.marker_idx1));
            assert!((0..n).contains(&row.marker_idx2));
        }
    }

    #[test]
    fn single_forward_read_extracts_row() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_single_reads(&assay, &markers, 100, mix, 1, Strand::Forward);

        let ex = extractor();
        let n = ex.panel_searcher.panel().markers().len() as i32;
        for r in &records {
            let read = Read::SingleForward(&r);
            let row = ex.extract(&read).expect("valid pair should extract");
            // Indices must be valid positions in the panel's marker list.
            assert!((0..n).contains(&row.marker_idx1));
            assert!((0..n).contains(&row.marker_idx2));
        }
    }

    #[test]
    fn single_reverse_read_extracts_row() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_single_reads(&assay, &markers, 100, mix, 1, Strand::Reverse);

        let ex = extractor();
        let n = ex.panel_searcher.panel().markers().len() as i32;
        for r in &records {
            let read = Read::SingleReverse(&r);
            let row = ex.extract(&read).expect("valid pair should extract");
            // Indices must be valid positions in the panel's marker list.
            assert!((0..n).contains(&row.marker_idx1));
            assert!((0..n).contains(&row.marker_idx2));
        }
    }

    /// An `N` injected into either read yields `DropReason::HasN`.
    #[test]
    fn paired_read_with_n_drops_as_has_n() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 1.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_read_pairs(&assay, &markers, 20, mix, 2);

        let ex = extractor();
        for (r1, r2) in &records {
            let read = Read::Paired { r1: &r1, r2: &r2 };
            assert_eq!(ex.extract(&read), Err(DropReason::HasN));
        }
    }

    #[test]
    fn single_forward_read_with_n_drops_as_has_n() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 1.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_single_reads(&assay, &markers, 20, mix, 2, Strand::Forward);

        let ex = extractor();
        for r in &records {
            let read = Read::SingleForward(&r);
            assert_eq!(ex.extract(&read), Err(DropReason::HasN));
        }
    }

    #[test]
    fn single_reverse_read_with_n_drops_as_has_n() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 1.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_single_reads(&assay, &markers, 20, mix, 2, Strand::Reverse);

        let ex = extractor();
        for r in &records {
            let read = Read::SingleReverse(&r);
            assert_eq!(ex.extract(&read), Err(DropReason::HasN));
        }
    }

    /// An `N` in a linker region (outside pid/umi) must not drop the read.
    #[test]
    fn n_outside_pid_umi_regions_is_kept() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let mut records = generate_single_reads(&assay, &markers, 1, mix, 1, Strand::Forward);
        let mut record = records.pop().expect("one read generated");
        let (lbs_start, _) = assay.forward_bounds("lbs-1").expect("lbs-1 bounds");
        record.seq[lbs_start] = b'N';
        let rec_ref = &record;

        let ex = extractor();
        assert!(
            ex.extract(&Read::SingleForward(&rec_ref)).is_ok(),
            "N in lbs-1 should be ignored"
        );
    }

    /// A pid that is absent from the panel drops at the pid-1 lookup.
    #[test]
    fn paired_read_with_unknown_pid_drops_at_lookup() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 1.0,
            read_len: 150,
        };
        let records = generate_read_pairs(&assay, &markers, 20, mix, 3);

        let ex = extractor();
        for (r1, r2) in &records {
            // Unknown pids are random bases; pid-1 is checked first, so the pair
            // drops there (with vanishing probability of a random panel hit).
            let read = Read::Paired { r1: &r1, r2: &r2 };
            assert_eq!(ex.extract(&read), Err(DropReason::Pid1NotInPanel));
        }
    }

    #[test]
    fn single_forward_read_with_unknown_pid_drops_at_lookup() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 1.0,
            read_len: 150,
        };
        let records = generate_single_reads(&assay, &markers, 20, mix, 3, Strand::Forward);

        let ex = extractor();
        for r in &records {
            // Unknown pids are random bases; pid-1 is checked first, so the pair
            // drops there (with vanishing probability of a random panel hit).
            let read = Read::SingleForward(&r);
            assert_eq!(ex.extract(&read), Err(DropReason::Pid1NotInPanel));
        }
    }

    #[test]
    fn single_reverse_read_with_unknown_pid_drops_at_lookup() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 1.0,
            read_len: 150,
        };
        let records = generate_single_reads(&assay, &markers, 20, mix, 3, Strand::Reverse);

        let ex = extractor();
        for r in &records {
            // Unknown pids are random bases; pid-1 is checked first, so the pair
            // drops there (with vanishing probability of a random panel hit).
            let read = Read::SingleReverse(&r);
            assert_eq!(ex.extract(&read), Err(DropReason::Pid1NotInPanel));
        }
    }

    /// End-to-end proof that extraction routes pid lookups through the
    /// `PanelSearcher`: a read whose pid-1 is one base off the panel is dropped
    /// under exact matching (threshold 0) but recovered to the same marker once
    /// the threshold allows a single mismatch. Only the searcher path can
    /// resolve the mutated barcode, so a pass here means it is wired in.
    #[test]
    fn near_match_pid_is_recovered_via_searcher() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let record = generate_single_reads(&assay, &markers, 1, mix, 1, Strand::Forward)
            .pop()
            .expect("one read generated");

        // Baseline: the exact extractor resolves the unmutated read.
        let exact = Extractor::new(
            &proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig::default(),
        );
        let rec_ref = &record;
        let baseline = exact
            .extract(&Read::SingleForward(&rec_ref))
            .expect("baseline read should extract");

        // Introduce a single mismatch at the start of the pid-1 region.
        let (pid_start, _) = assay.forward_bounds("pid-1").expect("pid-1 bounds");
        let mut mutated = OwnedRecord {
            id: record.id.clone(),
            seq: record.seq.clone(),
            qual: record.qual.clone(),
        };
        mutated.seq[pid_start] = if mutated.seq[pid_start] == b'A' {
            b'C'
        } else {
            b'A'
        };
        let mutated_ref = &mutated;

        // Under exact matching the mutated pid-1 is not in the panel.
        assert_eq!(
            exact.extract(&Read::SingleForward(&mutated_ref)),
            Err(DropReason::Pid1NotInPanel)
        );

        // With a threshold of 1 the searcher recovers it to the same marker.
        let fuzzy = Extractor::new(
            &proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig {
                mismatch_threshold: Some(1),
                ..Default::default()
            },
        );
        let recovered = fuzzy
            .extract(&Read::SingleForward(&mutated_ref))
            .expect("near-match read should be recovered");
        assert_eq!(recovered.marker_idx1, baseline.marker_idx1);
    }

    #[test]
    fn truncated_read_drops_as_too_short() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 100,
        };
        let records = generate_single_reads(&assay, &markers, 20, mix, 3, Strand::Forward);

        let ex = extractor();
        for r in &records {
            // Unknown pids are random bases; pid-1 is checked first, so the pair
            // drops there (with vanishing probability of a random panel hit).
            let read = Read::SingleForward(&r);
            assert_eq!(ex.extract(&read), Err(DropReason::TooShort));
        }
    }

    #[test]
    fn has_n_takes_precedence_over_too_short() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 1.0,
            frac_unknown_pid: 1.0,
            read_len: 100,
        };
        let records = generate_single_reads(&assay, &markers, 20, mix, 3, Strand::Forward);

        let ex = extractor();
        for r in &records {
            // Unknown pids are random bases; pid-1 is checked first, so the pair
            // drops there (with vanishing probability of a random panel hit).
            let read = Read::SingleForward(&r);
            assert_eq!(ex.extract(&read), Err(DropReason::HasN));
        }
    }

    #[test]
    fn single_and_paired_modes_agree_on_same_molecule() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();

        // Reads exactly as long as the amplicon: no tail, so revcomp is a
        // correctly framed raw reverse read.
        let (_, amplicon_len) = assay
            .forward_bounds("amplicon")
            .expect("amplicon region in assay");
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: amplicon_len,
        };
        let forward_reads = generate_single_reads(&assay, &markers, 50, mix, 7, Strand::Forward);

        let ex = extractor();
        for fwd in &forward_reads {
            let rev_owned = OwnedRecord {
                id: Vec::new(),
                seq: revcomp(&fwd.seq),
                qual: fwd.qual.iter().rev().copied().collect(),
            };
            let rev = &rev_owned;

            let row_fwd = ex
                .extract(&Read::SingleForward(&fwd))
                .expect("single-forward should extract");
            let row_rev = ex
                .extract(&Read::SingleReverse(&rev))
                .expect("single-reverse should extract");
            let row_paired = ex
                .extract(&Read::Paired { r1: &fwd, r2: &rev })
                .expect("paired should extract");

            assert_eq!(row_fwd, row_rev, "single-forward vs single-reverse");
            assert_eq!(row_fwd, row_paired, "single-forward vs paired");
        }
    }

    #[test]
    fn low_avg_quality_drops_as_low_quality() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let mut records = generate_single_reads(&assay, &markers, 1, mix, 1, Strand::Forward);
        let mut record = records.pop().expect("one read generated");
        // Q10 ('+') everywhere — below a min_avg_quality of 20.
        record.qual = vec![b'+'; record.seq.len()];
        let rec_ref = &record;

        let ex = Extractor::new(
            &proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig {
                min_avg_quality: Some(20),
                ..Default::default()
            },
        );
        assert_eq!(
            ex.extract(&Read::SingleForward(&rec_ref)),
            Err(DropReason::LowQuality)
        );
    }

    #[test]
    fn high_avg_quality_is_kept_when_threshold_set() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_single_reads(&assay, &markers, 5, mix, 1, Strand::Forward);
        // Fixtures use constant 'I' (Q40); threshold 30 should keep them.
        let ex = Extractor::new(
            &proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig {
                min_avg_quality: Some(30),
                ..Default::default()
            },
        );
        for r in &records {
            assert!(
                ex.extract(&Read::SingleForward(&r)).is_ok(),
                "Q40 read should pass min_avg_quality=30"
            );
        }
    }

    #[test]
    fn quality_filter_disabled_when_threshold_none() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let mut records = generate_single_reads(&assay, &markers, 1, mix, 1, Strand::Forward);
        let mut record = records.pop().expect("one read generated");
        record.qual = vec![b'+'; record.seq.len()]; // Q10
        let rec_ref = &record;

        let ex = extractor(); // min_avg_quality = None
        assert!(
            ex.extract(&Read::SingleForward(&rec_ref)).is_ok(),
            "low-quality read should be kept when filter is disabled"
        );
    }

    /// Only pid/umi bases count toward the quality average: a read with clean
    /// informative regions but noisy linker/flank bases must be kept.
    #[test]
    fn low_quality_outside_pid_umi_regions_is_kept() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let mut records = generate_single_reads(&assay, &markers, 1, mix, 1, Strand::Forward);
        let mut record = records.pop().expect("one read generated");

        // Whole read Q10 ('+'), then restore the four informative regions to Q40 ('I').
        record.qual = vec![b'+'; record.seq.len()];
        for id in ["pid-1", "umi-1", "pid-2", "umi-2"] {
            let (s, e) = assay.forward_bounds(id).expect("region bounds");
            for q in &mut record.qual[s..e] {
                *q = b'I';
            }
        }
        let rec_ref = &record;

        // Informative avg = Q40 (kept); whole-read avg ~= Q25 (< 30, would wrongly drop).
        let ex = Extractor::new(
            &proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig {
                min_avg_quality: Some(30),
                ..Default::default()
            },
        );
        assert!(
            ex.extract(&Read::SingleForward(&rec_ref)).is_ok(),
            "low quality outside pid/umi must not drop the read"
        );
    }

    #[test]
    fn has_n_takes_precedence_over_low_quality() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 1.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let mut records = generate_single_reads(&assay, &markers, 5, mix, 2, Strand::Forward);
        for r in &mut records {
            r.qual = vec![b'+'; r.seq.len()]; // also low quality
        }
        let ex = Extractor::new(
            &proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig {
                min_avg_quality: Some(20),
                ..Default::default()
            },
        );
        for r in &records {
            assert_eq!(
                ex.extract(&Read::SingleForward(&r)),
                Err(DropReason::HasN),
                "HasN is checked before LowQuality"
            );
        }
    }

    #[test]
    fn low_complexity_umi_drops() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let mut records = generate_single_reads(&assay, &markers, 1, mix, 1, Strand::Forward);
        let mut record = records.pop().expect("one read generated");
        let (umi_start, umi_end) = assay.forward_bounds("umi-1").expect("umi-1 bounds");
        for b in &mut record.seq[umi_start..umi_end] {
            *b = b'A';
        }
        let rec_ref = &record;

        let ex = Extractor::new(
            &proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig {
                max_umi_base_freq: Some(0.9),
                ..Default::default()
            },
        );
        assert_eq!(
            ex.extract(&Read::SingleForward(&rec_ref)),
            Err(DropReason::LowComplexity)
        );
    }

    #[test]
    fn diverse_umi_is_kept_when_threshold_set() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_single_reads(&assay, &markers, 20, mix, 1, Strand::Forward);
        let ex = Extractor::new(
            &proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig {
                max_umi_base_freq: Some(0.9),
                ..Default::default()
            },
        );
        for r in &records {
            assert!(
                ex.extract(&Read::SingleForward(&r)).is_ok(),
                "random UMI should pass max_umi_base_freq=0.9"
            );
        }
    }

    #[test]
    fn complexity_filter_disabled_when_none() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let mut records = generate_single_reads(&assay, &markers, 1, mix, 1, Strand::Forward);
        let mut record = records.pop().expect("one read generated");
        let (umi_start, umi_end) = assay.forward_bounds("umi-1").expect("umi-1 bounds");
        for b in &mut record.seq[umi_start..umi_end] {
            *b = b'A';
        }
        let rec_ref = &record;

        let ex = extractor(); // max_umi_base_freq = None
        assert!(
            ex.extract(&Read::SingleForward(&rec_ref)).is_ok(),
            "all-A UMI should be kept when complexity filter is disabled"
        );
    }

    #[test]
    fn lbs_overlap_filter_drops_lbs1_contaminated_umi() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let mut records = generate_single_reads(&assay, &markers, 1, mix, 1, Strand::Forward);
        let mut record = records.pop().expect("one read generated");
        let (umi_start, umi_end) = assay.forward_bounds("umi-1").expect("umi-1 bounds");
        let lbs1 = assay
            .region_sequence("lbs-1")
            .expect("lbs-1 sequence")
            .as_bytes();
        record.seq[umi_start..umi_end].copy_from_slice(lbs1);
        let rec_ref = &record;

        let ex = Extractor::new(
            &proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig {
                lbs_overlap_max_edits: Some(6),
                ..Default::default()
            },
        );
        assert_eq!(
            ex.extract(&Read::SingleForward(&rec_ref)),
            Err(DropReason::LbsOverlap)
        );
    }

    /// A reverse-strand read whose UMI is LBS-contaminated in the canonical
    /// orientation must drop as `LbsOverlap`. The raw reverse UMI bytes are the
    /// reverse complement of lbs-1, so only the canonical (`strand.encode`)
    /// encoding lines up with the stored orientation.
    #[test]
    fn lbs_overlap_filter_drops_reverse_strand_contaminated_umi() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let mut records = generate_single_reads(&assay, &markers, 1, mix, 1, Strand::Reverse);
        let mut record = records.pop().expect("one read generated");
        // umi-1 on a reverse read sits at the reverse bounds; fill it with the
        // reverse complement of lbs-1 so `encode_dna_2bit_rc` recovers lbs-1.
        let (umi_start, umi_end) = assay.reverse_bounds("umi-1").expect("umi-1 reverse bounds");
        let lbs1 = assay
            .region_sequence("lbs-1")
            .expect("lbs-1 sequence")
            .as_bytes();
        record.seq[umi_start..umi_end].copy_from_slice(&revcomp(lbs1));
        let rec_ref = &record;

        let ex = Extractor::new(
            &proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig {
                lbs_overlap_max_edits: Some(6),
                ..Default::default()
            },
        );
        assert_eq!(
            ex.extract(&Read::SingleReverse(&rec_ref)),
            Err(DropReason::LbsOverlap)
        );
    }

    #[test]
    fn lbs_overlap_filter_disabled_keeps_lbs1_umi() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let mut records = generate_single_reads(&assay, &markers, 1, mix, 1, Strand::Forward);
        let mut record = records.pop().expect("one read generated");
        let (umi_start, umi_end) = assay.forward_bounds("umi-1").expect("umi-1 bounds");
        let lbs1 = assay
            .region_sequence("lbs-1")
            .expect("lbs-1 sequence")
            .as_bytes();
        record.seq[umi_start..umi_end].copy_from_slice(lbs1);
        let rec_ref = &record;

        // Explicitly disable the LBS filter (the default now enables it).
        let ex = Extractor::new(
            &proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig {
                lbs_overlap_max_edits: None,
                ..Default::default()
            },
        );
        assert!(
            ex.extract(&Read::SingleForward(&rec_ref)).is_ok(),
            "exact lbs-1 UMI should be kept when LBS filter is disabled"
        );
    }
}
