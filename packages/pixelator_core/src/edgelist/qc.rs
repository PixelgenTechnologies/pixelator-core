//! Quality metrics measured over raw reads, whatever the pipeline does with them.
//!
//! Everything here measures rather than decides. A metric covers every read the
//! run sees, including the ones [`extract`](super::extract) goes on to discard,
//! and computing one never fails: a read too short to reach a region contributes
//! nothing to that region's tally instead of reporting an error.

// A metric belongs here when it can be read off the raw bases alone. One that
// depends on what extraction decided does not - per-marker read counts need a
// [`Row`](super::extract::Row), so they are tallied by the processor, where that
// outcome is known.

use std::ops::{Add, AddAssign};

use super::read_layout::{ConjugateSide, Read, RegionLayout, Strand};
use crate::common::Assay;
use paraseq::Record;

/// Phred+33 byte above which a base is "Q30" (quality >= 30)
const Q30_MIN_BYTE: u8 = 33 + 30; // 63

/// A Q30 tally over some set of bases: how many reached Q30, out of how many
/// were examined.
///
/// Both halves are kept rather than just their ratio so that tallies from
/// different reads can be summed before [`Q30Count::rate`] is taken. Averaging
/// per-read rates instead would give a short read the same weight as a long
/// one.
#[derive(Default, Clone, Copy)]
pub struct Q30Count {
    pub q30_bases: usize,
    pub total_bases: usize,
}

impl Q30Count {
    fn count(qual: &[u8]) -> Self {
        let q30_bases = qual.iter().filter(|&&b| b >= Q30_MIN_BYTE).count();
        Self {
            q30_bases,
            total_bases: qual.len(),
        }
    }

    pub fn rate(&self) -> f64 {
        if self.total_bases == 0 {
            0.0
        } else {
            self.q30_bases as f64 / self.total_bases as f64
        }
    }
}

impl Add for Q30Count {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            q30_bases: self.q30_bases + other.q30_bases,
            total_bases: self.total_bases + other.total_bases,
        }
    }
}

impl AddAssign for Q30Count {
    fn add_assign(&mut self, other: Self) {
        self.q30_bases += other.q30_bases;
        self.total_bases += other.total_bases;
    }
}

/// The Q30 profile of one read or read pair, as produced by
/// [`QcProfiler::eval`].
///
/// `q30_whole` covers every base that was sequenced, both mates included for a
/// pair, while the four region tallies cover only the `pid`/`umi` bases of each
/// conjugate. They are named after [`ConjugateSide`] rather than the assay's
/// `pid-1`/`pid-2` region ids, since which read a conjugate is read from
/// depends on the sequencing mode.
#[derive(Default, Clone)]
pub struct ReadQc {
    pub q30_whole: Q30Count,
    pub q30_pid_a: Q30Count,
    pub q30_umi_a: Q30Count,
    pub q30_pid_b: Q30Count,
    pub q30_umi_b: Q30Count,
}

impl AddAssign for ReadQc {
    fn add_assign(&mut self, other: Self) {
        self.q30_whole += other.q30_whole;
        self.q30_pid_a += other.q30_pid_a;
        self.q30_umi_a += other.q30_umi_a;
        self.q30_pid_b += other.q30_pid_b;
        self.q30_umi_b += other.q30_umi_b;
    }
}

/// Evaluates Q30 for a specific region of the quality string.
fn region_counts(qual: Option<&[u8]>, (s, e): (usize, usize)) -> Q30Count {
    match qual.and_then(|q| q.get(s..e)) {
        Some(slice) => Q30Count::count(slice),
        None => Q30Count::default(),
    }
}

/// Measures the Q30 profile of a read: the whole read, plus one tally per
/// region of interest.
///
/// Region offsets are resolved from the assay once, at construction, into an
/// owned [`RegionLayout`].
#[derive(Clone)]
pub struct QcProfiler {
    layout: RegionLayout,
}

impl QcProfiler {
    pub fn new(assay: &Assay) -> Self {
        Self {
            layout: RegionLayout::new(assay),
        }
    }

    /// Measures `read`, reading each conjugate's regions at the bounds for the
    /// orientation it was sequenced in.
    ///
    /// This always succeeds. A region the read is too short to reach simply
    /// contributes nothing to its tally, so there is no failure to report and
    /// no read that goes unmeasured.
    pub fn eval<R: Record>(&self, read: &Read<'_, R>) -> ReadQc {
        let q30_whole = |r: &R| r.qual().map(Q30Count::count).unwrap_or_default();

        match read {
            Read::SingleForward(r) => {
                let bounds_a = self.layout.get(ConjugateSide::A, Strand::Forward);
                let bounds_b = self.layout.get(ConjugateSide::B, Strand::Forward);
                ReadQc {
                    q30_whole: q30_whole(r),
                    q30_pid_a: region_counts(r.qual(), bounds_a.pid),
                    q30_umi_a: region_counts(r.qual(), bounds_a.umi),
                    q30_pid_b: region_counts(r.qual(), bounds_b.pid),
                    q30_umi_b: region_counts(r.qual(), bounds_b.umi),
                }
            }
            Read::SingleReverse(r) => {
                let bounds_a = self.layout.get(ConjugateSide::A, Strand::Reverse);
                let bounds_b = self.layout.get(ConjugateSide::B, Strand::Reverse);
                ReadQc {
                    q30_whole: q30_whole(r),
                    q30_pid_a: region_counts(r.qual(), bounds_a.pid),
                    q30_umi_a: region_counts(r.qual(), bounds_a.umi),
                    q30_pid_b: region_counts(r.qual(), bounds_b.pid),
                    q30_umi_b: region_counts(r.qual(), bounds_b.umi),
                }
            }
            Read::Paired { r1, r2 } => {
                let bounds_a = self.layout.get(ConjugateSide::A, Strand::Forward);
                let bounds_b = self.layout.get(ConjugateSide::B, Strand::Reverse);
                ReadQc {
                    q30_whole: q30_whole(r1) + q30_whole(r2),
                    q30_pid_a: region_counts(r1.qual(), bounds_a.pid),
                    q30_umi_a: region_counts(r1.qual(), bounds_a.umi),
                    q30_pid_b: region_counts(r2.qual(), bounds_b.pid),
                    q30_umi_b: region_counts(r2.qual(), bounds_b.umi),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{
        OwnedRecord, ReadMixConfig, generate_read_pairs, generate_single_reads, proxiome_v2_assay,
        proxiome_v2_marker_sequences,
    };

    /// Phred+33 byte well below the threshold (Q10).
    const LOW: u8 = b'+';
    /// Phred+33 byte well above the threshold (Q40).
    const HIGH: u8 = b'I';

    /// Number of Q40 bases to plant in each region of interest.
    type Fingerprint = [(&'static str, usize); 4];

    /// `pid-1`/`pid-2` are both 10 bases and `umi-1`/`umi-2` both 28, so region
    /// lengths alone cannot tell the four apart. Planting a distinct number of
    /// Q40 bases in each makes a misrouted region (wrong conjugate, wrong
    /// strand, wrong field) show up as the wrong count. The two fingerprints
    /// are disjoint so that in paired mode all eight (read, region)
    /// combinations yield different numbers.
    const FP_R1: Fingerprint = [("umi-1", 1), ("pid-1", 2), ("pid-2", 3), ("umi-2", 4)];
    const FP_R2: Fingerprint = [("umi-1", 5), ("pid-1", 6), ("pid-2", 7), ("umi-2", 8)];

    /// An amplicon-length quality string that is Q10 everywhere except for the
    /// leading `n_high` bases of each fingerprinted region.
    fn fingerprint_qual(assay: &Assay, strand: Strand, fingerprint: Fingerprint) -> Vec<u8> {
        let (_, amplicon_len) = assay.forward_bounds("amplicon").expect("amplicon bounds");
        let mut qual = vec![LOW; amplicon_len];
        for (region_id, n_high) in fingerprint {
            let (start, end) = strand.bounds(assay, region_id).expect("region bounds");
            assert!(start + n_high <= end, "fingerprint must fit in {region_id}");
            qual[start..start + n_high].fill(HIGH);
        }
        qual
    }

    /// QC only reads the quality string, so the bases themselves are filler.
    fn record_with_qual(qual: Vec<u8>) -> OwnedRecord {
        OwnedRecord {
            id: Vec::new(),
            seq: vec![b'A'; qual.len()],
            qual,
        }
    }

    fn region_len(assay: &Assay, region_id: &str) -> usize {
        let (start, end) = assay.forward_bounds(region_id).expect("region bounds");
        end - start
    }

    #[test]
    fn count_thresholds_at_phred_30() {
        // '?' is exactly Q30 and counts; '>' is Q29 and does not.
        assert_eq!(Q30Count::count(b"?").q30_bases, 1);
        assert_eq!(Q30Count::count(b">").q30_bases, 0);

        let counts = Q30Count::count(b"II++");
        assert_eq!((counts.q30_bases, counts.total_bases), (2, 4));
    }

    #[test]
    fn count_of_an_empty_slice_is_zero() {
        let counts = Q30Count::count(b"");
        assert_eq!((counts.q30_bases, counts.total_bases), (0, 0));
    }

    #[test]
    fn rate_is_the_q30_fraction_and_is_zero_without_bases() {
        assert_eq!(Q30Count::count(b"II++").rate(), 0.5);
        assert_eq!(Q30Count::count(b"IIII").rate(), 1.0);
        assert_eq!(Q30Count::default().rate(), 0.0);
    }

    #[test]
    fn counts_sum_on_add_and_add_assign() {
        let a = Q30Count::count(b"II++");
        let b = Q30Count::count(b"I+");

        let sum = a + b;
        assert_eq!((sum.q30_bases, sum.total_bases), (3, 6));

        let mut acc = a;
        acc += b;
        assert_eq!((acc.q30_bases, acc.total_bases), (3, 6));
    }

    #[test]
    fn region_counts_skips_a_record_without_quality() {
        let counts = region_counts(None, (0, 4));
        assert_eq!((counts.q30_bases, counts.total_bases), (0, 0));
    }

    #[test]
    fn region_counts_skips_a_region_the_read_does_not_reach() {
        // A truncated region contributes nothing rather than a partial count,
        // so a short read cannot bias the per-region rate.
        let qual = vec![HIGH; 10];
        let counts = region_counts(Some(&qual), (5, 20));
        assert_eq!((counts.q30_bases, counts.total_bases), (0, 0));
    }

    #[test]
    fn region_counts_covers_only_its_own_slice() {
        let mut qual = vec![LOW; 10];
        qual[2..5].fill(HIGH);

        let counts = region_counts(Some(&qual), (2, 5));
        assert_eq!((counts.q30_bases, counts.total_bases), (3, 3));
    }

    #[test]
    fn single_forward_reads_both_conjugates_at_forward_bounds() {
        let assay = proxiome_v2_assay();
        let profiler = QcProfiler::new(&assay);
        let record = record_with_qual(fingerprint_qual(&assay, Strand::Forward, FP_R1));
        let read = &record;

        let qc = profiler.eval(&Read::SingleForward(&read));

        assert_eq!(qc.q30_umi_a.q30_bases, 1);
        assert_eq!(qc.q30_pid_a.q30_bases, 2);
        assert_eq!(qc.q30_pid_b.q30_bases, 3);
        assert_eq!(qc.q30_umi_b.q30_bases, 4);
        assert_eq!(qc.q30_umi_a.total_bases, region_len(&assay, "umi-1"));
        assert_eq!(qc.q30_pid_a.total_bases, region_len(&assay, "pid-1"));
        // The planted bases are the only Q40 ones in the read.
        assert_eq!(qc.q30_whole.q30_bases, 10);
        assert_eq!(qc.q30_whole.total_bases, record.qual.len());
    }

    #[test]
    fn single_reverse_reads_both_conjugates_at_reverse_bounds() {
        let assay = proxiome_v2_assay();
        let profiler = QcProfiler::new(&assay);
        let record = record_with_qual(fingerprint_qual(&assay, Strand::Reverse, FP_R1));
        let read = &record;

        let qc = profiler.eval(&Read::SingleReverse(&read));

        assert_eq!(qc.q30_umi_a.q30_bases, 1);
        assert_eq!(qc.q30_pid_a.q30_bases, 2);
        assert_eq!(qc.q30_pid_b.q30_bases, 3);
        assert_eq!(qc.q30_umi_b.q30_bases, 4);
        assert_eq!(qc.q30_whole.q30_bases, 10);
    }

    #[test]
    fn the_two_orientations_of_one_molecule_give_the_same_counts() {
        let assay = proxiome_v2_assay();
        let profiler = QcProfiler::new(&assay);

        // Sequencing the same molecule from the other end reverses the read,
        // and with it the quality string. Each region then holds the same
        // bases in the opposite order, which cannot change a count.
        let forward_qual = fingerprint_qual(&assay, Strand::Forward, FP_R1);
        let reverse_qual: Vec<u8> = forward_qual.iter().rev().copied().collect();
        let forward = record_with_qual(forward_qual);
        let reverse = record_with_qual(reverse_qual);
        let (forward, reverse) = (&forward, &reverse);

        let fwd = profiler.eval(&Read::SingleForward(&forward));
        let rev = profiler.eval(&Read::SingleReverse(&reverse));

        assert_eq!(rev.q30_umi_a.q30_bases, fwd.q30_umi_a.q30_bases);
        assert_eq!(rev.q30_pid_a.q30_bases, fwd.q30_pid_a.q30_bases);
        assert_eq!(rev.q30_pid_b.q30_bases, fwd.q30_pid_b.q30_bases);
        assert_eq!(rev.q30_umi_b.q30_bases, fwd.q30_umi_b.q30_bases);
        assert_eq!(rev.q30_whole.q30_bases, fwd.q30_whole.q30_bases);
    }

    #[test]
    fn paired_takes_conjugate_a_from_r1_and_conjugate_b_from_r2() {
        let assay = proxiome_v2_assay();
        let profiler = QcProfiler::new(&assay);
        let r1 = record_with_qual(fingerprint_qual(&assay, Strand::Forward, FP_R1));
        let r2 = record_with_qual(fingerprint_qual(&assay, Strand::Reverse, FP_R2));
        let (r1, r2) = (&r1, &r2);

        let qc = profiler.eval(&Read::Paired { r1: &r1, r2: &r2 });

        // A side comes from R1 read forward, B side from R2 read reverse.
        assert_eq!(qc.q30_umi_a.q30_bases, 1);
        assert_eq!(qc.q30_pid_a.q30_bases, 2);
        assert_eq!(qc.q30_pid_b.q30_bases, 7);
        assert_eq!(qc.q30_umi_b.q30_bases, 8);
        // The whole-read counts span both mates.
        assert_eq!(qc.q30_whole.q30_bases, (1 + 2 + 3 + 4) + (5 + 6 + 7 + 8));
        assert_eq!(qc.q30_whole.total_bases, r1.qual.len() + r2.qual.len());
    }

    #[test]
    fn read_qc_sums_each_field_independently() {
        let assay = proxiome_v2_assay();
        let profiler = QcProfiler::new(&assay);
        let record = record_with_qual(fingerprint_qual(&assay, Strand::Forward, FP_R1));
        let read = &record;

        let mut acc = ReadQc::default();
        acc += profiler.eval(&Read::SingleForward(&read));
        acc += profiler.eval(&Read::SingleForward(&read));

        assert_eq!(acc.q30_umi_a.q30_bases, 2);
        assert_eq!(acc.q30_umi_b.q30_bases, 8);
        assert_eq!(acc.q30_whole.q30_bases, 20);
        assert_eq!(acc.q30_whole.total_bases, 2 * record.qual.len());
    }

    #[test]
    fn a_short_read_still_reports_the_regions_it_covers() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        // 100 bases stop short of pid-2 (95..105) and umi-2 (105..133).
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 100,
        };
        let records = generate_single_reads(&assay, &markers, 1, mix, 1, Strand::Forward);
        let read = records.first().expect("one generated read");

        let profiler = QcProfiler::new(&assay);
        let qc = profiler.eval(&Read::SingleForward(&read));

        assert_eq!(qc.q30_umi_a.total_bases, region_len(&assay, "umi-1"));
        assert_eq!(qc.q30_pid_a.total_bases, region_len(&assay, "pid-1"));
        assert_eq!(qc.q30_pid_b.total_bases, 0);
        assert_eq!(qc.q30_umi_b.total_bases, 0);
        assert_eq!(qc.q30_whole.total_bases, 100);
    }

    #[test]
    fn generated_pairs_accumulate_over_every_read() {
        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig::default();
        let records = generate_read_pairs(&assay, &markers, 20, mix, 7);
        let profiler = QcProfiler::new(&assay);

        let mut total = ReadQc::default();
        for (r1, r2) in &records {
            total += profiler.eval(&Read::Paired { r1: &r1, r2: &r2 });
        }

        // The fixtures emit a constant 'I' (Q40) quality string, and dropped
        // reads are measured just like kept ones.
        assert_eq!(total.q30_whole.rate(), 1.0);
        assert_eq!(total.q30_pid_a.rate(), 1.0);
        assert_eq!(total.q30_umi_b.rate(), 1.0);
        assert_eq!(total.q30_whole.total_bases, 20 * 2 * mix.read_len);
        assert_eq!(
            total.q30_pid_a.total_bases,
            20 * region_len(&assay, "pid-1")
        );
    }
}
