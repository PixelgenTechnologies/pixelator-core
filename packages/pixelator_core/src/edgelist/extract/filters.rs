/// Returns whether `seq` contains any `N` base.
pub(super) fn has_n(seq: &[u8]) -> bool {
    seq.contains(&b'N')
}

/// Returns whether the mean Phred+33 quality across the informative `regions`
/// (pid + umi) is strictly below `min`.
///
/// Only the informative regions are considered; linker/uei bases are ignored.
/// An empty total (e.g. missing quality) is treated as not-below (kept),
/// matching the "missing quality keeps the read" policy when `qual()` is `None`.
pub(super) fn avg_phred_below(regions: &[&[u8]], min: u8) -> bool {
    let mut sum: u64 = 0;
    let mut len: usize = 0;
    for region in regions {
        sum += region.iter().map(|&q| (q - 33) as u64).sum::<u64>();
        len += region.len();
    }
    if len == 0 {
        return false;
    }
    (sum as f64 / len as f64) < min as f64
}

/// Returns whether any single base accounts for strictly more than `max_freq`
/// of the UMI length.
pub(super) fn umi_base_freq_exceeds(umi: &[u8], max_freq: f64) -> bool {
    if umi.is_empty() {
        return false;
    }
    let mut counts = [0u32; 4];
    for &b in umi {
        let i = match b {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' => 3,
            _ => continue,
        };
        counts[i] += 1;
    }
    let max = counts.into_iter().max().unwrap_or(0);
    (max as f64 / umi.len() as f64) > max_freq
}

/// Recommended maximum edit distance for the LBS–UMI overlap filter. Chosen
/// against the observed contaminant UMIs (all caught at `k ≤ 6`, none at
/// distance 7 or beyond).
pub const DEFAULT_LBS_OVERLAP_MAX_EDITS: u8 = 6;

/// Read-filtering and panel-matching configuration threaded through the
/// edgelist pipeline.
#[derive(Clone, Copy, Debug)]
pub struct FilterConfig {
    /// Maximum number of mismatches tolerated when matching a pid against the
    /// panel. `None` means exact matching (threshold 0).
    pub mismatch_threshold: Option<u8>,
    /// Minimum average Phred quality across the informative regions; reads below
    /// this drop as `LowQuality`. `None` disables the gate.
    pub min_avg_quality: Option<u8>,
    /// Maximum single-base frequency allowed in the UMI; reads above this drop
    /// as `LowComplexity`. `None` disables the gate.
    pub max_umi_base_freq: Option<f64>,
    /// Maximum bounded infix Levenshtein distance for the LBS–UMI overlap
    /// filter. A UMI is dropped when it aligns to any substring of an LBS
    /// reference (or its reverse complement) within this many edits. `None`
    /// disables the gate.
    pub lbs_overlap_max_edits: Option<u8>,
}

impl Default for FilterConfig {
    /// The LBS-overlap filter is on by default (like the neighbouring quality and
    /// complexity gates in production), so the default enables it; the other
    /// gates stay `None` (opt-in) as before.
    fn default() -> Self {
        Self {
            mismatch_threshold: None,
            min_avg_quality: None,
            max_umi_base_freq: None,
            lbs_overlap_max_edits: Some(DEFAULT_LBS_OVERLAP_MAX_EDITS),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_n_detects_uppercase_n_anywhere() {
        assert!(has_n(b"ACNT"));
        assert!(has_n(b"NACGT"));
        assert!(has_n(b"ACGTN"));
    }

    #[test]
    fn has_n_is_false_without_uppercase_n() {
        assert!(!has_n(b"ACGT"));
        // Only the uppercase `N` sentinel counts; lowercase `n` is not treated
        // as a missing base here.
        assert!(!has_n(b"acnt"));
    }

    #[test]
    fn has_n_empty_slice_is_false() {
        assert!(!has_n(b""));
    }

    #[test]
    fn avg_phred_below_true_when_mean_under_threshold() {
        // '+' is Phred+33 for Q10; a mean of 10 is below 20.
        assert!(avg_phred_below(&[b"++++"], 20));
    }

    #[test]
    fn avg_phred_below_false_when_mean_at_or_above_threshold() {
        // 'I' is Q40, well above 30.
        assert!(!avg_phred_below(&[b"IIII"], 30));
        // '5' is exactly Q20; the gate is strict (`<`), so Q20 is kept at min 20.
        assert!(!avg_phred_below(&[b"5555"], 20));
    }

    #[test]
    fn avg_phred_below_pools_all_regions_into_one_mean() {
        // Two Q40 bases ('I') and two Q10 bases ('+') average to Q25.
        assert!(avg_phred_below(&[b"II", b"++"], 30));
        assert!(!avg_phred_below(&[b"II", b"++"], 20));
    }

    #[test]
    fn avg_phred_below_empty_total_is_kept() {
        // No regions, or only empty regions, means no quality signal: kept.
        assert!(!avg_phred_below(&[], 40));
        assert!(!avg_phred_below(&[b"", b""], 40));
    }

    #[test]
    fn umi_base_freq_exceeds_true_for_low_complexity() {
        // A homopolymer is 100% one base, above any threshold below 1.0.
        assert!(umi_base_freq_exceeds(b"AAAA", 0.9));
        // Lowercase bases are counted the same as uppercase.
        assert!(umi_base_freq_exceeds(b"aaaa", 0.9));
    }

    #[test]
    fn umi_base_freq_exceeds_false_for_diverse_umi() {
        // Each base appears once: max frequency is 0.25.
        assert!(!umi_base_freq_exceeds(b"ACGT", 0.9));
    }

    #[test]
    fn umi_base_freq_exceeds_is_strict_at_boundary() {
        // Three of four bases are 'A' -> frequency 0.75.
        assert!(!umi_base_freq_exceeds(b"AAAC", 0.75));
        assert!(umi_base_freq_exceeds(b"AAAC", 0.7));
    }

    #[test]
    fn umi_base_freq_exceeds_counts_non_acgt_toward_length_only() {
        // `N` bases are not tallied per-base but still count toward the length,
        // so two of four positions are 'A' -> frequency 0.5.
        assert!(!umi_base_freq_exceeds(b"AANN", 0.6));
        assert!(umi_base_freq_exceeds(b"AANN", 0.4));
    }

    #[test]
    fn umi_base_freq_exceeds_empty_umi_is_kept() {
        assert!(!umi_base_freq_exceeds(b"", 0.5));
    }
}
