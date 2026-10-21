//! What one edgelist run counted: reads kept (in total and per marker), why the
//! rest were dropped, and the Q30 profile of everything that came through.
//!
//! The layout here serves accumulation, not reporting: values combine with
//! [`AddAssign`], so every figure is stored in a form that adds, and the
//! report's fractions and names are derived by the accessors rather than
//! stored.

use std::cmp::Reverse;
use std::ops::AddAssign;

use rustc_hash::FxHashMap as HashMap;

use crate::common::Panel;

use super::extract::DropReason;
use super::qc::ReadQc;

/// Read tallies accumulated over an edgelist run.
#[derive(Clone, Default)]
pub struct EdgelistStats {
    kept: usize,
    /// Per-reason breakdown of the dropped reads.
    pub(super) dropped_by: HashMap<DropReason, usize>,
    pub(super) read_qc: ReadQc,
    /// Kept reads per marker on each side, keyed by panel position - the same
    /// index the row buffer stores. Markers never seen have no entry; names are
    /// attached by the ranking accessors, which take the panel.
    marker1_reads: HashMap<i32, usize>,
    marker2_reads: HashMap<i32, usize>,
}

/// Q30 fractions under the names the report uses.
///
/// Each is the share of bases at Phred 30 or better, pooled over every read the
/// run saw - dropped reads included, since quality is measured ahead of the
/// keep/drop decision.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Q30Statistics {
    /// Whole read: R1 and R2 together when paired, the single read otherwise.
    pub total: f64,
    /// Over the pid-1 region.
    pub pid1: f64,
    /// Over the mui-1 region.
    pub umi1: f64,
    /// Over the pid-2 region.
    pub pid2: f64,
    /// Over the umi-2 region.
    pub umi2: f64,
}

impl AddAssign for EdgelistStats {
    fn add_assign(&mut self, other: Self) {
        self.kept += other.kept;
        for (reason, count) in other.dropped_by {
            *self.dropped_by.entry(reason).or_insert(0) += count;
        }
        self.read_qc += other.read_qc;
        add_counts(&mut self.marker1_reads, other.marker1_reads);
        add_counts(&mut self.marker2_reads, other.marker2_reads);
    }
}

fn add_counts(counts: &mut HashMap<i32, usize>, other: HashMap<i32, usize>) {
    for (marker_idx, count) in other {
        *counts.entry(marker_idx).or_insert(0) += count;
    }
}

/// Pairs each of `panel`'s marker ids with its tally, most frequent first.
fn rank_markers<'p>(panel: &'p Panel, counts: &HashMap<i32, usize>) -> Vec<(&'p str, usize)> {
    let markers = panel.markers();
    // Walking the panel below would silently skip an index it does not hold.
    debug_assert!(
        counts.keys().all(|&idx| (idx as usize) < markers.len()),
        "marker tallies do not come from this panel"
    );
    let mut ranked: Vec<_> = markers
        .iter()
        .enumerate()
        .map(|(idx, marker)| {
            let count = counts.get(&(idx as i32)).copied().unwrap_or_default();
            (marker.marker_id.as_str(), count)
        })
        .collect();
    ranked.sort_unstable_by_key(|&(id, count)| (Reverse(count), id));
    ranked
}

impl EdgelistStats {
    pub(super) fn keep_read(&mut self, marker_idx1: i32, marker_idx2: i32) {
        self.kept += 1;
        *self.marker1_reads.entry(marker_idx1).or_insert(0) += 1;
        *self.marker2_reads.entry(marker_idx2).or_insert(0) += 1;
    }

    pub(super) fn drop_read(&mut self, reason: DropReason) {
        *self.dropped_by.entry(reason).or_insert(0) += 1;
    }

    pub(super) fn add_qc(&mut self, read_qc: ReadQc) {
        self.read_qc += read_qc;
    }

    pub fn total_reads(&self) -> usize {
        self.total_kept_reads() + self.total_dropped_reads()
    }

    pub fn total_kept_reads(&self) -> usize {
        self.kept
    }

    pub fn total_dropped_reads(&self) -> usize {
        self.dropped_by.values().sum()
    }

    /// Q30 fractions under their report names.
    pub fn q30(&self) -> Q30Statistics {
        Q30Statistics {
            total: self.read_qc.q30_whole.rate(),
            pid1: self.read_qc.q30_pid_a.rate(),
            umi1: self.read_qc.q30_umi_a.rate(),
            pid2: self.read_qc.q30_pid_b.rate(),
            umi2: self.read_qc.q30_umi_b.rate(),
        }
    }

    /// Drop tallies by reason name, most frequent first.
    ///
    /// Every drop reason appears, including the ones that never fired, so the
    /// key set is the same on every run.
    pub fn dropped_by_frequency(&self) -> Vec<(&'static str, usize)> {
        let mut dropped_by_list: Vec<_> = DropReason::ALL
            .iter()
            .map(|reason| {
                (
                    reason.as_str(),
                    self.dropped_by.get(reason).copied().unwrap_or_default(),
                )
            })
            .collect();
        dropped_by_list.sort_unstable_by_key(|&(reason, count)| (Reverse(count), reason));
        dropped_by_list
    }

    /// Kept reads by `marker_1` id, most frequent first, ties by id.
    ///
    /// `panel` must be the one the run used, since the tallies are keyed by
    /// position in it. Every panel marker appears, including those never seen,
    /// so the key set is the panel on every run. The counts sum to
    /// [`total_kept_reads`] and match the `marker_1` column of the Parquet.
    ///
    /// [`total_kept_reads`]: Self::total_kept_reads
    pub fn marker1_by_frequency<'p>(&self, panel: &'p Panel) -> Vec<(&'p str, usize)> {
        rank_markers(panel, &self.marker1_reads)
    }

    /// As [`marker1_by_frequency`](Self::marker1_by_frequency), for `marker_2`.
    pub fn marker2_by_frequency<'p>(&self, panel: &'p Panel) -> Vec<(&'p str, usize)> {
        rank_markers(panel, &self.marker2_reads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::PanelMarker;
    use crate::edgelist::qc::Q30Count;

    /// A three-marker panel whose ids are out of alphabetical order, so a
    /// ranking that fell back to panel order would show.
    fn three_marker_panel() -> Panel {
        let marker = |id: &str, s1: &str, s2: &str| PanelMarker {
            marker_id: id.to_string(),
            sequence_1: s1.to_string(),
            sequence_2: s2.to_string(),
        };
        Panel::from_markers(vec![
            marker("CD8", "AAAA", "CCCC"),
            marker("CD4", "GGGG", "TTTT"),
            marker("CD3", "ACGT", "TGCA"),
        ])
        .expect("valid panel")
    }

    /// A `ReadQc` whose five regions carry distinct rates, so an accessor wired
    /// to the wrong region reports a visibly wrong number rather than a
    /// plausible one. Denominators are powers of two to keep every expected
    /// rate exactly representable as an `f64`.
    fn fingerprinted_qc() -> ReadQc {
        let count = |q30_bases| Q30Count {
            q30_bases,
            total_bases: 8,
        };
        ReadQc {
            q30_whole: count(1), // 0.125
            q30_umi_a: count(2), // 0.25
            q30_pid_a: count(3), // 0.375
            q30_pid_b: count(4), // 0.5
            q30_umi_b: count(5), // 0.625
        }
    }

    /// A `ReadQc` carrying only a whole-read tally.
    fn whole_read_qc(q30_bases: usize, total_bases: usize) -> ReadQc {
        ReadQc {
            q30_whole: Q30Count {
                q30_bases,
                total_bases,
            },
            ..Default::default()
        }
    }

    #[test]
    fn default_is_all_zero() {
        let stats = EdgelistStats::default();
        assert_eq!(stats.total_reads(), 0);
        assert_eq!(stats.total_kept_reads(), 0);
        assert_eq!(stats.total_dropped_reads(), 0);
    }

    #[test]
    fn keep_read_increments_reads_and_kept_only() {
        let mut stats = EdgelistStats::default();
        for _ in 0..3 {
            stats.keep_read(0, 0);
        }
        assert_eq!(stats.total_reads(), 3);
        assert_eq!(stats.total_kept_reads(), 3);
        assert_eq!(stats.total_dropped_reads(), 0);
    }

    #[test]
    fn drop_read_increments_reads_and_dropped_only() {
        let mut stats = EdgelistStats::default();
        for _ in 0..2 {
            stats.drop_read(DropReason::HasN);
        }
        assert_eq!(stats.total_reads(), 2);
        assert_eq!(stats.total_dropped_reads(), 2);
        assert_eq!(stats.total_kept_reads(), 0);
        assert_eq!(stats.dropped_by.get(&DropReason::HasN), Some(&2));
    }

    #[test]
    fn drop_read_tallies_per_reason() {
        let mut stats = EdgelistStats::default();
        stats.drop_read(DropReason::HasN);
        stats.drop_read(DropReason::Pid1NotInPanel);
        stats.drop_read(DropReason::Pid1NotInPanel);

        assert_eq!(stats.total_dropped_reads(), 3);
        assert_eq!(stats.dropped_by.get(&DropReason::HasN), Some(&1));
        assert_eq!(stats.dropped_by.get(&DropReason::Pid1NotInPanel), Some(&2));
        assert_eq!(stats.dropped_by.get(&DropReason::Pid2NotInPanel), None);
    }

    #[test]
    fn reads_equal_kept_plus_dropped() {
        let mut stats = EdgelistStats::default();
        stats.keep_read(0, 0);
        stats.keep_read(0, 0);
        stats.keep_read(0, 0);
        stats.drop_read(DropReason::HasN);
        stats.drop_read(DropReason::Pid2NotInPanel);

        assert_eq!(stats.total_reads(), 5);
        assert_eq!(stats.total_kept_reads(), 3);
        assert_eq!(stats.total_dropped_reads(), 2);
        assert_eq!(
            stats.total_reads(),
            stats.total_kept_reads() + stats.total_dropped_reads()
        );
    }

    #[test]
    fn add_assign_merges_each_field_independently() {
        let mut lhs_by = HashMap::default();
        lhs_by.insert(DropReason::HasN, 3);
        lhs_by.insert(DropReason::Pid1NotInPanel, 1);
        let mut total = EdgelistStats {
            kept: 6,
            dropped_by: lhs_by,
            ..Default::default()
        };

        let mut rhs_by = HashMap::default();
        rhs_by.insert(DropReason::HasN, 1);
        rhs_by.insert(DropReason::Pid2NotInPanel, 1);
        total += EdgelistStats {
            kept: 1,
            dropped_by: rhs_by,
            ..Default::default()
        };

        assert_eq!(total.total_reads(), 13);
        assert_eq!(total.total_kept_reads(), 7);
        assert_eq!(total.total_dropped_reads(), 6);
        assert_eq!(
            total.total_reads(),
            total.total_kept_reads() + total.total_dropped_reads()
        );
        assert_eq!(total.dropped_by.get(&DropReason::HasN), Some(&4));
        assert_eq!(total.dropped_by.get(&DropReason::Pid1NotInPanel), Some(&1));
        assert_eq!(total.dropped_by.get(&DropReason::Pid2NotInPanel), Some(&1));
    }

    /// The one test that pins the A/B to 1/2 translation: each region carries a
    /// different rate, so swapping any pair changes the assertion.
    #[test]
    fn q30_reports_each_region_under_its_report_name() {
        let mut stats = EdgelistStats::default();
        stats.add_qc(fingerprinted_qc());

        let q30 = stats.q30();
        assert_eq!(q30.total, 0.125, "total comes from q30_whole");
        assert_eq!(q30.umi1, 0.25, "umi1 comes from the A-side UMI");
        assert_eq!(q30.pid1, 0.375, "pid1 comes from the A-side PID");
        assert_eq!(q30.pid2, 0.5, "pid2 comes from the B-side PID");
        assert_eq!(q30.umi2, 0.625, "umi2 comes from the B-side UMI");
    }

    #[test]
    fn q30_pools_base_counts_instead_of_averaging_per_read_rates() {
        let mut stats = EdgelistStats::default();
        // A one-base read entirely at Q30, then a three-base read with none.
        stats.add_qc(whole_read_qc(1, 1));
        stats.add_qc(whole_read_qc(0, 3));

        // Pooled, that is 1 of 4 bases. Averaging the two per-read rates would
        // give 0.5 and let a short read outweigh a long one.
        assert_eq!(stats.q30().total, 0.25);
    }

    #[test]
    fn q30_is_zero_when_no_bases_were_measured() {
        let q30 = EdgelistStats::default().q30();

        assert_eq!(q30.total, 0.0);
        assert_eq!(q30.umi1, 0.0);
        assert_eq!(q30.pid1, 0.0);
        assert_eq!(q30.pid2, 0.0);
        assert_eq!(q30.umi2, 0.0);
    }

    #[test]
    fn dropped_by_frequency_puts_the_biggest_loss_first() {
        let mut stats = EdgelistStats::default();
        for _ in 0..5 {
            stats.drop_read(DropReason::Pid1NotInPanel);
        }
        for _ in 0..3 {
            stats.drop_read(DropReason::HasN);
        }
        stats.drop_read(DropReason::TooShort);

        let ranked = stats.dropped_by_frequency();
        assert_eq!(ranked[0], ("pid1_not_in_panel", 5));
        assert_eq!(ranked[1], ("has_n", 3));
        assert_eq!(ranked[2], ("too_short", 1));
    }

    #[test]
    fn dropped_by_frequency_lists_every_reason_even_at_zero() {
        let ranked = EdgelistStats::default().dropped_by_frequency();

        assert_eq!(ranked.len(), DropReason::ALL.len());
        assert!(ranked.iter().all(|&(_, count)| count == 0));
    }

    /// Ties are the common case rather than an edge case: every reason that did
    /// not fire sits at zero. Without a tiebreak their order would follow
    /// `HashMap` iteration, which is randomized per process and would churn the
    /// report on every run.
    #[test]
    fn dropped_by_frequency_breaks_ties_on_the_reason_name() {
        let mut stats = EdgelistStats::default();
        for _ in 0..9 {
            stats.drop_read(DropReason::LowQuality);
        }
        for _ in 0..4 {
            stats.drop_read(DropReason::Pid2NotInPanel);
        }
        for _ in 0..4 {
            stats.drop_read(DropReason::HasN);
        }

        assert_eq!(
            stats.dropped_by_frequency(),
            [
                ("low_quality", 9),
                // Tied at 4, so name order decides.
                ("has_n", 4),
                ("pid2_not_in_panel", 4),
                // All tied at 0, likewise in name order.
                ("lbs_overlap", 0),
                ("low_complexity", 0),
                ("pid1_ambiguous", 0),
                ("pid1_not_in_panel", 0),
                ("pid2_ambiguous", 0),
                ("too_short", 0),
            ]
        );
    }

    #[test]
    fn dropped_by_frequency_is_stable_across_repeated_calls() {
        let mut stats = EdgelistStats::default();
        stats.drop_read(DropReason::HasN);
        stats.drop_read(DropReason::LowQuality);

        assert_eq!(stats.dropped_by_frequency(), stats.dropped_by_frequency());
    }

    /// Stats need no panel to count, only to be named: a `Default` stats
    /// that tallied reads must rank them in full once handed the panel.
    #[test]
    fn marker_tallies_rank_each_side_independently() {
        let panel = three_marker_panel();
        let mut stats = EdgelistStats::default();
        // marker_1: CD4 x2, CD8 x1. marker_2: CD3 x3.
        stats.keep_read(1, 2);
        stats.keep_read(1, 2);
        stats.keep_read(0, 2);

        assert_eq!(
            stats.marker1_by_frequency(&panel),
            [("CD4", 2), ("CD8", 1), ("CD3", 0)]
        );
        assert_eq!(
            stats.marker2_by_frequency(&panel),
            [("CD3", 3), ("CD4", 0), ("CD8", 0)]
        );
    }

    #[test]
    fn marker_tallies_list_every_panel_marker_even_at_zero() {
        let panel = three_marker_panel();
        let stats = EdgelistStats::default();
        let ranked = stats.marker1_by_frequency(&panel);

        // All tied at zero, so id order decides.
        assert_eq!(ranked, [("CD3", 0), ("CD4", 0), ("CD8", 0)]);
        assert_eq!(stats.marker2_by_frequency(&panel), ranked);
    }

    #[test]
    fn marker_tallies_ignore_dropped_reads() {
        let panel = three_marker_panel();
        let mut stats = EdgelistStats::default();
        stats.keep_read(2, 0);
        stats.drop_read(DropReason::Pid2NotInPanel);

        let side1: usize = stats
            .marker1_by_frequency(&panel)
            .iter()
            .map(|&(_, c)| c)
            .sum();
        let side2: usize = stats
            .marker2_by_frequency(&panel)
            .iter()
            .map(|&(_, c)| c)
            .sum();
        assert_eq!(side1, stats.total_kept_reads());
        assert_eq!(side2, stats.total_kept_reads());
    }

    /// Ranking against a panel smaller than the one the run counted with would
    /// otherwise drop the out-of-range tallies without a trace.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "marker tallies do not come from this panel")]
    fn ranking_against_the_wrong_panel_fails_loudly() {
        let mut stats = EdgelistStats::default();
        stats.keep_read(7, 0);
        stats.marker1_by_frequency(&three_marker_panel());
    }

    /// Workers' stats are folded into a global tally, so per-marker counts must
    /// add per marker - and adding an empty `Default` must change nothing.
    #[test]
    fn add_assign_merges_marker_tallies() {
        let panel = three_marker_panel();
        let mut total = EdgelistStats::default();
        total.keep_read(0, 1);

        let mut worker = EdgelistStats::default();
        worker.keep_read(0, 1);
        worker.keep_read(2, 1);
        total += worker;
        total += EdgelistStats::default();

        assert_eq!(
            total.marker1_by_frequency(&panel),
            [("CD8", 2), ("CD3", 1), ("CD4", 0)]
        );
        assert_eq!(
            total.marker2_by_frequency(&panel),
            [("CD4", 3), ("CD3", 0), ("CD8", 0)]
        );
    }
}
