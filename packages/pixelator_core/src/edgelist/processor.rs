use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Mutex};

use arrow::record_batch::RecordBatch;
use paraseq::ProcessError;
use paraseq::prelude::*;

use crate::common::{Assay, Panel};

use super::extract::{Extractor, FilterConfig};
use super::qc::QcProfiler;
use super::read_layout::{Read, Strand};
use super::row_buffer::RowBuffer;
use super::stats::EdgelistStats;

// Tracked on its own rather than derived from EdgelistStats, which already counts
// every read. The two have different lifecycles: a worker's stats are folded into
// the global tally and reset at end of thread, so reading progress off them couples
// the two and turns that reset into a hazard.
//
// The shared counter is cumulative and is never reset, unlike the per-thread
// tallies it runs alongside.

/// Publishes read counts into a counter shared with a progress reporter,
/// batching the atomic writes so the per-read path stays thread-local.
#[derive(Clone, Default)]
struct Progress {
    shared: Arc<AtomicU64>,
    /// Reads counted since the last [`Progress::publish`].
    pending: u64,
}

impl Progress {
    fn new(shared: Arc<AtomicU64>) -> Self {
        Self { shared, pending: 0 }
    }

    /// Counts one processed read, thread-locally.
    fn count_read(&mut self) {
        self.pending += 1;
    }

    /// Publishes and clears whatever has accumulated since the last call.
    ///
    /// Called from the batch/thread-complete hooks, never from the per-read hot
    /// path, so a worker touches the shared atomic once per batch rather than
    /// once per read.
    fn publish(&mut self) {
        let pending = std::mem::take(&mut self.pending);
        if pending > 0 {
            self.shared.fetch_add(pending, Ordering::Relaxed);
        }
    }
}

/// The mode-agnostic core shared by [`PairedProcessor`] and [`SingleProcessor`].
///
/// Owns the batch machinery (row buffer, channel, stats, extractor) and all the
/// logic that does not depend on how a [`Read`] is assembled, so the per-mode
/// processors stay thin wrappers that only decide which `Read` variant to build.
#[derive(Clone)]
struct RowSink {
    buffer: RowBuffer,

    tx: SyncSender<RecordBatch>,
    batch_threshold: usize,

    local_stats: EdgelistStats,
    global_stats: Arc<Mutex<EdgelistStats>>,

    progress: Progress,

    extractor: Extractor,
    qc_profiler: QcProfiler,
}

impl RowSink {
    fn new(
        tx: SyncSender<RecordBatch>,
        batch_threshold: usize,
        assay: Assay,
        panel: Panel,
        filters: FilterConfig,
    ) -> Self {
        Self {
            buffer: RowBuffer::new(batch_threshold, &panel),
            tx,
            batch_threshold,
            local_stats: EdgelistStats::default(),
            global_stats: Arc::new(Mutex::new(EdgelistStats::default())),
            progress: Progress::default(),
            extractor: Extractor::new(&assay, panel, filters),
            qc_profiler: QcProfiler::new(&assay),
        }
    }

    /// Extracts one molecule from `read` and records the outcome: a valid row is
    /// buffered and counted as kept; a [`DropReason`](super::extract::DropReason)
    /// is tallied.
    ///
    /// QC is measured first, ahead of the keep/drop decision, so that dropped
    /// reads still contribute to the quality totals.
    fn process<Rf: Record>(&mut self, read: Read<'_, Rf>) {
        self.progress.count_read();
        self.local_stats.add_qc(self.qc_profiler.eval(&read));
        match self.extractor.extract(read) {
            Ok(row) => {
                self.buffer
                    .push(row.marker_idx1, row.marker_idx2, row.umi1, row.umi2);
                self.local_stats.keep_read(row.marker_idx1, row.marker_idx2);
            }
            Err(reason) => self.local_stats.drop_read(reason),
        }
    }

    fn flush_and_send(&mut self) -> Result<(), ProcessError> {
        let batch = self
            .buffer
            .flush()
            .map_err(IntoProcessError::into_process_error)?;
        self.tx
            .send(batch)
            .map_err(IntoProcessError::into_process_error)
    }

    /// Batch boundary (`on_batch_complete`): publishes progress and flushes the
    /// buffer if it has reached the batch threshold.
    fn finish_batch(&mut self) -> Result<(), ProcessError> {
        self.progress.publish();
        if self.buffer.len() >= self.batch_threshold {
            self.flush_and_send()?;
        }
        Ok(())
    }

    /// End of thread (`on_thread_complete`): publishes the last of the progress,
    /// flushes any trailing rows, and folds this worker's stats into the shared
    /// tally.
    fn finish_thread(&mut self) -> Result<(), ProcessError> {
        self.progress.publish();
        if !self.buffer.is_empty() {
            self.flush_and_send()?;
        }

        let mut global_stats = self.global_stats.lock().expect("lock global stats");
        *global_stats += std::mem::take(&mut self.local_stats);

        Ok(())
    }
}

/// Paired-end processor: side one comes from the forward read, side two from the
/// reverse read.
#[derive(Clone)]
pub struct PairedProcessor {
    sink: RowSink,
}

impl PairedProcessor {
    pub fn new(
        tx: SyncSender<RecordBatch>,
        batch_threshold: usize,
        assay: Assay,
        panel: Panel,
        filters: FilterConfig,
    ) -> Self {
        Self {
            sink: RowSink::new(tx, batch_threshold, assay, panel, filters),
        }
    }

    /// Share `counter` across worker clones so a reporter can observe progress.
    pub fn with_progress_counter(mut self, counter: Arc<AtomicU64>) -> Self {
        self.sink.progress = Progress::new(counter);
        self
    }

    /// Consumes the processor and returns the tally merged from every worker.
    ///
    /// The counts only become final once `process_parallel` has returned, since
    /// each worker folds its local tally in as it finishes.
    pub fn into_stats(self) -> EdgelistStats {
        // Taking `self` enforces that ordering: `process_parallel` borrows the
        // processor mutably, so reading the tally and then resuming the run does
        // not compile.
        self.sink
            .global_stats
            .lock()
            .expect("lock global stats")
            .clone()
    }
}

impl<Rf: Record> PairedParallelProcessor<Rf> for PairedProcessor {
    fn process_record_pair(&mut self, r1: Rf, r2: Rf) -> Result<(), ProcessError> {
        self.sink.process(Read::Paired { r1: &r1, r2: &r2 });
        Ok(())
    }

    fn on_batch_complete(&mut self) -> Result<(), ProcessError> {
        self.sink.finish_batch()
    }

    fn on_thread_complete(&mut self) -> Result<(), ProcessError> {
        self.sink.finish_thread()
    }
}

/// Single-end processor: one read spans the whole amplicon, so both sides are
/// read from it at the configured `strand`.
#[derive(Clone)]
pub struct SingleProcessor {
    sink: RowSink,
    strand: Strand,
}

impl SingleProcessor {
    pub fn new(
        tx: SyncSender<RecordBatch>,
        batch_threshold: usize,
        assay: Assay,
        panel: Panel,
        strand: Strand,
        filters: FilterConfig,
    ) -> Self {
        Self {
            sink: RowSink::new(tx, batch_threshold, assay, panel, filters),
            strand,
        }
    }

    /// Share `counter` across worker clones so a reporter can observe progress.
    pub fn with_progress_counter(mut self, counter: Arc<AtomicU64>) -> Self {
        self.sink.progress = Progress::new(counter);
        self
    }

    /// Consumes the processor and returns the tally merged from every worker.
    ///
    /// See [`PairedProcessor::into_stats`].
    pub fn into_stats(self) -> EdgelistStats {
        self.sink
            .global_stats
            .lock()
            .expect("lock global stats")
            .clone()
    }
}

impl<Rf: Record> ParallelProcessor<Rf> for SingleProcessor {
    fn process_record(&mut self, r: Rf) -> Result<(), ProcessError> {
        let read = match self.strand {
            Strand::Forward => Read::SingleForward(&r),
            Strand::Reverse => Read::SingleReverse(&r),
        };
        self.sink.process(read);
        Ok(())
    }

    fn on_batch_complete(&mut self) -> Result<(), ProcessError> {
        self.sink.finish_batch()
    }

    fn on_thread_complete(&mut self) -> Result<(), ProcessError> {
        self.sink.finish_thread()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::mpsc::{Receiver, sync_channel};

    use super::*;
    use crate::edgelist::extract::DropReason;
    use crate::test_fixtures::{proxiome_v2_assay, proxiome_v2_panel};

    /// Builds a `PairedProcessor` from the shared proxiome-v2 fixtures, returning
    /// the receiving end of the channel so it stays alive for the duration of the
    /// test (dropping it would disconnect the sender).
    fn make_processor(batch_threshold: usize) -> (PairedProcessor, Receiver<RecordBatch>) {
        let (tx, rx) = sync_channel(1);
        let proc = PairedProcessor::new(
            tx,
            batch_threshold,
            proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig::default(),
        );
        (proc, rx)
    }

    /// As [`make_processor`], but a single-end `SingleProcessor` for `strand`.
    fn make_single_processor(
        batch_threshold: usize,
        strand: Strand,
    ) -> (SingleProcessor, Receiver<RecordBatch>) {
        let (tx, rx) = sync_channel(1);
        let proc = SingleProcessor::new(
            tx,
            batch_threshold,
            proxiome_v2_assay(),
            proxiome_v2_panel(),
            strand,
            FilterConfig::default(),
        );
        (proc, rx)
    }

    #[test]
    fn new_stores_batch_threshold_and_starts_empty() {
        let threshold = 128;
        let (proc, _rx) = make_processor(threshold);

        assert_eq!(proc.sink.batch_threshold, threshold);
        assert!(proc.sink.buffer.is_empty(), "buffer should start empty");
    }

    #[test]
    fn new_initializes_local_and_global_stats_to_zero() {
        let (proc, _rx) = make_processor(128);

        assert_eq!(proc.sink.local_stats.total_reads(), 0);
        assert_eq!(proc.sink.local_stats.total_kept_reads(), 0);
        assert_eq!(proc.sink.local_stats.total_dropped_reads(), 0);

        let global = proc.sink.global_stats.lock().expect("lock global stats");
        assert_eq!(global.total_reads(), 0);
        assert_eq!(global.total_kept_reads(), 0);
        assert_eq!(global.total_dropped_reads(), 0);
    }

    /// Progress is published only from batch/thread-complete hooks, not per read.
    #[test]
    fn progress_counter_updated_on_batch_and_thread_complete() {
        use crate::test_fixtures::{
            ReadMixConfig, generate_read_pairs, proxiome_v2_marker_sequences,
        };
        use paraseq::parallel::PairedParallelProcessor;
        use std::sync::atomic::Ordering;

        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_read_pairs(&assay, &markers, 50, mix, 3);

        let counter = Arc::new(AtomicU64::new(0));
        let (tx, _rx) = sync_channel(1);
        let mut proc = PairedProcessor::new(
            tx,
            records.len(),
            proxiome_v2_assay(),
            proxiome_v2_panel(),
            FilterConfig::default(),
        )
        .with_progress_counter(counter.clone());

        for (r1, r2) in &records {
            proc.process_record_pair(r1, r2)
                .expect("process record pair");
        }
        assert_eq!(
            counter.load(Ordering::Relaxed),
            0,
            "hot path must not touch the shared counter"
        );

        proc.sink.finish_batch().expect("batch complete");
        assert_eq!(counter.load(Ordering::Relaxed), records.len() as u64);

        // A second flush with no new reads must not double-count.
        proc.sink.finish_batch().expect("batch complete");
        assert_eq!(counter.load(Ordering::Relaxed), records.len() as u64);

        proc.sink.finish_thread().expect("thread complete");
        assert_eq!(counter.load(Ordering::Relaxed), records.len() as u64);
    }

    /// `finish_thread` zeroes the local stats on its way out. Progress is
    /// tracked apart from them, so publishing again from that state must be a
    /// no-op rather than republishing or underflowing.
    #[test]
    fn publishing_progress_after_finish_thread_is_a_no_op() {
        use crate::test_fixtures::{
            ReadMixConfig, generate_read_pairs, proxiome_v2_marker_sequences,
        };
        use paraseq::parallel::PairedParallelProcessor;
        use std::sync::atomic::Ordering;

        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_read_pairs(&assay, &markers, 50, mix, 5);

        let counter = Arc::new(AtomicU64::new(0));
        let (proc, _rx) = make_processor(records.len());
        let mut proc = proc.with_progress_counter(counter.clone());
        for (r1, r2) in &records {
            proc.process_record_pair(r1, r2)
                .expect("process record pair");
        }

        proc.sink.finish_thread().expect("thread complete");
        assert_eq!(counter.load(Ordering::Relaxed), records.len() as u64);

        proc.sink.progress.publish();
        assert_eq!(counter.load(Ordering::Relaxed), records.len() as u64);
    }

    #[test]
    fn progress_publishes_only_what_accumulated_since_the_last_call() {
        use std::sync::atomic::Ordering;

        let counter = Arc::new(AtomicU64::new(0));
        let mut progress = Progress::new(counter.clone());

        for _ in 0..3 {
            progress.count_read();
        }
        assert_eq!(
            counter.load(Ordering::Relaxed),
            0,
            "counting must stay thread-local"
        );

        progress.publish();
        assert_eq!(counter.load(Ordering::Relaxed), 3);

        // Only the two new reads, not the three already published.
        progress.count_read();
        progress.count_read();
        progress.publish();
        assert_eq!(counter.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn publishing_nothing_leaves_the_counter_alone() {
        use std::sync::atomic::Ordering;

        let counter = Arc::new(AtomicU64::new(7));
        let mut progress = Progress::new(counter.clone());

        progress.publish();
        progress.publish();

        assert_eq!(counter.load(Ordering::Relaxed), 7);
    }

    /// Workers are clones sharing one counter, so their contributions add up.
    #[test]
    fn progress_clones_add_into_the_same_counter() {
        use std::sync::atomic::Ordering;

        let counter = Arc::new(AtomicU64::new(0));
        let template = Progress::new(counter.clone());

        let mut worker_a = template.clone();
        let mut worker_b = template.clone();
        worker_a.count_read();
        worker_b.count_read();
        worker_b.count_read();
        worker_a.publish();
        worker_b.publish();

        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }

    /// A generated all-valid pair must survive the whole pipeline. This also
    /// validates the generator's orientation: if `pid-2` were not placed as the
    /// reverse-complement of the marker's `sequence_2`, `lookup_seq2` would miss
    /// and the read would be dropped instead.
    #[test]
    fn generated_valid_reads_are_all_kept() {
        use crate::test_fixtures::{
            ReadMixConfig, generate_read_pairs, proxiome_v2_marker_sequences,
        };
        use paraseq::parallel::PairedParallelProcessor;

        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_read_pairs(&assay, &markers, 1_000, mix, 42);

        let (mut proc, _rx) = make_processor(records.len());
        for (r1, r2) in &records {
            proc.process_record_pair(r1, r2)
                .expect("process record pair");
        }

        assert_eq!(proc.sink.local_stats.total_kept_reads(), records.len());
        assert_eq!(proc.sink.local_stats.total_dropped_reads(), 0);
    }

    /// The default mix (~5% N-drop + ~5% unknown-PID) should drop roughly a tenth
    /// of the reads.
    #[test]
    fn default_mix_drops_roughly_expected_fraction() {
        use crate::test_fixtures::{ReadMixConfig, proxiome_v2_read_pairs};
        use paraseq::parallel::PairedParallelProcessor;

        let records = proxiome_v2_read_pairs(1_000, ReadMixConfig::default(), 7);
        let (mut proc, _rx) = make_processor(records.len());
        for (r1, r2) in &records {
            let _ = proc.process_record_pair(r1, r2);
        }

        assert_eq!(proc.sink.local_stats.total_reads(), records.len());
        let dropped_frac =
            proc.sink.local_stats.total_dropped_reads() as f64 / records.len() as f64;
        assert!(
            (0.06..0.14).contains(&dropped_frac),
            "unexpected drop fraction: {dropped_frac}",
        );
    }

    /// Single-end reads spanning the whole amplicon are all kept, for both
    /// strands. Mirrors `generated_valid_reads_are_all_kept` for the single-end
    /// driver (`SingleProcessor::process_record`).
    #[test]
    fn single_processor_keeps_all_valid_reads() {
        use crate::test_fixtures::{
            ReadMixConfig, generate_single_reads, proxiome_v2_marker_sequences,
        };
        use paraseq::parallel::ParallelProcessor;

        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };

        for strand in [Strand::Forward, Strand::Reverse] {
            let records = generate_single_reads(&assay, &markers, 1_000, mix, 42, strand);

            let (mut proc, _rx) = make_single_processor(records.len(), strand);
            for r in &records {
                proc.process_record(r).expect("process record");
            }

            assert_eq!(
                proc.sink.local_stats.total_kept_reads(),
                records.len(),
                "strand {strand:?}"
            );
            assert_eq!(
                proc.sink.local_stats.total_dropped_reads(),
                0,
                "strand {strand:?}"
            );
        }
    }

    /// The single-end driver tallies drops too: an all-`N` mix drops every read
    /// as `HasN`.
    #[test]
    fn single_processor_tallies_drops() {
        use crate::test_fixtures::{
            ReadMixConfig, generate_single_reads, proxiome_v2_marker_sequences,
        };
        use paraseq::parallel::ParallelProcessor;

        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 1.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_single_reads(&assay, &markers, 500, mix, 11, Strand::Forward);

        let (mut proc, _rx) = make_single_processor(records.len(), Strand::Forward);
        for r in &records {
            proc.process_record(r).expect("process record");
        }

        assert_eq!(proc.sink.local_stats.total_kept_reads(), 0);
        assert_eq!(proc.sink.local_stats.total_dropped_reads(), records.len());
        assert_eq!(
            proc.sink.local_stats.dropped_by.get(&DropReason::HasN),
            Some(&records.len()),
        );
    }

    /// QC is measured before the keep/drop decision, so a dropped read still
    /// contributes to the Q30 counts. The all-`N` mix drops every read.
    #[test]
    fn qc_is_measured_for_dropped_reads_too() {
        use crate::test_fixtures::{
            ReadMixConfig, generate_single_reads, proxiome_v2_marker_sequences,
        };
        use paraseq::parallel::ParallelProcessor;

        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 1.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_single_reads(&assay, &markers, 500, mix, 11, Strand::Forward);

        let (mut proc, _rx) = make_single_processor(records.len(), Strand::Forward);
        for r in &records {
            proc.process_record(r).expect("process record");
        }

        assert_eq!(proc.sink.local_stats.total_kept_reads(), 0);

        let region_len = |region_id: &str| {
            let (start, end) = assay.forward_bounds(region_id).expect("region bounds");
            end - start
        };
        // Totals scale with the read count, so they also pin that every read is
        // measured and that the counts accumulate rather than overwrite.
        let qc = &proc.sink.local_stats.read_qc;
        assert_eq!(qc.q30_whole.total_bases, records.len() * mix.read_len);
        assert_eq!(
            qc.q30_pid_a.total_bases,
            records.len() * region_len("pid-1")
        );
        assert_eq!(
            qc.q30_umi_b.total_bases,
            records.len() * region_len("umi-2")
        );
        // The fixtures emit a constant 'I' (Q40) quality string.
        assert_eq!(qc.q30_whole.q30_bases, qc.q30_whole.total_bases);
    }

    /// At end of thread the worker's local stats — QC included — are folded into
    /// the shared global stats, and the local ones are reset.
    #[test]
    fn finish_thread_folds_qc_into_global_stats() {
        use crate::test_fixtures::{
            ReadMixConfig, generate_read_pairs, proxiome_v2_marker_sequences,
        };
        use paraseq::parallel::PairedParallelProcessor;

        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_read_pairs(&assay, &markers, 50, mix, 5);

        let (mut proc, _rx) = make_processor(records.len());
        for (r1, r2) in &records {
            proc.process_record_pair(r1, r2)
                .expect("process record pair");
        }

        let local_qc = proc.sink.local_stats.read_qc.clone();
        assert_ne!(
            local_qc.q30_whole.total_bases, 0,
            "reads should have been measured before the flush"
        );

        proc.sink.finish_thread().expect("thread complete");

        assert_eq!(proc.sink.local_stats.read_qc.q30_whole.total_bases, 0);
        assert_eq!(proc.sink.local_stats.total_kept_reads(), 0);

        let global = proc.sink.global_stats.lock().expect("lock global stats");
        assert_eq!(
            global.read_qc.q30_whole.total_bases,
            local_qc.q30_whole.total_bases
        );
        assert_eq!(
            global.read_qc.q30_pid_a.total_bases,
            local_qc.q30_pid_a.total_bases
        );
        assert_eq!(global.total_kept_reads(), records.len());
    }

    /// `edgelist` calls this on the one handle it kept, but the tally it needs
    /// covers every worker paraseq cloned from that handle. Both clones fold
    /// their own reads in, and the returned value has to show both.
    #[test]
    fn into_stats_returns_the_tally_folded_by_every_worker() {
        use crate::test_fixtures::{
            ReadMixConfig, generate_read_pairs, proxiome_v2_marker_sequences,
        };
        use paraseq::parallel::PairedParallelProcessor;

        let assay = proxiome_v2_assay();
        let markers = proxiome_v2_marker_sequences();
        let mix = ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let records = generate_read_pairs(&assay, &markers, 20, mix, 7);

        let (mut proc, rx) = make_processor(2 * records.len());
        let mut worker = proc.clone();
        for (r1, r2) in &records {
            proc.process_record_pair(r1, r2)
                .expect("process record pair");
            worker
                .process_record_pair(r1, r2)
                .expect("process record pair");
        }

        proc.sink.finish_thread().expect("thread complete");
        // The channel buffers a single batch, so drain the first worker's
        // before the second one flushes or the send blocks forever.
        rx.recv().expect("first worker's batch");
        worker.sink.finish_thread().expect("worker thread complete");

        let stats = proc.into_stats();

        assert_eq!(stats.total_kept_reads(), 2 * records.len());
        assert_ne!(
            stats.read_qc.q30_whole.total_bases, 0,
            "QC should fold in alongside the read counts"
        );
    }
}
