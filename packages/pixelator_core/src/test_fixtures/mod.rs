//! Realistic shared `Assay`/`Panel` fixtures for unit tests and benches.
//!
//! The fixtures load the real `proxiome-v2` assay and panel resource files that
//! are bundled in this crate under `src/test_fixtures/`. Paths are resolved
//! relative to `CARGO_MANIFEST_DIR` so they work from both unit tests and the
//! benchmark harness.
//!
//! This module is compiled either when running the crate's own tests
//! (`cfg(test)`) or when the `test-fixtures` feature is enabled (how benches
//! reach it via a self dev-dependency).
use std::borrow::Cow;
use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::common::assay::Region;
use crate::common::{Assay, Panel};
use crate::edgelist::Strand;

/// Absolute path to a bundled fixture file, resolved relative to this crate's
/// manifest directory.
fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/test_fixtures")
        .join(name)
}

/// Loads the bundled `proxiome-v2` assay.
pub fn proxiome_v2_assay() -> Assay {
    Assay::from_path(fixture_path("proxiome-v2.yaml")).expect("valid proxiome-v2 assay fixture")
}

/// Loads the bundled `proxiome-v2` immuno-155 panel.
pub fn proxiome_v2_panel() -> Panel {
    Panel::from_path(fixture_path("proxiome-v2-immuno-155-v2.0.csv"))
        .expect("valid proxiome-v2 panel fixture")
}

/// An owned FASTQ-like record for benchmarks and tests.
///
/// [`paraseq::Record`] is implemented for `&OwnedRecord` (references are
/// fundamental, so the impl is permitted) so a pre-generated slice of records
/// can be fed to [`PairedProcessor::process_record_pair`](crate::edgelist::PairedProcessor)
/// by reference, without cloning inside the hot loop being measured.
pub struct OwnedRecord {
    pub id: Vec<u8>,
    pub seq: Vec<u8>,
    pub qual: Vec<u8>,
}

impl paraseq::Record for &OwnedRecord {
    fn id(&self) -> &[u8] {
        self.id.as_slice()
    }

    fn seq(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.seq.as_slice())
    }

    fn seq_raw(&self) -> &[u8] {
        self.seq.as_slice()
    }

    fn qual(&self) -> Option<&[u8]> {
        Some(self.qual.as_slice())
    }
}

/// Controls the distribution of generated read pairs.
///
/// The two drop fractions are mutually exclusive and evaluated against a single
/// uniform roll in `[0, 1)`: a read is an N-drop when `roll < frac_has_n`, an
/// unknown-PID drop when `roll < frac_has_n + frac_unknown_pid`, and otherwise
/// a valid (kept) read.
#[derive(Debug, Clone, Copy)]
pub struct ReadMixConfig {
    /// Fraction of reads with an injected `N`, dropped by the `has_n` guard.
    pub frac_has_n: f64,
    /// Fraction of reads whose PID does not match the panel, dropped at lookup.
    pub frac_unknown_pid: f64,
    /// Length in bases of each generated R1 and R2 read.
    pub read_len: usize,
}

impl Default for ReadMixConfig {
    fn default() -> Self {
        Self {
            frac_has_n: 0.05,
            frac_unknown_pid: 0.05,
            read_len: 150,
        }
    }
}

/// Reverse-complements an ACGT sequence. Non-ACGT bytes are passed through.
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

/// Generates `n` random ACGT bases.
fn random_bases(rng: &mut StdRng, n: usize) -> Vec<u8> {
    const BASES: [u8; 4] = *b"ACGT";
    (0..n).map(|_| BASES[rng.random_range(0..4)]).collect()
}

/// Copies `seq` into `read` starting at `start`, clamped to `read`'s length.
fn place(read: &mut [u8], start: usize, seq: &[u8]) {
    if start >= read.len() {
        return;
    }
    let end = (start + seq.len()).min(read.len());
    read[start..end].copy_from_slice(&seq[..end - start]);
}

/// Collects the fixed (non-`N`) sequences of every region keyed by `region_id`.
///
/// Random regions (UMI/UEI/PID) carry an all-`N` placeholder sequence and are
/// skipped, since placing them would inject `N` and turn every read into an
/// N-drop.
fn collect_fixed_sequences(assay: &Assay) -> Vec<(String, Vec<u8>)> {
    fn walk(region: &Region, out: &mut Vec<(String, Vec<u8>)>) {
        if let Some(seq) = &region.sequence
            && !seq.as_bytes().contains(&b'N')
        {
            out.push((region.region_id.clone(), seq.as_bytes().to_vec()));
        }
        for child in &region.regions {
            walk(child, out);
        }
    }

    let mut out = Vec::new();
    for region in &assay.assay_spec {
        walk(region, &mut out);
    }
    out
}

/// Builds `n` semi-random paired reads laid out according to `assay`.
///
/// Region placement follows the assay bounds: `pid-1` on R1 is a panel marker's
/// `sequence_1` and `pid-2` on R2 is the reverse-complement of that marker's
/// `sequence_2` (matching the lookup contract in
/// [`PairedProcessor::process_record_pair`](crate::edgelist::PairedProcessor)); fixed LBS
/// regions use the assay sequences (reverse-complemented on R2); UMI/UEI regions
/// and any read tail beyond the amplicon are random ACGT. The mix of valid,
/// N-drop, and unknown-PID reads is controlled by `mix` and is deterministic for
/// a given `seed`.
pub fn generate_read_pairs(
    assay: &Assay,
    markers: &[(String, String)],
    n: usize,
    mix: ReadMixConfig,
    seed: u64,
) -> Vec<(OwnedRecord, OwnedRecord)> {
    assert!(
        !markers.is_empty(),
        "need at least one marker to generate reads"
    );

    let mut rng = StdRng::seed_from_u64(seed);
    let fixed = collect_fixed_sequences(assay);

    let cutoff_n = mix.frac_has_n;
    let cutoff_unknown = mix.frac_has_n + mix.frac_unknown_pid;

    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let roll: f64 = rng.random();

        let mut r1 = random_bases(&mut rng, mix.read_len);
        let mut r2 = random_bases(&mut rng, mix.read_len);

        // Fixed LBS regions: forward on R1, reverse-complemented on R2.
        for (id, seq) in &fixed {
            if let Some((start, _)) = assay.forward_bounds(id) {
                place(&mut r1, start, seq);
            }
            if let Some((start, _)) = assay.reverse_bounds(id) {
                place(&mut r2, start, &revcomp(seq));
            }
        }

        // PID regions: valid reads use a panel marker; unknown-PID reads use
        // random bases that (with overwhelming probability) miss the panel.
        let is_unknown = roll >= cutoff_n && roll < cutoff_unknown;
        let (pid1, pid2_rc) = if is_unknown {
            (random_bases(&mut rng, 10), random_bases(&mut rng, 10))
        } else {
            let (seq1, seq2) = &markers[rng.random_range(0..markers.len())];
            (seq1.as_bytes().to_vec(), revcomp(seq2.as_bytes()))
        };
        if let Some((start, _)) = assay.forward_bounds("pid-1") {
            place(&mut r1, start, &pid1);
        }
        if let Some((start, _)) = assay.reverse_bounds("pid-2") {
            place(&mut r2, start, &pid2_rc);
        }

        // N-drop reads: inject an `N` into a UMI region so `has_n` drops the
        // read before any 2-bit encoding is attempted.
        if roll < cutoff_n
            && let Some((start, _)) = assay.forward_bounds("umi-1")
            && start < r1.len()
        {
            r1[start] = b'N';
        }

        let qual = vec![b'I'; mix.read_len];
        out.push((
            OwnedRecord {
                id: Vec::new(),
                seq: r1,
                qual: qual.clone(),
            },
            OwnedRecord {
                id: Vec::new(),
                seq: r2,
                qual,
            },
        ));
    }
    out
}

pub fn generate_single_reads(
    assay: &Assay,
    markers: &[(String, String)],
    n: usize,
    mix: ReadMixConfig,
    seed: u64,
    strand: Strand,
) -> Vec<OwnedRecord> {
    assert!(
        !markers.is_empty(),
        "need at least one marker to generate reads"
    );

    let mut rng = StdRng::seed_from_u64(seed);
    let fixed = collect_fixed_sequences(assay);

    let cutoff_n = mix.frac_has_n;
    let cutoff_unknown = mix.frac_has_n + mix.frac_unknown_pid;

    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let roll: f64 = rng.random();

        let mut r = random_bases(&mut rng, mix.read_len);

        // Fixed LBS regions: forward on R1, reverse-complemented on R2.
        for (id, seq) in &fixed {
            if let Some((start, _)) = strand.bounds(assay, id) {
                match strand {
                    Strand::Forward => place(&mut r, start, seq),
                    Strand::Reverse => place(&mut r, start, &revcomp(seq)),
                }
            }
        }

        // PID regions: valid reads use a panel marker; unknown-PID reads use
        // random bases that (with overwhelming probability) miss the panel.
        let is_unknown = roll >= cutoff_n && roll < cutoff_unknown;
        let (pid1, pid2) = if is_unknown {
            (random_bases(&mut rng, 10), random_bases(&mut rng, 10))
        } else {
            let (seq1, seq2) = &markers[rng.random_range(0..markers.len())];
            // A reverse read is the reverse-complement of the amplicon, so both
            // pids are stored reverse-complemented; the extractor's reverse strand
            // undoes it via `encode_dna_2bit_rc`.
            match strand {
                Strand::Forward => (seq1.as_bytes().to_vec(), seq2.as_bytes().to_vec()),
                Strand::Reverse => (revcomp(seq1.as_bytes()), revcomp(seq2.as_bytes())),
            }
        };

        if let Some((start, _)) = strand.bounds(assay, "pid-1") {
            place(&mut r, start, &pid1);
        }
        if let Some((start, _)) = strand.bounds(assay, "pid-2") {
            place(&mut r, start, &pid2);
        }

        // N-drop reads: inject an `N` into a UMI region so `has_n` drops the
        // read before any 2-bit encoding is attempted.
        if roll < cutoff_n
            && let Some((start, _)) = strand.bounds(assay, "umi-1")
            && start < r.len()
        {
            r[start] = b'N';
        }

        let qual = vec![b'I'; mix.read_len];
        out.push(OwnedRecord {
            id: Vec::new(),
            seq: r,
            qual: qual.clone(),
        });
    }
    out
}

/// Significant bits of a 2-bit-encoded UMI: the `proxiome-v2` UMI is 28 bases
/// (`umi-1`/`umi-2` in the assay), so [`encode_dna_2bit`](crate::encoding::encode_dna_2bit)
/// packs it into `2 * 28 = 56` bits.
const UMI_BITS: u32 = 56;

/// Generates `n_rows` post-extraction edgelist tuples `(marker_idx1, marker_idx2,
/// umi1, umi2)` that mirror what the [`RowBuffer`](crate::edgelist::RowBuffer) and
/// Parquet writer actually see, for compression/throughput benchmarking.
///
/// The distribution is chosen to be realistic for that stage:
/// - **Markers** are drawn uniformly over `0..n_markers` (a real panel is ~155
///   markers), so the dictionary-encoded marker columns stay low-cardinality.
/// - **Duplication is modeled at the molecule level.** `ceil(n_rows / mean_dup)`
///   distinct molecules (full `(m1, m2, umi1, umi2)` tuples) are drawn once, then
///   sampled with replacement to fill `n_rows`. So a given molecule (and thus its
///   whole row, including both UMIs) repeats ~`mean_dup` times, matching PCR/
///   sequencing duplicates of one original molecule.
/// - **Ordering is scattered** (arrival order): duplicates are spread across the
///   output rather than clustered, which is the unsorted, realistic case that
///   governs how compressible the file is.
/// - **UMIs** are random [`UMI_BITS`]-bit values, so the `umi` columns are
///   high-entropy across ~7 bytes.
///
/// Output is deterministic for a given `seed`.
pub fn generate_edgelist_rows(
    n_rows: usize,
    n_markers: i32,
    mean_dup: f64,
    seed: u64,
) -> Vec<(i32, i32, u64, u64)> {
    assert!(n_markers > 0, "need at least one marker");
    assert!(mean_dup >= 1.0, "mean_dup must be >= 1.0");

    let mut rng = StdRng::seed_from_u64(seed);
    let umi_mask = (1u64 << UMI_BITS) - 1;

    let n_molecules = ((n_rows as f64 / mean_dup).ceil() as usize).max(1);
    let molecules: Vec<(i32, i32, u64, u64)> = (0..n_molecules)
        .map(|_| {
            (
                rng.random_range(0..n_markers),
                rng.random_range(0..n_markers),
                rng.random::<u64>() & umi_mask,
                rng.random::<u64>() & umi_mask,
            )
        })
        .collect();

    (0..n_rows)
        .map(|_| molecules[rng.random_range(0..n_molecules)])
        .collect()
}

/// Marker `(sequence_1, sequence_2)` pairs from the bundled `proxiome-v2` panel.
pub fn proxiome_v2_marker_sequences() -> Vec<(String, String)> {
    proxiome_v2_panel()
        .markers()
        .iter()
        .map(|m| (m.sequence_1.clone(), m.sequence_2.clone()))
        .collect()
}

/// Marker ids from the bundled `proxiome-v2` panel (realistic pid strings).
pub fn proxiome_v2_marker_ids() -> Vec<String> {
    proxiome_v2_panel()
        .markers()
        .iter()
        .map(|m| m.marker_id.clone())
        .collect()
}

/// Convenience wrapper that generates `proxiome-v2` read pairs using the bundled
/// assay and panel fixtures.
pub fn proxiome_v2_read_pairs(
    n: usize,
    mix: ReadMixConfig,
    seed: u64,
) -> Vec<(OwnedRecord, OwnedRecord)> {
    let assay = proxiome_v2_assay();
    let markers = proxiome_v2_marker_sequences();
    generate_read_pairs(&assay, &markers, n, mix, seed)
}
