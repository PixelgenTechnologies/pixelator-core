use crate::common::distances::{hamming_distance_2bit, within_hamming_distance};
use crate::encoding::encode_dna_2bit;

use csv::ReaderBuilder;
use rustc_hash::FxHashMap as HashMap;
use serde::Deserialize;
use std::error::Error;
use std::path::Path;

/// An antibody panel indexed for fast marker lookups by barcode sequence.
///
/// Each marker in a panel is associated with two barcode sequences. The panel
/// keeps a separate map for each, mapping the 2-bit-encoded sequence (see
/// [`encode_dna_2bit`]) to the marker identifier so that a barcode observed in
/// the data can be resolved back to its marker in O(1).
///
/// Both sequences are stored exactly as they appear in the panel file. Callers
/// are responsible for encoding an observed barcode into the same orientation
/// before looking it up (e.g. reverse-complementing a `sequence_2` barcode read
/// off the reverse read via [`encode_dna_2bit_rc`](crate::encoding::encode_dna_2bit_rc)).
///
/// Because the maps are keyed by encoded sequence, each sequence must be unique
/// within its column; [`Panel::from_path`] rejects panels where two markers
/// share a `sequence_1` or a `sequence_2`.
#[derive(Clone)]
pub struct Panel {
    /// All markers, in the order they were loaded from the panel file. This is
    /// the source of truth for enumerating markers/sequences; the lookup maps
    /// below are lossy (encoded, deduped) and cannot reconstruct it.
    markers: Vec<PanelMarker>,
    /// Encoded `sequence_1` -> index of the marker in `markers`.
    map_seq_1: HashMap<u64, i32>,
    /// Encoded `sequence_2` -> index of the marker in `markers`.
    map_seq_2: HashMap<u64, i32>,
}

/// A single marker loaded from a panel: its identifier and its two barcode
/// sequences, stored exactly as they appear in the panel file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanelMarker {
    /// Human-readable marker identifier (e.g. `CD11b`).
    pub marker_id: String,
    /// First barcode sequence for the marker.
    pub sequence_1: String,
    /// Second barcode sequence for the marker.
    pub sequence_2: String,
}

/// A single deserialized row of a panel CSV.
///
/// Only the fields consumed by [`Panel`] are declared here; any other columns
/// present in the file (for example `control`, `full_name`, `alt_id`,
/// `uniprot_id`) are matched by header name and ignored.
#[derive(Debug, Deserialize)]
struct Record {
    /// Human-readable marker identifier (e.g. `CD11b`).
    marker_id: String,
    /// First barcode sequence for the marker.
    sequence_1: String,
    /// Second barcode sequence for the marker.
    sequence_2: String,
}

impl Panel {
    /// Loads a panel from a CSV file at `path`.
    ///
    /// The file is expected to have a header row containing at least the
    /// `marker_id`, `sequence_1`, and `sequence_2` columns. Lines beginning with
    /// `#` are treated as comments and skipped, which allows the leading YAML
    /// metadata block present in the standard panel files.
    ///
    /// Each row's `sequence_1` and `sequence_2` are 2-bit encoded via
    /// [`encode_dna_2bit`] exactly as they appear in the file, and stored as
    /// keys pointing to the row's index in `markers`.
    ///
    /// Real panel files are ragged: rows commonly omit trailing empty columns
    /// (for example `sample_hashing`). Short rows are padded to the header width
    /// so header-based deserialization does not fail on the missing columns.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or read, as soon as any
    /// row fails to deserialize, or if a `sequence_1` or `sequence_2` is
    /// duplicated across markers.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let mut reader = ReaderBuilder::new()
            .comment(Some(b'#'))
            .flexible(true)
            .from_path(path)?;

        let headers = reader.headers()?.clone();

        let mut markers = Vec::new();

        for result in reader.records() {
            let mut row = result?;
            while row.len() < headers.len() {
                row.push_field("");
            }
            let record: Record = row.deserialize(Some(&headers))?;
            markers.push(PanelMarker {
                marker_id: record.marker_id,
                sequence_1: record.sequence_1,
                sequence_2: record.sequence_2,
            });
        }

        Self::from_markers(markers)
    }

    /// Builds a panel from markers already in memory, indexing markers as they appear in
    /// the input vector.
    pub fn from_markers(markers: Vec<PanelMarker>) -> Result<Self, Box<dyn Error>> {
        let mut map_seq_1 = HashMap::default();
        let mut map_seq_2 = HashMap::default();
        for (index, m) in markers.iter().enumerate() {
            let index = index as i32;
            if map_seq_1
                .insert(encode_dna_2bit(m.sequence_1.as_bytes()), index)
                .is_some()
            {
                return Err(format!(
                    "Duplicate sequence_1 '{}' found in panel (marker {})",
                    m.sequence_1, m.marker_id
                )
                .into());
            }
            if map_seq_2
                .insert(encode_dna_2bit(m.sequence_2.as_bytes()), index)
                .is_some()
            {
                return Err(format!(
                    "Duplicate sequence_2 '{}' found in panel (marker {})",
                    m.sequence_2, m.marker_id
                )
                .into());
            }
        }

        Ok(Self {
            markers,
            map_seq_1,
            map_seq_2,
        })
    }

    /// Returns the index (position in [`markers`](Self::markers)) of the marker
    /// whose `sequence_1` matches the 2-bit-encoded `seq`, or `None` if no marker
    /// in the panel uses that sequence.
    pub fn marker_index_seq1(&self, seq: u64) -> Option<i32> {
        self.map_seq_1.get(&seq).copied()
    }

    /// Returns the index (position in [`markers`](Self::markers)) of the marker
    /// whose `sequence_2` matches the 2-bit-encoded `seq`, or `None` if no marker
    /// in the panel uses that sequence.
    pub fn marker_index_seq2(&self, seq: u64) -> Option<i32> {
        self.map_seq_2.get(&seq).copied()
    }

    /// Returns the `marker_id` for a 2-bit-encoded `sequence_1` barcode, or
    /// `None` if no marker in the panel uses that sequence.
    pub fn lookup_seq1(&self, seq: u64) -> Option<&str> {
        self.marker_index_seq1(seq)
            .map(|idx| self.markers[idx as usize].marker_id.as_str())
    }

    /// Returns the `marker_id` for a 2-bit-encoded `sequence_2` barcode, or
    /// `None` if no marker in the panel uses that sequence.
    pub fn lookup_seq2(&self, seq: u64) -> Option<&str> {
        self.marker_index_seq2(seq)
            .map(|idx| self.markers[idx as usize].marker_id.as_str())
    }

    /// Returns all markers loaded from the panel, in file order.
    pub fn markers(&self) -> &[PanelMarker] {
        &self.markers
    }
}

/// Largest `mismatch_threshold` at which the fuzzy fallback is guaranteed never
/// to *silently* assign a barcode to the wrong marker.
///
/// # Background
///
/// A barcode read off the sequencer can carry substitution errors, so an
/// observed sequence often does not match any panel barcode exactly. We model
/// the number of errors as the Hamming distance (count of differing bases)
/// between the observed barcode and a panel barcode, and try to map a noisy read
/// back to the marker it most likely came from.
///
/// This is only possible because panels are *designed* with their barcodes
/// spread apart: any two differ by at least `d_min` bases (here `d_min = 3`).
/// That spacing is what makes the barcode self-correcting for up to 1 error.
///
/// # How the spacing bounds the safe threshold
///
/// Say a read is `t` mismatches from its true marker `A`. By the triangle
/// inequality for Hamming distance, its distance to any *other* marker `B` is
///
/// ```text
/// d(read, B) ≥ d(A, B) − d(A, read) ≥ d_min − t
/// ```
///
/// so the nearest a wrong marker can ever be is `d_min − t`. Weighing that
/// against how close the read is to the right marker (`≤ t`) gives three
/// regimes:
///
/// - `d_min > 2t`: the true marker is *strictly* closest — correction is
///   unambiguous.
/// - `d_min = 2t`: a wrong marker can *tie* the true one but never beat it — so
///   detecting and dropping ties is enough to stay correct.
/// - `d_min < 2t`: a wrong marker can be *strictly* closest — the read is
///   silently misassigned, and no tie check can catch it.
///
/// For `d_min = 3` this makes `t = 1` the largest safe threshold: at `t = 2` a
/// read could be at d_min - t = 1 distance away from an other marker, namely a
/// confident wrong answer. That is the boundary [`PanelSearcher::new`] warns past.
///
/// # Why only direct distances matter, not reverse complements
///
/// The bound above concerns distances between barcodes *as stored in the panel*;
/// reverse complements never enter it. A reverse-strand read is
/// reverse-complemented into the panel's orientation (via
/// [`encode_dna_2bit_rc`](crate::encoding::encode_dna_2bit_rc)) *before* lookup,
/// based on the strand it came from, so no reverse-complemented sequence is ever
/// added to the set of match targets. A panel barcode that happens to lie near
/// the reverse complement of another is therefore harmless — that form is never
/// searched. (In `proxiome-v2-immuno-155-v2.0` one barcode is only 1 mismatch
/// from another's reverse complement, yet the minimum *direct* distance is
/// still 4.)
const SAFE_MISMATCH_THRESHOLD: u8 = 1;

/// Outcome of resolving an observed barcode against the panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerMatch {
    /// Resolved to a single marker (index into the panel's markers).
    Hit(i32),
    /// No panel marker within the mismatch threshold.
    Miss,
    /// Two or more markers tied for the nearest distance; rejected as unsafe.
    Ambiguous,
}

/// Resolves an observed barcode to a panel marker, tolerating up to
/// `mismatch_threshold` mismatches.
///
/// A lookup first tries an exact hash-map hit (the common case). On a miss, the
/// fuzzy fallback is picked to fit the safety regimes documented on
/// `SAFE_MISMATCH_THRESHOLD`:
///
/// - **0**: no fuzzy fallback; exact matches only (`Miss` on miss).
/// - **1**: return the *first* marker within range (`search_first_hit`). With
///   `d_min ≥ 3`, at most one marker can be within 1 mismatch, so the first hit
///   is necessarily the only hit and there is nothing to disambiguate.
/// - **≥ 2**: two markers can now be within range at once, so scan them all,
///   keep the single closest, and return [`MarkerMatch::Ambiguous`] if the two
///   nearest tie (`search_best_match`).
///
/// [`PanelSearcher::new`] warns when `mismatch_threshold` exceeds
/// `SAFE_MISMATCH_THRESHOLD`: past that point the tie check is no longer enough,
/// because a wrong marker can be strictly closest and slip through as a
/// confident but incorrect match.
#[derive(Clone)]
pub struct PanelSearcher {
    panel: Panel,
    mismatch_threshold: u8,
    /// Contiguous `(encoded sequence, marker index)` pairs for the fuzzy
    /// fallback scan. Iterating a packed `Vec` (cache-resident and
    /// autovectorizable) is far faster than iterating the `HashMap`, which is
    /// kept only for the O(1) exact-hit fast path.
    scan_seq_1: Vec<(u64, i32)>,
    scan_seq_2: Vec<(u64, i32)>,
}

impl PanelSearcher {
    /// Builds a searcher over `panel` that tolerates up to `mismatch_threshold`
    /// mismatches when resolving a barcode.
    ///
    /// A `mismatch_threshold` above `SAFE_MISMATCH_THRESHOLD` exceeds the
    /// collision-free guarantee the panel is designed around (see the type-level
    /// docs), so a warning is logged: fuzzy matches may assign a read to a
    /// different marker than intended.
    pub fn new(panel: Panel, mismatch_threshold: u8) -> Self {
        if mismatch_threshold > SAFE_MISMATCH_THRESHOLD {
            log::warn!(
                "PanelSearcher mismatch_threshold {mismatch_threshold} exceeds the safe limit of \
                 {SAFE_MISMATCH_THRESHOLD}; within that radius, all barcodes are guaranteed to be \
                 self-correcting, but beyond it a query can be strictly closest to the wrong marker \
                 and be silently misassigned."
            );
        }
        let scan_seq_1 = panel.map_seq_1.iter().map(|(&s, &i)| (s, i)).collect();
        let scan_seq_2 = panel.map_seq_2.iter().map(|(&s, &i)| (s, i)).collect();
        Self {
            panel,
            mismatch_threshold,
            scan_seq_1,
            scan_seq_2,
        }
    }

    /// Returns the underlying panel.
    pub fn panel(&self) -> &Panel {
        &self.panel
    }

    pub fn search_seq1(&self, seq: u64) -> MarkerMatch {
        self.search(seq, &self.panel.map_seq_1, &self.scan_seq_1)
    }

    pub fn search_seq2(&self, seq: u64) -> MarkerMatch {
        self.search(seq, &self.panel.map_seq_2, &self.scan_seq_2)
    }

    /// Exact hash-map lookup, then a threshold-dependent fuzzy fallback:
    /// first-hit at threshold 1, best-match with tie rejection at threshold ≥ 2.
    fn search(&self, seq: u64, panel_map: &HashMap<u64, i32>, scan: &[(u64, i32)]) -> MarkerMatch {
        if let Some(index) = panel_map.get(&seq) {
            return MarkerMatch::Hit(*index);
        }

        // Dispatch by threshold. See `SAFE_MISMATCH_THRESHOLD`: at threshold 1 at
        // most one marker can be in range, so the first hit is the only hit; at
        // ≥ 2 several can be in range, so we must find the unique nearest and
        // reject ties.
        match self.mismatch_threshold {
            0 => MarkerMatch::Miss,
            1 => self.search_first_hit(seq, scan),
            _ => self.search_best_match(seq, scan),
        }
    }

    fn search_first_hit(&self, seq: u64, scan: &[(u64, i32)]) -> MarkerMatch {
        scan.iter()
            .find(|&&(marker_seq, _)| {
                within_hamming_distance(seq, marker_seq, self.mismatch_threshold)
            })
            .map(|&(_, index)| MarkerMatch::Hit(index))
            .unwrap_or(MarkerMatch::Miss)
    }

    fn search_best_match(&self, seq: u64, scan: &[(u64, i32)]) -> MarkerMatch {
        let mut best_index = None;
        let mut best_distance = self.mismatch_threshold + 1;
        let mut tie = false;
        for &(marker_seq, index) in scan {
            let distance = hamming_distance_2bit(seq, marker_seq);
            if distance > self.mismatch_threshold {
                continue;
            }
            if distance < best_distance {
                best_distance = distance;
                best_index = Some(index);
                tie = false;
            } else if distance == best_distance {
                tie = true;
            }
        }
        match best_index {
            Some(index) if !tie => MarkerMatch::Hit(index),
            Some(_) => MarkerMatch::Ambiguous,
            None => MarkerMatch::Miss,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Rows copied from `proxiome-v1-immuno-155-v1.0.csv` as
    /// (marker_id, sequence_1, sequence_2).
    const FIXTURE_ROWS: &[(&str, &str, &str)] = &[
        ("HLA-ABC", "ACTTCCTAGG", "ACTTCCTAGG"),
        ("B2M", "CCAGGTTCCG", "CCAGGTTCCG"),
        ("CD11b", "CAGCTATGGT", "CAGCTATGGT"),
        ("CD11c", "ATATGACGGT", "ATATGACGGT"),
        ("mIgG1", "ATGCTAGTCG", "ATGCTAGTCG"),
    ];

    /// Build a small panel CSV that mirrors the real file: a `#`-commented YAML
    /// header block followed by the full column set (only marker_id/sequence_1/
    /// sequence_2 are consumed by `Record`).
    fn fixture_csv() -> String {
        let mut csv = String::from(
            "# ---\n\
             # name: test-panel\n\
             # version: 1.0.0\n\
             # ---\n\
             marker_id,control,full_name,alt_id,uniprot_id,sequence_1,sequence_2\n",
        );
        for (marker, seq1, seq2) in FIXTURE_ROWS {
            csv.push_str(&format!("{marker},,Some name,,P00000,{seq1},{seq2}\n"));
        }
        csv
    }

    fn write_fixture() -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp file");
        file.write_all(fixture_csv().as_bytes())
            .expect("write fixture csv");
        file.flush().expect("flush fixture csv");
        file
    }

    #[test]
    fn from_path_loads_all_markers_and_sequences() {
        let file = write_fixture();
        let panel = Panel::from_path(file.path()).expect("load panel");

        for (marker, seq1, seq2) in FIXTURE_ROWS {
            assert_eq!(
                panel.lookup_seq1(encode_dna_2bit(seq1.as_bytes())),
                Some(*marker),
                "sequence_1 lookup failed for {marker}",
            );
            // Both sequences are stored exactly as they appear in the file.
            assert_eq!(
                panel.lookup_seq2(encode_dna_2bit(seq2.as_bytes())),
                Some(*marker),
                "sequence_2 lookup failed for {marker}",
            );
        }

        // Exactly one entry per data row: comment/header lines are skipped and
        // produce no bogus entries.
        assert_eq!(panel.map_seq_1.len(), FIXTURE_ROWS.len());
        assert_eq!(panel.map_seq_2.len(), FIXTURE_ROWS.len());
    }

    #[test]
    fn from_path_exposes_markers_in_file_order() {
        let file = write_fixture();
        let panel = Panel::from_path(file.path()).expect("load panel");

        let markers = panel.markers();
        assert_eq!(markers.len(), FIXTURE_ROWS.len());
        for (marker, (id, seq1, seq2)) in markers.iter().zip(FIXTURE_ROWS) {
            assert_eq!(marker.marker_id, *id);
            assert_eq!(marker.sequence_1, *seq1);
            assert_eq!(marker.sequence_2, *seq2);
        }
    }

    #[test]
    fn duplicate_sequence_is_rejected() {
        let mut csv = String::from("marker_id,sequence_1,sequence_2\n");
        // Two distinct markers sharing the same sequence_1.
        csv.push_str("CD11b,CAGCTATGGT,CAGCTATGGT\n");
        csv.push_str("CD11c,CAGCTATGGT,ATATGACGGT\n");

        let mut file = NamedTempFile::new().expect("create temp file");
        file.write_all(csv.as_bytes()).expect("write csv");
        file.flush().expect("flush csv");

        let msg = match Panel::from_path(file.path()) {
            Ok(_) => panic!("duplicate should error"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("Duplicate sequence_1"),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("CD11c"), "error should name the marker: {msg}");
    }

    #[test]
    fn lookup_returns_none_for_unknown_sequence() {
        let file = write_fixture();
        let panel = Panel::from_path(file.path()).expect("load panel");

        // A sequence that does not appear in the fixture.
        let unknown = encode_dna_2bit(b"TTTTTTTTTT");
        assert_eq!(panel.lookup_seq1(unknown), None);
        assert_eq!(panel.lookup_seq2(unknown), None);
    }

    #[test]
    fn lookup_seq1_and_seq2_are_independent() {
        let seq1 = "ACGTACGTAC";
        let seq2 = "TGCATGCATG";

        let panel = Panel::from_markers(vec![PanelMarker {
            marker_id: String::from("SYNTH"),
            sequence_1: seq1.to_string(),
            sequence_2: seq2.to_string(),
        }])
        .expect("build panel from markers");

        assert_eq!(
            panel.lookup_seq1(encode_dna_2bit(seq1.as_bytes())),
            Some("SYNTH")
        );
        assert_eq!(
            panel.lookup_seq2(encode_dna_2bit(seq2.as_bytes())),
            Some("SYNTH")
        );

        // No cross-contamination: seq2 is not in map_seq_1 and seq1 is not in map_seq_2.
        assert_eq!(panel.lookup_seq1(encode_dna_2bit(seq2.as_bytes())), None);
        assert_eq!(panel.lookup_seq2(encode_dna_2bit(seq1.as_bytes())), None);
    }

    #[test]
    fn panel_searcher_exact_match_returns_marker_index() {
        let panel = Panel::from_markers(vec![PanelMarker {
            marker_id: String::from("CD11b"),
            sequence_1: String::from("CAGCTATGGT"),
            sequence_2: String::from("CAGCTATGGT"),
        }])
        .expect("build panel from markers");
        let searcher = PanelSearcher::new(panel, 0);

        let seq = encode_dna_2bit(b"CAGCTATGGT");
        assert_eq!(searcher.search_seq1(seq), MarkerMatch::Hit(0));
        assert_eq!(searcher.search_seq2(seq), MarkerMatch::Hit(0));
    }

    #[test]
    fn panel_searcher_zero_threshold_rejects_near_match() {
        let panel = Panel::from_markers(vec![PanelMarker {
            marker_id: String::from("CD11b"),
            sequence_1: String::from("CAGCTATGGT"),
            sequence_2: String::from("CAGCTATGGT"),
        }])
        .expect("build panel from markers");
        let searcher = PanelSearcher::new(panel, 0);

        // One base differs from the panel sequence.
        let near = encode_dna_2bit(b"AAGCTATGGT");
        assert_eq!(searcher.search_seq1(near), MarkerMatch::Miss);
        assert_eq!(searcher.search_seq2(near), MarkerMatch::Miss);
    }

    #[test]
    fn panel_searcher_fuzzy_match_within_threshold() {
        let panel = Panel::from_markers(vec![PanelMarker {
            marker_id: String::from("CD11b"),
            sequence_1: String::from("CAGCTATGGT"),
            sequence_2: String::from("CAGCTATGGT"),
        }])
        .expect("build panel from markers");
        let searcher = PanelSearcher::new(panel, 1);

        let near = encode_dna_2bit(b"AAGCTATGGT");
        assert_eq!(searcher.search_seq1(near), MarkerMatch::Hit(0));
        assert_eq!(searcher.search_seq2(near), MarkerMatch::Hit(0));
    }

    #[test]
    fn panel_searcher_returns_miss_when_distance_exceeds_threshold() {
        let panel = Panel::from_markers(vec![PanelMarker {
            marker_id: String::from("CD11b"),
            sequence_1: String::from("CAGCTATGGT"),
            sequence_2: String::from("CAGCTATGGT"),
        }])
        .expect("build panel from markers");
        let searcher = PanelSearcher::new(panel, 1);

        // Two bases differ from the panel sequence.
        let far = encode_dna_2bit(b"AAGCTATGGA");
        assert_eq!(searcher.search_seq1(far), MarkerMatch::Miss);
        assert_eq!(searcher.search_seq2(far), MarkerMatch::Miss);
    }

    #[test]
    fn panel_searcher_returns_miss_for_unknown_sequence() {
        let panel = Panel::from_markers(vec![PanelMarker {
            marker_id: String::from("CD11b"),
            sequence_1: String::from("CAGCTATGGT"),
            sequence_2: String::from("CAGCTATGGT"),
        }])
        .expect("build panel from markers");
        let searcher = PanelSearcher::new(panel, 2);

        let unknown = encode_dna_2bit(b"TTTTTTTTTT");
        assert_eq!(searcher.search_seq1(unknown), MarkerMatch::Miss);
        assert_eq!(searcher.search_seq2(unknown), MarkerMatch::Miss);
    }

    #[test]
    fn panel_searcher_seq1_and_seq2_are_independent() {
        let panel = Panel::from_markers(vec![PanelMarker {
            marker_id: String::from("SYNTH"),
            sequence_1: String::from("ACGTACGTAC"),
            sequence_2: String::from("TGCATGCATG"),
        }])
        .expect("build panel from markers");
        let searcher = PanelSearcher::new(panel, 0);

        let seq1 = encode_dna_2bit(b"ACGTACGTAC");
        let seq2 = encode_dna_2bit(b"TGCATGCATG");

        assert_eq!(searcher.search_seq1(seq1), MarkerMatch::Hit(0));
        assert_eq!(searcher.search_seq2(seq2), MarkerMatch::Hit(0));
        assert_eq!(searcher.search_seq1(seq2), MarkerMatch::Miss);
        assert_eq!(searcher.search_seq2(seq1), MarkerMatch::Miss);
    }

    #[test]
    fn best_match_returns_nearest_within_threshold() {
        let panel = Panel::from_markers(vec![
            PanelMarker {
                marker_id: String::from("A"),
                sequence_1: String::from("AAAAAAAAAA"),
                sequence_2: String::from("GGGGGGGGGG"),
            },
            PanelMarker {
                marker_id: String::from("B"),
                sequence_1: String::from("TTTAAAAAAA"),
                sequence_2: String::from("CCCCCCCCCC"),
            },
        ])
        .expect("build panel from markers");
        let searcher = PanelSearcher::new(panel, 2);

        // Distance 1 from A, distance 2 from B.
        let query = encode_dna_2bit(b"TAAAAAAAAA");
        assert_eq!(searcher.search_seq1(query), MarkerMatch::Hit(0));
    }

    #[test]
    fn best_match_returns_nearest_regardless_of_scan_order() {
        let panel = Panel::from_markers(vec![
            PanelMarker {
                marker_id: String::from("B"),
                sequence_1: String::from("TTTAAAAAAA"),
                sequence_2: String::from("CCCCCCCCCC"),
            },
            PanelMarker {
                marker_id: String::from("A"),
                sequence_1: String::from("AAAAAAAAAA"),
                sequence_2: String::from("GGGGGGGGGG"),
            },
        ])
        .expect("build panel from markers");
        let searcher = PanelSearcher::new(panel, 2);

        // B is scanned first but A is closer.
        let query = encode_dna_2bit(b"TAAAAAAAAA");
        assert_eq!(searcher.search_seq1(query), MarkerMatch::Hit(1));
    }

    #[test]
    fn best_match_rejects_tie() {
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

        // Distance 2 from both A and B.
        let query = encode_dna_2bit(b"TTAAAAAAAA");
        assert_eq!(
            hamming_distance_2bit(query, encode_dna_2bit(b"AAAAAAAAAA")),
            2
        );
        assert_eq!(
            hamming_distance_2bit(query, encode_dna_2bit(b"TTTTAAAAAA")),
            2
        );
        assert_eq!(searcher.search_seq1(query), MarkerMatch::Ambiguous);
    }

    #[test]
    fn best_match_returns_miss_beyond_threshold() {
        let panel = Panel::from_markers(vec![PanelMarker {
            marker_id: String::from("CD11b"),
            sequence_1: String::from("CAGCTATGGT"),
            sequence_2: String::from("GGGGGGGGGG"),
        }])
        .expect("build panel from markers");
        let searcher = PanelSearcher::new(panel, 2);

        let unknown = encode_dna_2bit(b"TTTTTTTTTT");
        assert_eq!(searcher.search_seq1(unknown), MarkerMatch::Miss);
    }

    #[test]
    fn best_match_exact_still_hits() {
        let panel = Panel::from_markers(vec![PanelMarker {
            marker_id: String::from("CD11b"),
            sequence_1: String::from("CAGCTATGGT"),
            sequence_2: String::from("GGGGGGGGGG"),
        }])
        .expect("build panel from markers");
        let searcher = PanelSearcher::new(panel, 2);

        let seq = encode_dna_2bit(b"CAGCTATGGT");
        assert_eq!(searcher.search_seq1(seq), MarkerMatch::Hit(0));
    }
}
