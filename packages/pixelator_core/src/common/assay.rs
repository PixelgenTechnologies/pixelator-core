use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use serde::Deserialize;
use std::error::Error;
use std::path::Path;
use std::str::FromStr;

/// The region_ids that every assay is required to define.
const REQUIRED_REGIONS: [&str; 5] = ["amplicon", "pid-1", "pid-2", "umi-1", "umi-2"];

/// Type of a region within an assay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RegionType {
    Umi,
    Pid,
    Lbs,
    Uei,
    Amplicon,
}

impl FromStr for RegionType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "umi" => Ok(Self::Umi),
            "pid" => Ok(Self::Pid),
            "lbs" => Ok(Self::Lbs),
            "uei" => Ok(Self::Uei),
            "amplicon" => Ok(Self::Amplicon),
            other => Err(format!("unknown region_type '{other}'")),
        }
    }
}

/// How the sequence of a region is determined.
///
/// - `Joined`: the sequence is the concatenation of the sub-regions' sequences.
/// - `Fixed`: the sequence is a fixed, given string.
/// - `Random`: the sequence is random (typically written as a run of `N`s).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SequenceType {
    Joined,
    Fixed,
    Random,
}

impl FromStr for SequenceType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "joined" => Ok(Self::Joined),
            "fixed" => Ok(Self::Fixed),
            "random" => Ok(Self::Random),
            other => Err(format!("unknown sequence_type '{other}'")),
        }
    }
}

/// A single region of an assay, possibly containing nested sub-regions.
///
/// The nested `regions` preserve the order given in the YAML, which is
/// significant: regions describe the sequential layout of the amplicon.
#[derive(Debug, Clone, Deserialize)]
pub struct Region {
    /// Unique identifier of the region (e.g. `umi-1`).
    pub region_id: String,
    /// Type of the region.
    pub region_type: RegionType,
    /// Human-readable name of the region.
    pub name: String,
    /// How the region's sequence is determined.
    pub sequence_type: SequenceType,
    /// String representation of the region's sequence, when applicable.
    pub sequence: Option<String>,
    /// Minimum length of the region in bases.
    pub min_len: Option<usize>,
    /// Maximum length of the region in bases.
    pub max_len: Option<usize>,
    /// Ordered list of nested sub-regions.
    #[serde(default)]
    pub regions: Vec<Region>,
}

/// A parsed assay design: a name and an ordered list of top-level regions.
#[derive(Debug, Clone, Deserialize)]
pub struct Assay {
    /// Name of the assay.
    pub name: String,
    /// Ordered list of top-level regions defining the assay structure.
    pub assay_spec: Vec<Region>,
    /// Precomputed forward slice bounds `(start, end)` per region, keyed by
    /// `region_id`. Filled in by [`Assay::populate_region_bounds`] after
    /// parsing; not part of the YAML. Reverse bounds are derived from these
    /// and `total_len` on demand (see [`Assay::reverse_bounds`]).
    #[serde(skip)]
    region_bounds: HashMap<String, (usize, usize)>,
    /// Total length of the assay read (sum of all leaf region lengths). Used to
    /// derive reverse bounds from forward bounds. Filled in by
    /// [`Assay::populate_region_bounds`].
    #[serde(skip)]
    total_len: usize,
}

impl Assay {
    /// Builds an assay from a name and region tree already in memory.
    ///
    /// Validates the region tree and precomputes slice bounds.
    ///
    /// # Errors
    ///
    /// Returns an error if the assay is structurally invalid: a duplicate
    /// `region_id`, a missing required region, or a leaf region without a
    /// `min_len`.
    pub fn new(name: String, assay_spec: Vec<Region>) -> Result<Self, Box<dyn Error>> {
        let mut assay = Self {
            name,
            assay_spec,
            region_bounds: HashMap::default(),
            total_len: 0,
        };
        assay.finalize()?;
        Ok(assay)
    }

    /// Loads and validates an assay from a YAML file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, the YAML cannot be
    /// deserialized, or the assay is structurally invalid: a duplicate
    /// `region_id`, a missing required region, or a leaf region without a
    /// `min_len`.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let text = std::fs::read_to_string(path)?;
        let mut assay: Self = serde_saphyr::from_str(&text)?;
        assay.finalize()?;
        Ok(assay)
    }

    /// Runs structural checks and fills in slice bounds.
    ///
    /// Serde leaves `region_bounds` / `total_len` empty; this is the shared
    /// step that turns that incomplete value (or a freshly built one) into a
    /// usable assay.
    fn finalize(&mut self) -> Result<(), Box<dyn Error>> {
        self.validate()?;
        self.populate_region_bounds()
    }

    /// Precomputes the forward and reverse slice bounds of every region and
    /// stores them keyed by `region_id`.
    ///
    /// Bounds are relative to the start of the whole assay read (all leaf
    /// regions concatenated). Region lengths use `min_len`, which assumes
    /// fixed-length regions (matching the Python `get_position_in_parent`).
    ///
    /// # Errors
    ///
    /// Returns an error if a leaf region has no `min_len`.
    fn populate_region_bounds(&mut self) -> Result<(), Box<dyn Error>> {
        let mut forward: HashMap<String, (usize, usize)> = HashMap::default();
        let mut offset = 0usize;
        for region in &self.assay_spec {
            accumulate_bounds(region, &mut offset, &mut forward)?;
        }

        self.region_bounds = forward;
        self.total_len = offset;
        Ok(())
    }

    /// Forward slice bounds `(s, e)` so that `read[s..e]` extracts the region
    /// from a forward-oriented assay read. Returns `None` if `region_id` is
    /// unknown.
    pub fn forward_bounds(&self, region_id: &str) -> Option<(usize, usize)> {
        self.region_bounds.get(region_id).copied()
    }

    /// Reverse slice bounds `(s, e)` so that `read[s..e]` extracts the region
    /// directly from the raw (non-reverse-complemented) reverse read. The read
    /// is sequenced from the amplicon 3' end, so these bounds index from the
    /// start of that read and are independent of read length. Returns `None` if
    /// `region_id` is unknown.
    pub fn reverse_bounds(&self, region_id: &str) -> Option<(usize, usize)> {
        self.region_bounds
            .get(region_id)
            .map(|&(start, end)| (self.total_len - end, self.total_len - start))
    }

    /// Returns the YAML `sequence` string for `region_id`, searching the nested
    /// region tree. Returns `None` if the id is unknown or the region has no
    /// sequence.
    pub fn region_sequence(&self, region_id: &str) -> Option<&str> {
        fn find<'a>(region: &'a Region, id: &str) -> Option<&'a str> {
            if region.region_id == id {
                return region.sequence.as_deref();
            }
            region.regions.iter().find_map(|child| find(child, id))
        }
        self.assay_spec
            .iter()
            .find_map(|region| find(region, region_id))
    }

    /// Returns the YAML `sequence` strings for every region of type
    /// `region_type`, searching the nested region tree in document order.
    ///
    /// Panics if a matching region has no `sequence`.
    pub fn region_sequences_by_type(&self, region_type: RegionType) -> Vec<&str> {
        fn collect<'a>(region: &'a Region, region_type: RegionType, out: &mut Vec<&'a str>) {
            if region.region_type == region_type {
                out.push(region.sequence.as_deref().unwrap_or_else(|| {
                    panic!("No sequence found for region type {region_type:?}")
                }));
            }
            for child in &region.regions {
                collect(child, region_type, out);
            }
        }

        let mut out = Vec::new();
        for region in &self.assay_spec {
            collect(region, region_type, &mut out);
        }
        out
    }

    /// Validates structural invariants of the assay:
    ///
    /// - every `region_id` is unique across the whole (nested) region tree, and
    /// - all [`REQUIRED_REGIONS`] are present somewhere in the tree.
    ///
    /// # Errors
    ///
    /// Returns an error describing the first duplicate `region_id` encountered,
    /// or the set of required regions that are missing.
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        let mut seen: HashSet<&str> = HashSet::default();
        for region in &self.assay_spec {
            collect_region_ids(region, &mut seen)?;
        }

        let missing: Vec<&str> = REQUIRED_REGIONS
            .iter()
            .copied()
            .filter(|id| !seen.contains(id))
            .collect();

        if !missing.is_empty() {
            return Err(format!("Missing required regions: {}", missing.join(", ")).into());
        }

        Ok(())
    }
}

/// Recursively records the forward `(start, end)` offsets of `region` and all
/// its sub-regions into `out`, advancing `offset` by each leaf's `min_len`.
///
/// A container region spans from the start of its first leaf to the end of its
/// last leaf (derived from the offset delta), so its length is the sum of its
/// leaves' lengths.
fn accumulate_bounds(
    region: &Region,
    offset: &mut usize,
    out: &mut HashMap<String, (usize, usize)>,
) -> Result<(), Box<dyn Error>> {
    let start = *offset;
    if region.regions.is_empty() {
        let len = region
            .min_len
            .ok_or_else(|| format!("Region {} has no min_len", region.region_id))?;
        *offset += len;
    } else {
        for child in &region.regions {
            accumulate_bounds(child, offset, out)?;
        }
    }
    out.insert(region.region_id.clone(), (start, *offset));
    Ok(())
}

/// Recursively inserts a region's id (and those of its sub-regions) into `seen`,
/// erroring on the first duplicate.
fn collect_region_ids<'a>(
    region: &'a Region,
    seen: &mut HashSet<&'a str>,
) -> Result<(), Box<dyn Error>> {
    if !seen.insert(&region.region_id) {
        return Err(format!("Duplicate region_id detected: {}", region.region_id).into());
    }

    for child in &region.regions {
        collect_region_ids(child, seen)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::str::FromStr;
    use tempfile::NamedTempFile;

    /// A complete, valid assay containing the `amplicon` parent and all seven
    /// required child regions.
    const VALID_ASSAY: &str = "\
name: test-assay
assay_spec:
  - region_id: amplicon
    region_type: amplicon
    name: amplicon
    sequence_type: joined
    regions:
      - region_id: umi-1
        region_type: umi
        name: Unique molecular identifier 1
        sequence_type: random
        sequence: NNNNNNNNNNNNNNNNNNNNNNNNNNNN
        min_len: 28
        max_len: 28
      - region_id: pid-1
        region_type: pid
        name: Protein identifier 1
        sequence_type: random
        sequence: NNNNNNNNNN
        min_len: 10
        max_len: 10
      - region_id: lbs-1
        region_type: lbs
        name: Linker binding site 1
        sequence_type: fixed
        sequence: CAAGTGACGCTGGTCAAACACTCATGTC
        min_len: 28
        max_len: 28
      - region_id: uei
        region_type: uei
        name: Unique event identifier 1
        sequence_type: random
        sequence: NNNNNNNNNNNNNNN
        min_len: 15
        max_len: 15
      - region_id: lbs-2
        region_type: lbs
        name: Linker binding site 2
        sequence_type: fixed
        sequence: CGCTTAGATGTCGG
        min_len: 14
        max_len: 14
      - region_id: pid-2
        region_type: pid
        name: Protein identifier 2
        sequence_type: random
        sequence: NNNNNNNNNN
        min_len: 10
        max_len: 10
      - region_id: umi-2
        region_type: umi
        name: Unique molecular identifier 2
        sequence_type: random
        sequence: NNNNNNNNNNNNNNNNNNNNNNNNNNNN
        min_len: 28
        max_len: 28
";

    fn write_fixture(contents: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp file");
        file.write_all(contents.as_bytes()).expect("write fixture");
        file.flush().expect("flush fixture");
        file
    }

    #[test]
    fn from_path_parses_full_assay() {
        let file = write_fixture(VALID_ASSAY);
        let assay = Assay::from_path(file.path()).expect("load assay");

        assert_eq!(assay.name, "test-assay");
        assert_eq!(assay.assay_spec.len(), 1);

        let amplicon = &assay.assay_spec[0];
        assert_eq!(amplicon.region_id, "amplicon");
        assert_eq!(amplicon.region_type, RegionType::Amplicon);
        assert_eq!(amplicon.sequence_type, SequenceType::Joined);
        assert_eq!(amplicon.regions.len(), 7);

        // Spot-check a nested region.
        let uei = &amplicon.regions[3];
        assert_eq!(uei.region_id, "uei");
        assert_eq!(uei.region_type, RegionType::Uei);
        assert_eq!(uei.sequence_type, SequenceType::Random);
        assert_eq!(uei.sequence.as_deref(), Some("NNNNNNNNNNNNNNN"));
        assert_eq!(uei.min_len, Some(15));
        assert_eq!(uei.max_len, Some(15));

        // A fixed region carries its literal sequence.
        let lbs_2 = &amplicon.regions[4];
        assert_eq!(lbs_2.region_type, RegionType::Lbs);
        assert_eq!(lbs_2.sequence_type, SequenceType::Fixed);
        assert_eq!(lbs_2.sequence.as_deref(), Some("CGCTTAGATGTCGG"));
    }

    #[test]
    fn enum_values_deserialize() {
        #[derive(Deserialize)]
        struct Wrapper {
            region_type: RegionType,
            sequence_type: SequenceType,
        }

        let cases = [
            ("umi", RegionType::Umi),
            ("pid", RegionType::Pid),
            ("lbs", RegionType::Lbs),
            ("uei", RegionType::Uei),
            ("amplicon", RegionType::Amplicon),
        ];
        for (raw, expected) in cases {
            let yaml = format!("region_type: {raw}\nsequence_type: joined\n");
            let parsed: Wrapper = serde_saphyr::from_str(&yaml).expect("parse enum");
            assert_eq!(parsed.region_type, expected);
        }

        let seq_cases = [
            ("joined", SequenceType::Joined),
            ("fixed", SequenceType::Fixed),
            ("random", SequenceType::Random),
        ];
        for (raw, expected) in seq_cases {
            let yaml = format!("region_type: umi\nsequence_type: {raw}\n");
            let parsed: Wrapper = serde_saphyr::from_str(&yaml).expect("parse enum");
            assert_eq!(parsed.sequence_type, expected);
        }
    }

    #[test]
    fn enum_from_str_matches_yaml_names() {
        assert_eq!(RegionType::from_str("umi").unwrap(), RegionType::Umi);
        assert_eq!(RegionType::from_str("pid").unwrap(), RegionType::Pid);
        assert_eq!(RegionType::from_str("lbs").unwrap(), RegionType::Lbs);
        assert_eq!(RegionType::from_str("uei").unwrap(), RegionType::Uei);
        assert_eq!(
            RegionType::from_str("amplicon").unwrap(),
            RegionType::Amplicon
        );
        assert!(RegionType::from_str("UMI").is_err());

        assert_eq!(
            SequenceType::from_str("joined").unwrap(),
            SequenceType::Joined
        );
        assert_eq!(
            SequenceType::from_str("fixed").unwrap(),
            SequenceType::Fixed
        );
        assert_eq!(
            SequenceType::from_str("random").unwrap(),
            SequenceType::Random
        );
        assert!(SequenceType::from_str("JOINED").is_err());
    }

    #[test]
    fn duplicate_region_id_is_rejected() {
        // `pid-1` appears twice (top-level and nested), which must be rejected.
        let yaml = VALID_ASSAY.replace("region_id: umi-1", "region_id: pid-1");
        let file = write_fixture(&yaml);

        let err = Assay::from_path(file.path()).expect_err("should reject duplicate");
        assert!(
            err.to_string().contains("Duplicate region_id"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn missing_required_region_is_rejected() {
        // Rename `pid-1` so the required region is absent.
        let yaml = VALID_ASSAY.replace("region_id: pid-1", "region_id: extra");
        let file = write_fixture(&yaml);

        let err = Assay::from_path(file.path()).expect_err("should reject missing region");
        assert!(
            err.to_string().contains("Missing required regions"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn valid_assay_passes_validation() {
        let file = write_fixture(VALID_ASSAY);
        assert!(Assay::from_path(file.path()).is_ok());
    }

    #[test]
    fn new_matches_from_path_bounds() {
        let file = write_fixture(VALID_ASSAY);
        let from_file = Assay::from_path(file.path()).expect("load assay");
        let parsed: Assay = serde_saphyr::from_str(VALID_ASSAY).expect("deserialize");
        let from_new = Assay::new(parsed.name, parsed.assay_spec).expect("new");

        assert_eq!(from_file.name, from_new.name);
        assert_eq!(
            from_file.forward_bounds("umi-1"),
            from_new.forward_bounds("umi-1")
        );
        assert_eq!(
            from_file.forward_bounds("uei"),
            from_new.forward_bounds("uei")
        );
        assert_eq!(
            from_file.forward_bounds("amplicon"),
            from_new.forward_bounds("amplicon")
        );
        assert_eq!(
            from_file.reverse_bounds("umi-1"),
            from_new.reverse_bounds("umi-1")
        );
        assert_eq!(
            from_file.reverse_bounds("umi-2"),
            from_new.reverse_bounds("umi-2")
        );
    }

    #[test]
    fn forward_bounds_are_precomputed() {
        let file = write_fixture(VALID_ASSAY);
        let assay = Assay::from_path(file.path()).expect("load assay");

        // Leaf lengths: umi-1=28, pid-1=10, lbs-1=28, uei=15, lbs-2=14,
        // pid-2=10, umi-2=28; total=133.
        assert_eq!(assay.forward_bounds("umi-1"), Some((0, 28)));
        assert_eq!(assay.forward_bounds("uei"), Some((66, 81)));
        // The container spans the whole read.
        assert_eq!(assay.forward_bounds("amplicon"), Some((0, 133)));
    }

    #[test]
    fn reverse_bounds_index_into_the_raw_read() {
        let file = write_fixture(VALID_ASSAY);
        let assay = Assay::from_path(file.path()).expect("load assay");

        // The raw reverse read starts at the amplicon 3' end, so umi-2 (the
        // 3'-most region) comes first, then pid-2, etc. (total=133).
        assert_eq!(assay.reverse_bounds("umi-2"), Some((0, 28)));
        assert_eq!(assay.reverse_bounds("pid-2"), Some((28, 38)));
        // umi-1 (the 5'-most region) sits at the far end of the reverse read.
        assert_eq!(assay.reverse_bounds("umi-1"), Some((105, 133)));
    }

    #[test]
    fn bounds_return_none_for_unknown_region() {
        let file = write_fixture(VALID_ASSAY);
        let assay = Assay::from_path(file.path()).expect("load assay");

        assert_eq!(assay.forward_bounds("nope"), None);
        assert_eq!(assay.reverse_bounds("nope"), None);
    }

    #[test]
    fn region_sequences_by_type_collects_nested_matches_in_order() {
        let file = write_fixture(VALID_ASSAY);
        let assay = Assay::from_path(file.path()).expect("load assay");

        assert_eq!(
            assay.region_sequences_by_type(RegionType::Lbs),
            vec!["CAAGTGACGCTGGTCAAACACTCATGTC", "CGCTTAGATGTCGG",],
        );
        assert_eq!(
            assay.region_sequences_by_type(RegionType::Umi),
            vec![
                "NNNNNNNNNNNNNNNNNNNNNNNNNNNN",
                "NNNNNNNNNNNNNNNNNNNNNNNNNNNN",
            ],
        );
        assert_eq!(
            assay.region_sequences_by_type(RegionType::Uei),
            vec!["NNNNNNNNNNNNNNN"],
        );
        assert_eq!(
            assay.region_sequences_by_type(RegionType::Pid),
            vec!["NNNNNNNNNN", "NNNNNNNNNN"],
        );
    }

    #[test]
    fn region_sequences_by_type_returns_empty_when_no_match() {
        // LBS is not among the required region ids, so an assay without any
        // lbs regions is still valid.
        let yaml = VALID_ASSAY.replace("region_type: lbs", "region_type: uei");
        let file = write_fixture(&yaml);
        let assay = Assay::from_path(file.path()).expect("load assay");

        assert!(assay.region_sequences_by_type(RegionType::Lbs).is_empty());
    }
}
