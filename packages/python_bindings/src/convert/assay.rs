use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pixelator_core::common::assay::{Region, RegionType, SequenceType};
use pixelator_core::common::Assay;

use super::common::{parse_from_str, reject_path_like};

/// Python-shaped assay: only the fields we copy across the FFI boundary.
#[derive(FromPyObject)]
struct PyAssay {
    name: String,
    assay_spec: Vec<PyRegion>,
}

/// Python-shaped region. Extra attributes (`data`, `parent_id`) are ignored.
#[derive(FromPyObject)]
struct PyRegion {
    region_id: String,
    #[pyo3(from_py_with = extract_region_type)]
    region_type: RegionType,
    name: String,
    #[pyo3(from_py_with = extract_sequence_type)]
    sequence_type: SequenceType,
    sequence: Option<String>,
    min_len: Option<usize>,
    max_len: Option<usize>,
    #[pyo3(from_py_with = extract_regions)]
    regions: Vec<PyRegion>,
}

/// Build an [`Assay`] from a Python `PNAAssay` (or any duck-typed equivalent).
pub fn assay_from_py(ob: &Bound<'_, PyAny>) -> PyResult<Assay> {
    reject_path_like(
        ob,
        "assay",
        "PNAAssay",
        "Load it with `pna_config.get_assay(...)` or `PNAAssay.from_yaml(...)`.",
    )?;
    let py: PyAssay = ob.extract()?;
    Assay::new(
        py.name,
        py.assay_spec.into_iter().map(Region::from).collect(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

impl From<PyRegion> for Region {
    fn from(region: PyRegion) -> Self {
        Self {
            region_id: region.region_id,
            region_type: region.region_type,
            name: region.name,
            sequence_type: region.sequence_type,
            sequence: region.sequence.filter(|s| !s.is_empty()),
            min_len: region.min_len,
            max_len: region.max_len,
            regions: region.regions.into_iter().map(Self::from).collect(),
        }
    }
}

fn extract_region_type(ob: &Bound<'_, PyAny>) -> PyResult<RegionType> {
    parse_from_str(ob, "region_type")
}

fn extract_sequence_type(ob: &Bound<'_, PyAny>) -> PyResult<SequenceType> {
    parse_from_str(ob, "sequence_type")
}

fn extract_regions(ob: &Bound<'_, PyAny>) -> PyResult<Vec<PyRegion>> {
    if ob.is_none() {
        Ok(Vec::new())
    } else {
        ob.extract()
    }
}
