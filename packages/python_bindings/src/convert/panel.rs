use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pixelator_core::common::{Panel, PanelMarker};

use super::common::reject_path_like;

/// One panel row, extracted from a dict (`DataFrame.to_dicts()`).
/// Extra columns (`control`, `uniprot_id`, …) are ignored.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct PyMarker {
    marker_id: String,
    sequence_1: String,
    sequence_2: String,
}

/// Build a [`Panel`] from a Python `PNAAntibodyPanel` (or any duck-typed equivalent).
pub fn panel_from_py(ob: &Bound<'_, PyAny>) -> PyResult<Panel> {
    reject_path_like(
        ob,
        "panel",
        "PNAAntibodyPanel",
        "Load it with `load_antibody_panel(...)` or `PNAAntibodyPanel.from_csv(...)`.",
    )?;
    let rows: Vec<PyMarker> = ob
        .call_method0("to_polars")?
        .call_method0("to_dicts")?
        .extract()?;
    Panel::from_markers(rows.into_iter().map(PanelMarker::from).collect())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

impl From<PyMarker> for PanelMarker {
    fn from(marker: PyMarker) -> Self {
        Self {
            marker_id: marker.marker_id,
            sequence_1: marker.sequence_1,
            sequence_2: marker.sequence_2,
        }
    }
}
