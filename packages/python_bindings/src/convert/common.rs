use std::str::FromStr;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;

/// Reject a `str` or `pathlib.Path` so callers get a clear TypeError instead of
/// a failed attribute extract after the API stopped taking file paths.
pub(super) fn reject_path_like(
    ob: &Bound<'_, PyAny>,
    arg: &str,
    expected: &str,
    hint: &str,
) -> PyResult<()> {
    if ob.is_instance_of::<PyString>() || is_pathlib_path(ob)? {
        return Err(PyTypeError::new_err(format!(
            "{arg} must be a {expected} instance, not a file path. {hint}"
        )));
    }
    Ok(())
}

fn is_pathlib_path(ob: &Bound<'_, PyAny>) -> PyResult<bool> {
    let pathlib = ob.py().import("pathlib")?;
    let path_cls = pathlib.getattr("Path")?;
    ob.is_instance(&path_cls)
}

pub(super) fn parse_from_str<T: FromStr>(ob: &Bound<'_, PyAny>, field: &str) -> PyResult<T>
where
    T::Err: std::fmt::Display,
{
    let raw: String = ob.extract()?;
    T::from_str(&raw).map_err(|e| PyValueError::new_err(format!("invalid {field}: {e}")))
}
