//! Convert Python `PNAAssay` / `PNAAntibodyPanel` objects into Rust types.
//!
//! Parsing and validation stay on the Python side. This module copies the
//! in-memory fields across the FFI boundary and lets the core crate finish
//! construction (`Assay::new`, `Panel::from_markers`).

mod assay;
mod common;
mod panel;

pub use assay::assay_from_py;
pub use panel::panel_from_py;
