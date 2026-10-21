//! Extension entry point. Bindings live in [`graph`] and [`edgelist`]; this
//! module only installs the allocator and registers the public names.

use pyo3::prelude::*;

mod convert;
mod edgelist;
mod graph;

// MiMalloc backs every Rust allocation in this extension, including the
// statically linked `pixelator-core` code, so the hot edgelist path keeps its
// allocator performance even when driven from Python -- except on macOS, where
// mimalloc reserves a fixed pthread thread-local-storage slot for the per-thread
// heap pointer. PyArrow bundles its own, independently-built mimalloc that
// reserves the same kind of slot, and when both are loaded into the same process
// they clobber each other's per-thread heap pointer. Falling back to the system
// allocator on macOS avoids that.
#[cfg(not(target_os = "macos"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[pymodule(name = "pixelator_core")]
mod pixelator_core_py {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::graph::find_graph_statistics;

    #[pymodule_export]
    use super::graph::run_connected_components;

    #[pymodule_export]
    use super::graph::run_label_propagation;

    #[pymodule_export]
    use super::graph::run_leiden;

    #[pymodule_export]
    use super::graph::run_hybrid_community_detection;

    #[pymodule_export]
    use super::graph::PyGraphProperties;

    #[pymodule_export]
    use super::edgelist::run_edgelist;

    #[pymodule_export]
    use super::edgelist::encode_dna_2bit;

    #[pymodule_export]
    use super::edgelist::decode_dna_2bit;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        pyo3_log::init();
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
}
