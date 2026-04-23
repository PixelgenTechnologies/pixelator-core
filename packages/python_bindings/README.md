# pixelator_core_py

[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/PixelgenTechnologies/pixelator-core/actions/workflows/wheels.yml)

Python bindings for `pixelator-core`, exposing Rust implementations of graph
statistics and community detection algorithms for fast execution from Python.

## Features

- Compute graph-level statistics.
- Run community detection with:
  - Fast Label Propagation (FLP)
  - Leiden
  - Hybrid FLP + Leiden flow (`run_hybrid_community_detection`)

> NB: for the sake of memory efficiency, all these bindings read and write the data from and to parquet files.

## Requirements

- Python 3.10+
- Rust toolchain (for building from source)
- `maturin` (installed automatically when using `uv`)

CI currently builds wheels for Linux, macOS, and Windows across Python 3.10-3.13.

## Installation

### Option 1: Local development install (recommended in this repo)

```bash
cd packages/python_bindings
uv sync --extra dev --no-install-project
uv run maturin develop
uv sync --refresh --extra dev
```

This builds and installs `pixelator_core` into your project `uv` environment.

### Option 2: Build a wheel and install it

```bash
cd packages/python_bindings
maturin build --release --out dist/
pip install dist/*.whl
```

## Quickstart

```python
from pixelator_core import (
    find_graph_statistics,
    run_label_propagation,
    run_leiden,
    run_hybrid_community_detection,
)

parquet_file = "./edgelist.parquet"

# 1) Inspect the graph
n_nodes, n_edges, n_components, frac_lcc = find_graph_statistics(parquet_file)
print(n_nodes, n_edges, n_components, frac_lcc)

# 2) Run hybrid graph workflow and write filtered edge list
(
    output_file,
    pre_recovery_stats,
    post_flp_stats,
    post_recovery_stats,
) = run_hybrid_community_detection(
    parquet_file=parquet_file,
    resolution=1.0,
    output="./filtered_edge_list.parquet",
    flp_epochs=1,
    randomness=0.1,
    seed=42,
    max_iteration=None,
    multiplet_recovery=True,
)
print("Filtered edge list written to:", output_file)
print("Pre recovery nodes:", pre_recovery_stats.node_count)
```

## Input and Output

- Input is expected to be an edge-list Parquet file compatible with `pixelator-core`.
- `run_label_propagation` and `run_leiden` produce node-partition Parquet outputs.
- `run_hybrid_community_detection` produces a filtered edge-list Parquet output.
- Output paths are optional; defaults are used when omitted.

## Logging

The module initializes Rust logging via `pyo3-log` when imported. Configure Python logging in your application to surface logs.

## Troubleshooting

- Build fails with compiler/toolchain errors:
  - Ensure Rust is installed and up to date (`rustup update`).
- Bindings not updated after build
  - Reinstall in the active environment with `uv run maturin develop --release`.
  - Run `uv sync --reinstall-package pixelgen-pixelator-core`
- Architecture mismatch (especially on macOS):
  - Ensure Python interpreter architecture matches your target wheel/build.

## Development

From the repository root, run Python binding tests with:

```bash
bash scripts/test-python-bindings.sh
```

For a manual build-only flow in `packages/python_bindings`:

```bash
cd packages/python_bindings
uv sync --extra dev --no-install-project
uv run maturin develop
uv sync --refresh --extra dev
```

## License

This package is part of the `pixelator-core` repository and follows the same license.
