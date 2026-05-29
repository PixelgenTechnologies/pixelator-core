# `pixelator_core` Python Bindings

[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/PixelgenTechnologies/pixelator-core/actions/workflows/wheels.yml)

Python bindings for `pixelator-core`, exposing Rust implementations of graph
statistics and community detection algorithms for fast execution from Python.

## Features

- Compute graph-level statistics.
- Run community detection with:
  - Fast Label Propagation (FLP)
  - Leiden
  - Hybrid FLP + Leiden flow (`run_hybrid_community_detection`)

> NB: There are two main types of APIs provided by this package: parquet-backed APIs workflows where
> memory-efficiency is a high priority and dedicated NetworkX APIs for in-memory graphs.

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

## Quick Start

```python
from pixelator_core import (
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

### NetworkX usage

NetworkX APIs are provided for easy of use, for scenarios when the lowest possible
memory usage is not a priority.

```python
import networkx as nx
from pixelator_core import run_label_propagation_networkx, run_leiden_networkx

G = nx.Graph()
G.add_edge("cell_a", "cell_b", weight=2)
G.add_edge("cell_b", "cell_c")
G.add_node("isolated_cell")

flp_communities = run_label_propagation_networkx(G, epochs=2)
leiden_communities = run_leiden_networkx(
    G,
    resolution=1.0,
    randomness=0.1,
    seed=42,
)

# Both return:
# A list of disjoint sets (partition of G). Each set represents one community.
# All communities together contain all the nodes in G.
print(flp_communities)
print(leiden_communities)
```

Weight constraints for NetworkX APIs:
- `run_leiden_networkx` accepts non-negative integer-like weights and rejects fractional weights (for example `0.5`).
- `run_label_propagation_networkx` uses the same non-negative integer-like requirement and also requires weights to fit in `u8` (`0..=255`).

## Input and Output

- `find_graph_statistics`, `run_label_propagation`, `run_leiden`, and `run_hybrid_community_detection`
  expect an edge-list Parquet file compatible with `pixelator-core`.
- `run_label_propagation_networkx` and `run_leiden_networkx` accept a `networkx.Graph` object
  and return communities in-memory as a list of disjoint sets over original node labels.
- NetworkX edge weights for Leiden/FLP must be non-negative integer-like values; FLP additionally enforces the `u8` range (`0..=255`).
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
