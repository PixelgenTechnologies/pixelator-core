# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-10-29

### Changed

- `graph_collapse` takes a string `component` column, sorted by component.
- `graph_collapse` returns only `original` and `corrected` columns.

### Fixed

- Debug messages in community detection don't panic on empty graphs

## [1.0.1] - 2026-10-29

### Removed

- Python 3.10 support.

### Fixed

- Windows wheels failed to build, so 1.0.0 shipped without wheels.

## [1.0.0] - 2026-10-29

### Added

- Edgelist binding
- Graph collapse binding
- Optional Parquet of reads dropped while building an edgelist.

### Changed

- Faster, lower-memory label propagation, connected components, and graph statistics on large graphs.
- Building a graph from a parquet edge list reads the file only once.
- Leiden is faster and uses less peak memory on large graphs.

### Fixed

- UMI keys are now mixed before hashing to prevent UMI bias increase hash collision likelihood.
- Release the Python GIL during native execution in Python bindings.

## [0.2.0] - 2026-09-14

### Added

- Expose merge threshold in python bindings. This makes it possible to merge
  communities when they share many edges together after Leiden.
- Expose `run_connected_components` in python bindings, making it possible to
  find connected components of a graph directly from Python.

### Changed

- Make it possible to configure arrow writer when writing batch records to parquet.
- Lower peak memory while building a `Graph` by ~65%.

### Deprecated

### Removed

### Fixed

### Security

---
## [0.1.1] - 2026-04-23

### Fixed

- Fix output for Leiden's python binding

## [0.1.0] - 2026-04-16

Initial release: Rust implementation of performance-critical community-detection and graph routines from [pixelator](https://github.com/PixelgenTechnologies/pixelator), with Python bindings, a CLI, and published wheels.

### Added

- **Rust library (`pixelator_core`)**
  - Fast label propagation with pluggable assignment strategies and Parquet-based graph I/O.
  - Leiden community detection with a modularity quality function on weighted partitioned graphs.
  - Hybrid pipeline combining fast label propagation, graph aggregation, optional Leiden refinement, and multiplet recovery (`new_graph`).
  - Shared utilities: graph types and statistics, node partitioning (fast and Leiden-specific), linear algebra helpers, Parquet read/write for edge lists and node partitions.
- **CLI (`community-detection`)**
  - `flp` — run fast label propagation on a Parquet edge list and write node partitions.
  - `leiden` — run Leiden with configurable resolution and write partitions.
  - `stats` — print basic graph statistics (nodes, edge weight sum, components, largest-component fraction).
- **Python bindings (`pixelator_core` / package name `pixelgen-pixelator-core`)**
  - `run_label_propagation`, `run_leiden`, `run_hybrid_community_detection`, `find_graph_statistics`, and `GraphProperties`-style results for hybrid workflows.
  - Module `__version__` aligned with the Rust crate version.

[Unreleased]: https://github.com/PixelgenTechnologies/pixelator-core/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/PixelgenTechnologies/pixelator-core/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/PixelgenTechnologies/pixelator-core/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/PixelgenTechnologies/pixelator-core/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/PixelgenTechnologies/pixelator-core/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/PixelgenTechnologies/pixelator-core/releases/tag/v0.1.1
[0.1.0]: https://github.com/PixelgenTechnologies/pixelator-core/releases/tag/v0.1.0
