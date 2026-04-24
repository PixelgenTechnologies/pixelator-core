# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Expose merge threshold in python bindings. This makes it possible to merge
  communities when they share many edges together after Leiden.

### Changed

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

[Unreleased]: https://github.com/PixelgenTechnologies/pixelator-core/compare/v0.1.1...HEAD
[0.1.0]: https://github.com/PixelgenTechnologies/pixelator-core/releases/tag/v0.1.1
[0.1.0]: https://github.com/PixelgenTechnologies/pixelator-core/releases/tag/v0.1.0
