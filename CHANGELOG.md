# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0](https://github.com/PixelgenTechnologies/pixelator-core/compare/v2.0.0...v3.0.0) (2026-11-26)


### ⚠ BREAKING CHANGES

* make graph-collapse work on batched components. ([#53](https://github.com/PixelgenTechnologies/pixelator-core/issues/53))

### Features

* Enable ctrl+c termination in graph commands ([#57](https://github.com/PixelgenTechnologies/pixelator-core/issues/57)) ([90992fc](https://github.com/PixelgenTechnologies/pixelator-core/commit/90992fcdd4c8dcbec6ff28ddd890613b9dbf090f))
* make graph-collapse work on batched components. ([#53](https://github.com/PixelgenTechnologies/pixelator-core/issues/53)) ([a5cc337](https://github.com/PixelgenTechnologies/pixelator-core/commit/a5cc3372c7b33e34990ee3bec771100e62178165))


### Bug Fixes

* **deps:** bump cc to 1.11.1 to fix Windows wheel build ([#61](https://github.com/PixelgenTechnologies/pixelator-core/issues/61)) ([399d933](https://github.com/PixelgenTechnologies/pixelator-core/commit/399d933032d8303905c3599efa1802bdc7801991))

## [2.0.0](https://github.com/PixelgenTechnologies/pixelator-core/compare/v1.0.2...v2.0.0) (2026-11-18)


### ⚠ BREAKING CHANGES

* support Python 3.12 to 3.14 and drop Python 3.11 ([#54](https://github.com/PixelgenTechnologies/pixelator-core/issues/54))

### Features

* **edgelist:** add marker_sequence_1 and marker_sequence_2 columns to the edgelist Parquet ([#48](https://github.com/PixelgenTechnologies/pixelator-core/issues/48)) ([c2c6f83](https://github.com/PixelgenTechnologies/pixelator-core/commit/c2c6f83eff421bedefb99b34cd200fd5110d7f02))
* support Python 3.12 to 3.14 and drop Python 3.11 ([#54](https://github.com/PixelgenTechnologies/pixelator-core/issues/54)) ([4fc291d](https://github.com/PixelgenTechnologies/pixelator-core/commit/4fc291dfaa955e29f2f70d68026fec4c88402149))

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
