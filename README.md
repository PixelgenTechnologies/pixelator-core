# Pixelator-core

[![build](https://github.com/PixelgenTechnologies/pixelator-core/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/PixelgenTechnologies/pixelator-core/actions/workflows/build.yml)
[![tests](https://github.com/PixelgenTechnologies/pixelator-core/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/PixelgenTechnologies/pixelator-core/actions/workflows/test.yml)
[![lint](https://github.com/PixelgenTechnologies/pixelator-core/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/PixelgenTechnologies/pixelator-core/actions/workflows/lint.yml)
[![benchmark](https://github.com/PixelgenTechnologies/pixelator-core/actions/workflows/bench.yml/badge.svg?branch=main)](https://pixelgentechnologies.github.io/pixelator-core/dev/bench/index.html)
[![docs](https://img.shields.io/badge/docs-GitHub_Pages-blue.svg?style=flat-square)](https://pixelgentechnologies.github.io/pixelator-core/dev/docs/)
[![wheels](https://github.com/PixelgenTechnologies/pixelator-core/actions/workflows/wheels.yml/badge.svg?branch=main)](https://github.com/PixelgenTechnologies/pixelator-core/actions/workflows/wheels.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust edition](https://img.shields.io/badge/Rust-2024%20edition-dea584)](https://doc.rust-lang.org/edition-guide/rust-2024/index.html)

This repository implements the core, compute-heavy functions from [`pixelator`](https://github.com/PixelgenTechnologies/pixelator) in Rust. Those routines are exposed through Python bindings as well as a CLI and the `pixelator-core` library so you can run performance-critical steps from either Python or the command line.

<br />

<div align="center"><a href="https://pixelgentechnologies.github.io/pixelator-core/dev/docs/">Documentation</a></div>

<br />

## Building the packages

Build all packages as follows (skip `--release` if you don't want an optimized build):

```
cargo build --release
```

After building the cli will be in `target/release/community-detection` and can be run with:

```
./target/release/community-detection --help
```

## Python bindings

The python bindings are in the `packages/python_bindings` folder. To get started (using `uv`) use:

```bash
cd packages/python_bindings
uv run maturin develop --release
```

This will build the bindings and install them in your current python environment. You can then use them as follows:

```python
from pixelator_core_py import run_label_propagation

run_label_propagation(
    parquet_file="./edgelist.parquet",
    epochs=1,
    output="./output.parquet",
)
```

If you want to build a wheel file, e.g. to install elsewhere, you can do so with:

```bash
cd packages/python_bindings
maturin build --release --out dist/
```

A wheel file will be created in the `dist/` folder after building. You can install it in other environments with pip:

```bash
pip install path/to/wheel/file.whl
```

## Benchmark tests

We have some basic benchmarking in-place which is useful to get a feel for the performance of different potential choices. To run the benchmarks use:

```
cargo bench
```

And you will get some output like:

```
Timer precision: 10 ns
pixelator_core_benchmarks              fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ bench_community_detection                         │               │               │               │         │
│  ├─ bench_fast_label_propagation     2.617 ms      │ 7.626 ms      │ 3.233 ms      │ 3.799 ms      │ 100     │ 100
│  ├─ bench_leiden_cpm                 (ignored)     │               │               │               │         │
│  ├─ bench_leiden_modularity          14.41 ms      │ 17.03 ms      │ 14.54 ms      │ 14.59 ms      │ 100     │ 100
│  ╰─ bench_leiden_modularity_medium   14.05 s       │ 14.87 s       │ 14.19 s       │ 14.37 s       │ 3       │ 3
╰─ bench_parquet_io                                  │               │               │               │         │
   ├─ bench_create_graph_from_parquet  1.229 s       │ 1.47 s        │ 1.246 s       │ 1.286 s       │ 10      │ 10
   ├─ bench_parquet_reading            242.5 ms      │ 269.1 ms      │ 250.9 ms      │ 252.5 ms      │ 10      │ 10
   ╰─ bench_parquet_writing            213 ms        │ 251.5 ms      │ 224.4 ms      │ 224.9 ms      │ 10      │ 10
```

To add more benchmarks see the `packages/pixelator_core/benches/main.rs` file.

The benchmarks are automatically run on each pull request and uploaded to https://pixelgentechnologies.github.io/pixelator-core/dev/bench/index.html
