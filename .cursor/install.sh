#!/usr/bin/env bash
#
# Idempotent Cloud Agent bootstrap for pixelator-core.
# Prepares the Rust toolchain, uv, git-LFS fixtures, and a warm build.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# The workspace targets Rust 2024 edition; pin the same toolchain the release
# Dockerfile uses (rust:1.90) so cargo fmt/clippy match the maintainers' setup.
rustup toolchain install 1.90 --profile minimal --no-self-update
rustup component add rustfmt clippy --toolchain 1.90
rustup default 1.90

# uv drives the Python bindings build/test flow (see scripts/test-python-bindings.sh).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Test fixtures (parquet graphs) are stored in git LFS and are required by the
# benchmarks, CLI runs, and the pytest fixture for the bindings.
git lfs pull

# Warm the Rust build so the workspace is immediately usable.
cargo build --all-targets --locked

# Build and install the Python extension into a local uv environment so
# `pixelator_core` is importable without a manual maturin step.
( cd packages/python_bindings && uv sync --extra dev --no-install-project && uv run maturin develop )
