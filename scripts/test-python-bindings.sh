#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_BINDINGS_DIR="${REPO_ROOT}/packages/python_bindings"

cd "${PY_BINDINGS_DIR}"
uv sync --extra dev
uv run maturin develop
uv sync --reinstall-package pixelgen-pixelator-core
uv run pytest tests/ -v
