"""Guard the workspace version on release PRs into main.

The version in the root Cargo.toml is the single source of truth for the Rust
crates and the Python wheel, so releasing means bumping exactly one number. This
checks that the bump actually happened and is a usable release: a final version,
higher than the one on main, that has not been tagged already.

Reads BASE_SHA from the environment and exits non-zero on any violation.
"""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.version import InvalidVersion, Version


def workspace_version(cargo_toml: str) -> str:
    return tomllib.loads(cargo_toml)["workspace"]["package"]["version"]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, check=True, text=True).stdout


def released_versions() -> set[Version]:
    """Versions of every tag in the repo, ignoring tags that are not versions."""
    versions = set()
    for tag in git("tag", "--list").split():
        try:
            versions.add(Version(tag.removeprefix("v")))
        except InvalidVersion:
            continue
    return versions


def problems(head_raw: str, base_raw: str, released: set[Version]) -> list[str]:
    """Reasons head_raw cannot be released, worded with the Cargo.toml spelling."""
    head, base = Version(head_raw), Version(base_raw)

    found = []
    if head.is_prerelease:
        found.append(f"{head_raw} is a pre-release; main only takes final releases")
    if head <= base:
        found.append(f"{head_raw} must be strictly greater than {base_raw} on main")
    if head in released:
        found.append(f"{head_raw} is already tagged; pick a version that has not been released")
    return found


def main() -> int:
    base_sha = os.environ["BASE_SHA"]

    head_raw = workspace_version(Path("Cargo.toml").read_text())
    base_raw = workspace_version(git("show", f"{base_sha}:Cargo.toml"))
    print(f"base (main) version: {base_raw}")
    print(f"head version:        {head_raw}")

    try:
        found = problems(head_raw, base_raw, released_versions())
    except InvalidVersion as err:
        print(f"Error: Cargo.toml does not hold a valid version: {err}")
        return 1

    if found:
        print("\nError: workspace.package.version in Cargo.toml is not ready for release:")
        for problem in found:
            print(f"  - {problem}")
        return 1

    print(f"\nOK: {head_raw} is a new final release above {base_raw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
