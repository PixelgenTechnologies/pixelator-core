#!/usr/bin/env bash
#
# Profile the `community-detection leiden` binary with hardware counters.
#
# What this script does:
# 1. Builds the project using the custom Cargo profile `profiling`.
# 2. Runs a short "warm up" execution to get caches/JIT-like effects settled.
# 3. Runs the real measurement pinned to CPU core 0 while collecting `perf stat`
#    counters (with `sudo`).
#
# Usage:
#   ./profile.sh <input>
#
# Arguments:
#   <input> - forwarded as the first CLI argument to the `leiden` binary.
#
# Notes / prerequisites:
# - `perf` requires sufficient permissions; this script uses `sudo`.
#

cargo build --profile profiling

echo "Warming up..."
target/profiling/community-detection --debug leiden --resolution 0.1 "$1" > /dev/null

echo "Monitoring run..."
sudo taskset -c 0 perf stat -e instructions,cache-references,cache-misses,cycles \
    target/profiling/community-detection --debug leiden --resolution 0.1 $1
