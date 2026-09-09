# Plan: lower peak memory during graph construction

## Hypothesis

**Confirmed.** Building a `Graph` currently keeps the same edges in memory more than once, and the `TriMat` (COO) step is the main extra copy. On the Parquet load path the edgelist is also fully materialized *before* `TriMat` is filled, so peak usage is even higher than COO + CSR alone.

`Graph::from_edges` is the shared constructor (CLI, Python bindings, tests, Leiden aggregation). Changing it is the highest-leverage fix. Public Python/CLI surfaces do not expose `from_edges`; they can stay as they are.

## Current construction path

### 1. Parquet → graph (CLI / Python)

`create_graph_and_umi_mapping_from_parquet_file` (`packages/pixelator_core/src/common/io.rs`):

1. Stream the Parquet file into `Vec<UMIPair>` (full edgelist, two `usize` per row).
2. Build `UmiToNodeIndexMapping` from that slice (hashmap of unique UMIs — this copy is necessary and stays for the rest of the run).
3. Map the same `Vec` again into `Edge`s and call `Graph::from_edges`.
4. The `Vec<UMIPair>` is not dropped until after the graph exists.

### 2. Edges → CSR (`Graph::from_edges`)

```42:56:packages/pixelator_core/src/common/graph/mod.rs
    pub fn from_edges<I>(edges: I, num_nodes: NodeIdx) -> Self
    where
        I: Iterator<Item = Edge<T>>,
    {
        let mut tri_mat = TriMat::<T>::new((num_nodes, num_nodes));
        for Edge { src, dest, weight } in edges {
            tri_mat.add_triplet(src, dest, weight);
            if src != dest {
                tri_mat.add_triplet(dest, src, weight);
            }
        }

        let adjacency_matrix = tri_mat.to_csr();
        Self::from_adjacency_matrix(adjacency_matrix)
    }
```

Behavior that must be preserved (covered by existing tests):

- Undirected: `(u, v)` with `u != v` is stored as both `(u, v)` and `(v, u)`.
- Self-loops are stored once.
- Repeated input edges are **summed** (sprs COO convention; see `test_edge_metrics`).
- Isolated nodes exist because `num_nodes` sets the matrix shape.

### 3. What `TriMat::to_csr` actually does

sprs `TriMat` is three parallel `Vec`s (`row_inds`, `col_inds`, `data`). Conversion is `&self` (`to_csr`), so the COO buffers stay alive.

`TriMatIter::into_cs` then:

1. Copies every triplet into a new `Vec<(row, col, value)>`.
2. Sorts that vector.
3. Merges duplicate coordinates by adding values.
4. Allocates CSR `indptr` / `indices` / `data`.

So during conversion, the directed edgelist exists as COO **and** as the sort buffer **and** as the emerging CSR.

Leiden aggregation (`build_aggregated_graph`) already yields one undirected copy of each edge, then `from_edges` expands it to two triplets and hits the same path.

## Peak memory (order of magnitude)

Let `E` = number of input undirected edges (including duplicate rows), `n` = nodes, `T = u8`. Ignore allocator slack.

| Live at peak of `from_edges` | Approx. size |
| --- | --- |
| Input `Vec<UMIPair>` (Parquet path only) | `16E` |
| `TriMat` (both directions) | `2E × (8 + 8 + 1) ≈ 34E` |
| sprs sort buffer `Vec<(I, I, T)>` | `2E × ~24 ≈ 48E` (tuple padding) |
| CSR output (both directions, unique nnz) | `~16E` indices+data + `8n` indptr |

**Peak ≈ `16E + 34E + 48E + 16E` ≈ `114E + 8n`**, plus the UMI hashmap.

After construction, only CSR + mapping remain (`~16E + 8n` plus UMIs). Construction is the memory spike.

## Proposed solution

Keep `Graph::from_edges(edges, num_nodes)` and `from_adjacency_matrix`. Replace the `TriMat` implementation; optionally stream Parquet twice so the UMI-pair `Vec` is never held with the graph buffers.

### A. Build CSR without `TriMat` (required)

In `from_edges`:

1. Collect **undirected** triplets only: store `(min(src,dest), max(src,dest), weight)`. Self-loops stay as `(u, u, w)`.
2. `sort_unstable` and merge equal `(u, v)` by adding weights (same semantics as sprs).
3. Count CSR row lengths: each off-diagonal undirected edge contributes 1 to two rows; a self-loop contributes 1 to one row.
4. Allocate CSR `indptr`, `indices`, `data`; fill both directions from the merged undirected list (column indices per row must be sorted — sprs requires that).
5. `CsMat::new(shape, indptr, indices, data)` then `from_adjacency_matrix`.
6. Drop the undirected buffer (end of function). Peak is **undirected COO + CSR**, not COO×2 + sort copy + CSR.

Use `Iterator::size_hint` / `ExactSizeIterator` to `with_capacity` and avoid realloc.

Rough new peak for `from_edges` itself: `~24E` (padded undirected tuples) + `~16E` CSR ≈ **`40E`**, about **3× less** than today’s `from_edges` spike, before counting the Parquet `Vec`.

Do **not** go through sprs `TriMat::to_csr`: that API cannot consume COO and always copies again.

### B. Do not keep the Parquet edgelist while building the graph (recommended, same APIs)

Change `UmiToNodeIndexMapping::from_umi_pairs` to accept an iterator (keep a thin `&[UMIPair]` wrapper if useful).

`create_graph_and_umi_mapping_from_parquet_file`:

1. Open Parquet, stream pairs, build the mapping (no full `Vec`).
2. Open the same file again, map pairs to `Edge`s, call `from_edges`.

Tradeoff: two sequential reads of the edge file vs one large `Vec`. For graphs that already fit, this is the difference between OOM and success. Mapping still needs one pass over all UMIs; that cannot be avoided without a different on-disk index.

### C. Out of scope unless aggregation shows up in profiles

`build_aggregated_graph` can fill a `HashMap<(usize, usize), usize>` of super-edges and emit CSR directly. Supergraphs are much smaller than the input graph; do this only if a profile still shows aggregation as a spike after A+B.

`partitions_as_matrix` also uses `TriMat` (`n` triplets, not `E`). Leave it.

## API

| Surface | Change |
| --- | --- |
| Python / CLI | None |
| `Graph::from_edges` / `from_adjacency_matrix` | Same signatures; internals only |
| `UmiToNodeIndexMapping::from_umi_pairs` | Prefer `impl IntoIterator<Item = UMIPair>`; keep slice via `from_umi_pairs(pairs.iter().copied())` or a deprecated alias |
| `create_graph_and_umi_mapping_from_parquet_file` | Same signature |

Breaking the mapping constructor is acceptable if it simplifies streaming; update in-crate call sites.

## Correctness checks (existing)

Re-run graph unit tests, especially:

- `test_graph_creation_and_stats`
- `test_edge_metrics` (duplicate `(0, 1)` summed)
- self-loop tests (`test_edges_from`, neighbor iterators)
- isolated nodes (`test_graph_neighbors_complex` node 6)
- `create_graph_and_umi_mapping_from_parquet_file` tests
- Leiden / FLP tests (aggregation still goes through `from_edges`)

## Verify memory actually drops

Do not rely on “it should use less.” Measure peak heap of construction in isolation.

1. **Micro-benchmark with an allocation profiler** on `Graph::from_edges` (and separately on `create_graph_and_umi_mapping_from_parquet_file`).
   - Wrap the global allocator with Divan’s `AllocProfiler`, or add a small `peak_alloc` / `dhat` binary under `packages/pixelator_core` (dev-only).
   - Input: synthetic `E` in `{1e6, 5e6, …}` with a known duplicate rate, plus `test_data/mix_40cells_0pc.parquet` (and a larger mix file if CI allows).
   - Record **peak bytes** and **current bytes after `from_edges` returns**.
2. **Assert a bound in a test or ignored CI job** so the regression is mechanical, e.g.
   - peak during `from_edges` `< 3 * E * size_of::<(usize, usize, T)>()` (undirected buffer + CSR + slack), and
   - peak during Parquet load is not `≈ 2 ×` the post-construction graph size plus mapping (that pattern is the old `Vec` + COO stack).
   - Tune constants on the baseline so they fail on the current `TriMat` path and pass after the rewrite.
3. **Process RSS smoke check** (optional, local/CI): `community-detection stats <parquet>` under `/usr/bin/time -v` or `heaptrack` before/after. Confirms mimalloc peak, not just Rust global-allocator accounting.
4. **Time**: two Parquet passes and an extra sort must not blow the existing `bench_create_graph_from_parquet` / `from_edges` benches; keep those in `packages/pixelator_core/benches/main.rs`.

Success: peak construction memory is on the order of **one undirected edgelist + CSR**, the UMI-pair `Vec` is gone from the load path, graphs and partitions match today’s tests, and construction time stays in the same ballpark.

## Implementation order

1. Rewrite `Graph::from_edges` → verify: existing graph tests green; peak-alloc test fails on `dev`, then passes on the rewrite.
2. Stream mapping + second Parquet pass → verify: IO tests; peak of `create_graph_and_umi_mapping_from_parquet_file` drops vs baseline recorded in the same harness.
3. Add/adjust benches for alloc peak (and wall time).
4. Changelog only when the code change ships (this document is the plan, not the behavior change).
