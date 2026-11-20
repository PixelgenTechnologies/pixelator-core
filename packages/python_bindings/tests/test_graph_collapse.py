"""Tests for the `graph_collapse` binding."""

import numpy as np
import pixelator_core
import pytest


def _umi(seq):
    return pixelator_core.encode_dna_2bit(seq)


def _collapse(rows, **kwargs):
    """Run `graph_collapse` on `(umi1, umi2, component)` rows.

    `component` is an integer id. The rows are passed as the three numpy
    arrays the binding takes.
    """
    umi1 = np.array([row[0] for row in rows], dtype=np.uint64)
    umi2 = np.array([row[1] for row in rows], dtype=np.uint64)
    component = np.array([row[2] for row in rows], dtype=np.uint32)
    kwargs.setdefault("n_threads", 1)
    return pixelator_core.graph_collapse(umi1, umi2, component, **kwargs)


def _pairs(renames):
    return set(zip(renames["original"].tolist(), renames["corrected"].tolist()))


def test_chain_collapses_within_its_component_only():
    """A leaf chain collapses onto its end, and a second component does not.

    CCAA is one symbol from CAAA, and CAAA is one symbol from AAAA, the end of
    the chain. CAAA is also one symbol from CCAA; AAAA comes first because rows
    are searched in UMI order. GGGG and TTTT are umi1, and the other UMIs are
    umi2, so no UMI sits on both sides. The other component is a single edge
    between distant UMIs, and its row sits among the chain's rows.
    """
    umi_m = _umi("GGGG")
    umi_h = _umi("AAAA")
    umi_a = _umi("CCAA")
    umi_b = _umi("CAAA")
    umi_n = _umi("TTTT")
    rows = [
        (umi_m, umi_h, 0),
        (umi_m, umi_a, 0),
        (_umi("ACAC"), _umi("GTGT"), 1),
        (umi_m, umi_b, 0),
        (umi_n, umi_h, 0),
    ]

    renames, _stats = _collapse(rows, bfs_threshold=2, hamming_threshold=1)

    assert list(renames) == ["component", "original", "corrected"]
    assert renames["component"].dtype == np.uint32
    assert renames["original"].dtype == np.uint64
    assert renames["corrected"].dtype == np.uint64
    assert renames["component"].tolist() == [0, 0]
    assert _pairs(renames) == {(umi_a, umi_h), (umi_b, umi_h)}


def test_renames_are_sorted_by_component_then_original():
    """The same chain in components 5 and 2 is reported per component, in order."""
    umi_m = _umi("GGGG")
    umi_h = _umi("AAAA")
    umi_a = _umi("CCAA")
    umi_b = _umi("CAAA")
    umi_n = _umi("TTTT")
    chain = [(umi_m, umi_h), (umi_m, umi_a), (umi_m, umi_b), (umi_n, umi_h)]
    rows = [(u1, u2, c) for u1, u2 in chain for c in (5, 2)]

    renames, stats = _collapse(rows, bfs_threshold=2, hamming_threshold=1)

    first, second = sorted([umi_a, umi_b])
    assert list(
        zip(
            renames["component"].tolist(),
            renames["original"].tolist(),
            renames["corrected"].tolist(),
        )
    ) == [(2, first, umi_h), (2, second, umi_h), (5, first, umi_h), (5, second, umi_h)]
    assert stats["total_components"] == 2


def test_stats_count_what_the_search_did():
    """The stats dict covers every component, with the keys the report will use.

    This is the chain from the test above, plus the second component. Component 0
    has three leaves (CCAA, CAAA and TTTT) and component 1 has two. Only
    CCAA -> CAAA and CAAA -> AAAA are found, each two hops and one symbol apart.
    """
    umi_m = _umi("GGGG")
    umi_h = _umi("AAAA")
    umi_a = _umi("CCAA")
    umi_b = _umi("CAAA")
    umi_n = _umi("TTTT")
    rows = [
        (umi_m, umi_h, 0),
        (umi_m, umi_a, 0),
        (umi_m, umi_b, 0),
        (umi_n, umi_h, 0),
        (_umi("ACAC"), _umi("GTGT"), 1),
    ]

    _renames, stats = _collapse(rows, bfs_threshold=2, hamming_threshold=1)

    assert stats == {
        "total_components": 2,
        "nodes_considered": 5,
        "targets_found": 2,
        "bfs_distance_distribution": {2: 2},
        "hamming_distance_distribution": {1: 2},
    }


def test_stats_distributions_are_in_ascending_distance_order():
    """Dict insertion order carries the ordering, so it must be ascending.

    Component 0 is the path AAAA - TTTT - GGGG - CGCG - CAAA - GCGC, where the
    umi1 ends AAAA and CAAA are four hops and one symbol apart, and AAAA is the
    only leaf with a match. Component 1 is a star around GGGG with leaves AAAA
    and CCAA, two hops and two symbols apart, of which one match is kept.
    Component 0 comes first, so a tally in arrival order would list distance 4
    before 2.
    """
    rows = [
        (_umi("AAAA"), _umi("TTTT"), 0),
        (_umi("GGGG"), _umi("TTTT"), 0),
        (_umi("GGGG"), _umi("CGCG"), 0),
        (_umi("CAAA"), _umi("CGCG"), 0),
        (_umi("CAAA"), _umi("GCGC"), 0),
        (_umi("GGGG"), _umi("AAAA"), 1),
        (_umi("GGGG"), _umi("CCAA"), 1),
    ]

    _renames, stats = _collapse(rows, bfs_threshold=4, hamming_threshold=2)

    assert stats["total_components"] == 2
    assert stats["nodes_considered"] == 4
    assert stats["targets_found"] == 2
    assert list(stats["bfs_distance_distribution"].items()) == [(2, 1), (4, 1)]
    assert list(stats["hamming_distance_distribution"].items()) == [(1, 1), (2, 1)]


def test_max_degree_skips_nodes_above_the_cutoff():
    """A degree-2 node is searched only when max_degree is at least 2.

    AAAA has two neighbors and is one symbol from CAAA, which has three. Every
    other node is four symbols from the candidates on its own side, so the
    default (leaves only) collapses nothing.
    """
    umi_d = _umi("AAAA")
    umi_s = _umi("CAAA")
    rows = [
        (umi_d, _umi("GGGG"), 0),
        (umi_d, _umi("TTTT"), 0),
        (umi_s, _umi("GGGG"), 0),
        (umi_s, _umi("CCCC"), 0),
        (umi_s, _umi("ATAT"), 0),
    ]

    leaves_only, leaves_only_stats = _collapse(
        rows, bfs_threshold=2, hamming_threshold=1
    )
    with_degree_two, with_degree_two_stats = _collapse(
        rows, bfs_threshold=2, hamming_threshold=1, max_degree=2
    )

    assert _pairs(leaves_only) == set()
    assert _pairs(with_degree_two) == {(umi_d, umi_s)}
    # The three leaves are searched either way. Degree 2 adds AAAA and GGGG.
    assert leaves_only_stats["nodes_considered"] == 3
    assert leaves_only_stats["targets_found"] == 0
    assert with_degree_two_stats["nodes_considered"] == 5
    assert with_degree_two_stats["targets_found"] == 1


def test_row_order_does_not_change_the_result():
    """A leaf with two equally good targets gets the same one for any row order.

    AAAA is a leaf of GGGG. CAAA and GAAA are also neighbors of GGGG, each one
    symbol from AAAA, and each has a second neighbor, so neither is a leaf. Which
    one the search meets first depends on the neighbor order of GGGG, which would
    otherwise follow the row order.
    """
    rows = [
        (_umi("GGGG"), _umi("AAAA"), 0),
        (_umi("GGGG"), _umi("CAAA"), 0),
        (_umi("GGGG"), _umi("GAAA"), 0),
        (_umi("TTTT"), _umi("CAAA"), 0),
        (_umi("ACAC"), _umi("GAAA"), 0),
    ]
    orders = [rows, rows[::-1], [rows[i] for i in (2, 0, 4, 1, 3)]]

    results = [
        _pairs(_collapse(order, bfs_threshold=2, hamming_threshold=1)[0])
        for order in orders
    ]

    assert len(results[0]) == 1
    assert results[1] == results[0]
    assert results[2] == results[0]


def test_empty_edgelist_returns_empty_renames():
    """No rows still produce the three typed arrays and all-zero stats."""
    renames, stats = _collapse([])

    assert {name: array.dtype for name, array in renames.items()} == {
        "component": np.uint32,
        "original": np.uint64,
        "corrected": np.uint64,
    }
    assert all(len(array) == 0 for array in renames.values())
    assert stats == {
        "total_components": 0,
        "nodes_considered": 0,
        "targets_found": 0,
        "bfs_distance_distribution": {},
        "hamming_distance_distribution": {},
    }


def test_arrays_of_different_lengths_are_a_value_error():
    umi1 = np.array([1, 2], dtype=np.uint64)
    umi2 = np.array([3, 4], dtype=np.uint64)
    component = np.array([0], dtype=np.uint32)

    with pytest.raises(ValueError, match="must have the same length, got 2, 2 and 1"):
        pixelator_core.graph_collapse(umi1, umi2, component, n_threads=1)


def test_a_component_of_another_dtype_is_a_type_error():
    umi1 = np.array([1], dtype=np.uint64)
    umi2 = np.array([2], dtype=np.uint64)
    component = np.array([0], dtype=np.int64)

    with pytest.raises(TypeError):
        pixelator_core.graph_collapse(umi1, umi2, component, n_threads=1)
