from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from radar.structure import correlation as corr_mod
from radar.structure import distance as dist_mod
from radar.structure import mst as mst_mod
from tests.test_correlation import factor_returns


def tree_from(rets: pd.DataFrame, estimator: str = "sample") -> nx.Graph:
    corr = corr_mod.estimate_correlation(rets, estimator)
    return mst_mod.build_mst(dist_mod.correlation_to_distance(corr.matrix))


def test_mst_is_a_spanning_tree():
    rets = factor_returns(20, 400, n_blocks=4, market_loading=0.6, block_loading=0.5)
    tree = tree_from(rets)
    assert tree.number_of_nodes() == 20
    assert tree.number_of_edges() == 19
    assert nx.is_connected(tree)
    assert nx.is_tree(tree)


def test_known_small_example():
    """A chain: X-Y is closest, then Y-Z. X-Z must not be chosen."""
    d = pd.DataFrame(
        [[0.0, 0.1, 0.9], [0.1, 0.0, 0.2], [0.9, 0.2, 0.0]],
        index=list("XYZ"), columns=list("XYZ"),
    )
    tree = mst_mod.build_mst(d)
    assert mst_mod.edge_set(tree) == {frozenset("XY"), frozenset("YZ")}


def test_zero_distance_edge_is_not_dropped():
    """The reason this uses networkx rather than scipy.sparse.csgraph: sparse routines
    read a zero weight as 'no edge', and perfectly correlated assets have distance 0."""
    d = pd.DataFrame(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        index=list("ABC"), columns=list("ABC"),
    )
    tree = mst_mod.build_mst(d)
    assert nx.is_connected(tree)
    assert frozenset("AB") in mst_mod.edge_set(tree)


def test_empty_and_malformed_input_rejected():
    with pytest.raises(ValueError, match="empty"):
        mst_mod.build_mst(pd.DataFrame())
    with pytest.raises(ValueError, match="NaN"):
        mst_mod.build_mst(pd.DataFrame([[0.0, np.nan], [np.nan, 0.0]],
                                       index=list("AB"), columns=list("AB")))


# ---------------------------------------------------------------------------
# The central descriptive claim: block structure emerges without labels.
# ---------------------------------------------------------------------------


def test_sector_structure_emerges_without_labels():
    """Returns are generated with four hidden blocks. Labels are used only to score the
    tree, never to build it, so a high lift is evidence the MST found the structure."""
    n_assets, n_blocks = 40, 4
    rets = factor_returns(
        n_assets, 1000, n_blocks=n_blocks, market_loading=0.4, block_loading=1.0, seed=1
    )
    groups = {f"A{i:03d}": f"block{i % n_blocks}" for i in range(n_assets)}

    tree = tree_from(rets)
    result = mst_mod.group_purity(tree, groups)

    assert result["n_edges"] == n_assets - 1
    assert result["baseline"] == pytest.approx(0.231, abs=0.01)  # ~1/n_blocks
    assert result["purity"] > 0.8
    assert result["lift"] > 3.0


def test_group_purity_is_near_baseline_when_labels_are_meaningless():
    rets = factor_returns(40, 1000, n_blocks=1, market_loading=0.5, seed=2)
    random_labels = {f"A{i:03d}": f"block{i % 4}" for i in range(40)}
    result = mst_mod.group_purity(tree_from(rets), random_labels)
    assert result["lift"] < 2.0


def test_group_purity_handles_unlabelled_nodes():
    rets = factor_returns(10, 300, n_blocks=2, block_loading=0.8)
    tree = tree_from(rets)
    result = mst_mod.group_purity(tree, {"A000": "x", "A001": "x"})
    assert result["n_edges"] <= 1


# ---------------------------------------------------------------------------
# Stability -- the metric that keeps scrubber animation honest.
# ---------------------------------------------------------------------------


def test_edge_survival_is_one_for_an_identical_tree():
    tree = tree_from(factor_returns(15, 500, n_blocks=3, block_loading=0.8))
    assert mst_mod.edge_survival(tree, tree) == pytest.approx(1.0)


def test_edge_survival_falls_when_the_tree_changes():
    a = tree_from(factor_returns(20, 300, n_blocks=4, block_loading=0.9, seed=1))
    b = tree_from(factor_returns(20, 300, n_blocks=4, block_loading=0.9, seed=999))
    survival = mst_mod.edge_survival(a, b)
    assert 0.0 <= survival < 1.0


def test_independent_windows_of_pure_noise_churn_heavily():
    """The reason this metric exists: MST topology turns over substantially from
    estimation noise alone, so churn is not evidence of a regime change."""
    a = tree_from(factor_returns(30, 120, market_loading=0.0, seed=10))
    b = tree_from(factor_returns(30, 120, market_loading=0.0, seed=11))
    assert mst_mod.edge_survival(a, b) < 0.3


def test_edge_survival_ignores_universe_membership_changes():
    rets = factor_returns(15, 500, n_blocks=3, block_loading=0.8, seed=3)
    full = tree_from(rets)
    reduced = tree_from(rets.drop(columns=["A014"]))
    # Comparison is over shared nodes only, so dropping a member is not spurious churn.
    assert mst_mod.edge_survival(full, reduced) > 0.5


def test_edge_survival_undefined_without_overlap():
    a = tree_from(factor_returns(5, 200))
    b = a.copy()
    b = nx.relabel_nodes(b, {n: f"Z{n}" for n in b.nodes()})
    assert np.isnan(mst_mod.edge_survival(a, b))


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def test_tree_length_shortens_as_correlation_rises():
    """Crisis behaviour in one number: stronger common movement -> shorter tree."""
    calm = tree_from(factor_returns(30, 800, market_loading=0.2, seed=6))
    crisis = tree_from(factor_returns(30, 800, market_loading=2.0, seed=6))
    assert mst_mod.normalised_tree_length(crisis) < mst_mod.normalised_tree_length(calm)


def test_degrees_and_hub_agree():
    tree = tree_from(factor_returns(25, 600, n_blocks=5, market_loading=0.6, block_loading=0.5))
    deg = mst_mod.degrees(tree)
    assert deg.sum() == 2 * tree.number_of_edges()
    assert list(deg.values) == sorted(deg.values, reverse=True)
    assert mst_mod.hub(tree) == deg.index[0]


def test_to_frame_is_sorted_shortest_first():
    tree = tree_from(factor_returns(12, 400, n_blocks=3, block_loading=0.7))
    frame = mst_mod.to_frame(tree)
    assert list(frame.columns) == ["source", "target", "distance"]
    assert len(frame) == 11
    assert frame["distance"].is_monotonic_increasing
