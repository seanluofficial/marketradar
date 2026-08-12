from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import leaves_list

from radar.structure import hierarchy as cl
from radar.structure.distance import correlation_to_distance
from tests.helpers import factor_returns


def blocky_distance(n_blocks: int = 3, per_block: int = 5) -> pd.DataFrame:
    rets = factor_returns(
        n_blocks * per_block, 2000, n_blocks=n_blocks, market_loading=0.3,
        block_loading=1.2, seed=1,
    )
    return correlation_to_distance(rets.corr())


def test_linkage_shape_and_labels():
    d = blocky_distance()
    result = cl.cluster_from_distance(d)
    assert result.linkage.shape == (len(d) - 1, 4)
    assert result.labels == list(d.columns)
    assert sorted(result.order) == sorted(d.columns)
    assert len(result.order) == len(d)


def test_quasi_diagonal_order_matches_scipy_leaves_list():
    """The seriation is implemented explicitly so the allocator depends on inspectable
    code, but it must agree with scipy's own leaf ordering."""
    for method in cl.LINKAGE_METHODS:
        d = blocky_distance()
        link = cl.linkage(cl.condensed_distance(d), method=method)
        assert cl.quasi_diagonal_order(link) == [int(i) for i in leaves_list(link)]


def test_seriation_groups_blocks_contiguously():
    """The property HRP relies on: correlated assets end up adjacent, so the reordered
    correlation matrix is quasi-diagonal."""
    n_blocks, per_block = 3, 5
    d = blocky_distance(n_blocks, per_block)
    order = cl.cluster_from_distance(d).order

    block_of = {f"A{i:03d}": i % n_blocks for i in range(n_blocks * per_block)}
    runs = [block_of[t] for t in order]
    # A perfectly grouped ordering changes block exactly n_blocks - 1 times.
    switches = sum(1 for a, b in zip(runs, runs[1:]) if a != b)
    assert switches == n_blocks - 1


def test_flat_clusters_recover_known_blocks():
    n_blocks, per_block = 3, 5
    d = blocky_distance(n_blocks, per_block)
    labels = cl.cluster_from_distance(d).flat_clusters(n_blocks)

    truth = pd.Series(
        {f"A{i:03d}": i % n_blocks for i in range(n_blocks * per_block)}
    )
    assert cl.cluster_stability(labels, truth) == pytest.approx(1.0)


def test_cophenetic_correlation_is_high_for_genuine_hierarchy():
    assert cl.cluster_from_distance(blocky_distance()).cophenetic_correlation > 0.7


def test_cophenetic_correlation_is_lower_for_structureless_data():
    rets = factor_returns(15, 2000, n_blocks=1, market_loading=0.0, seed=4)
    flat = cl.cluster_from_distance(correlation_to_distance(rets.corr()))
    assert flat.cophenetic_correlation < cl.cluster_from_distance(
        blocky_distance()
    ).cophenetic_correlation


def test_depth_ratio_is_larger_for_a_single_dominant_merge():
    """The diagnostic that says 'hierarchical methods have nothing to exploit here'."""
    nested = cl.cluster_from_distance(blocky_distance(n_blocks=3, per_block=5))
    # One asset far from a tight cluster -> the final merge dominates the tree height.
    n = 12
    corr = np.full((n, n), 0.95)
    np.fill_diagonal(corr, 1.0)
    corr[0, 1:] = corr[1:, 0] = 0.0
    labels = [f"X{i}" for i in range(n)]
    shallow = cl.cluster_from_distance(
        correlation_to_distance(pd.DataFrame(corr, index=labels, columns=labels))
    )
    assert shallow.depth_ratio() > nested.depth_ratio()


# ---------------------------------------------------------------------------
# cluster_stability
# ---------------------------------------------------------------------------


def test_identical_partitions_score_one():
    labels = pd.Series([0, 0, 1, 1, 2], index=list("ABCDE"))
    assert cl.cluster_stability(labels, labels) == pytest.approx(1.0)


def test_relabelled_partitions_still_score_one():
    """ARI compares partitions, not label values."""
    a = pd.Series([0, 0, 1, 1], index=list("ABCD"))
    b = pd.Series([7, 7, 3, 3], index=list("ABCD"))
    assert cl.cluster_stability(a, b) == pytest.approx(1.0)


def test_random_partitions_score_near_zero():
    rng = np.random.default_rng(0)
    idx = [f"A{i}" for i in range(200)]
    a = pd.Series(rng.integers(0, 4, 200), index=idx)
    b = pd.Series(rng.integers(0, 4, 200), index=idx)
    assert abs(cl.cluster_stability(a, b)) < 0.1


def test_stability_uses_only_shared_assets():
    a = pd.Series([0, 0, 1, 1], index=list("ABCD"))
    b = pd.Series([0, 0, 1], index=list("ABC"))
    assert cl.cluster_stability(a, b) == pytest.approx(1.0)
    assert np.isnan(cl.cluster_stability(a, pd.Series([0], index=["Z"])))


# ---------------------------------------------------------------------------
# The export surface the allocator consumes
# ---------------------------------------------------------------------------


def test_clustering_returns_the_linkage_and_order_pair():
    rets = factor_returns(15, 600, n_blocks=3, market_loading=0.4, block_loading=1.0)
    result = cl.clustering(rets)
    assert result.linkage.shape == (14, 4)
    assert sorted(result.order) == sorted(rets.columns)


def test_asof_truncates_and_prevents_lookahead():
    rets = factor_returns(10, 600, n_blocks=2, block_loading=0.9, seed=6)
    cut = rets.index[300]

    truncated = cl.clustering(rets, asof=cut)
    direct = cl.clustering(rets.loc[:cut])
    np.testing.assert_allclose(truncated.linkage, direct.linkage)

    # And it must differ from using the full sample, or asof is doing nothing.
    full = cl.clustering(rets)
    assert not np.allclose(truncated.linkage, full.linkage)


def test_window_takes_the_trailing_observations():
    rets = factor_returns(10, 600, n_blocks=2, block_loading=0.9, seed=6)
    windowed = cl.clustering(rets, window=100)
    direct = cl.clustering(rets.tail(100))
    np.testing.assert_allclose(windowed.linkage, direct.linkage)


def test_method_and_estimator_are_honoured():
    rets = factor_returns(12, 400, n_blocks=3, block_loading=0.9)
    a = cl.clustering(rets, method="single", estimator="sample")
    b = cl.clustering(rets, method="ward", estimator="sample")
    assert a.method == "single" and b.method == "ward"
    assert not np.allclose(a.linkage, b.linkage)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="centroid|Choose from"):
        cl.cluster_from_distance(blocky_distance(), method="centroid")


def test_too_few_assets_or_observations_rejected():
    d = blocky_distance()
    with pytest.raises(ValueError, match="at least 2 assets"):
        cl.cluster_from_distance(d.iloc[:1, :1])
    rets = factor_returns(5, 100)
    with pytest.raises(ValueError, match="observations available"):
        cl.clustering(rets, asof=rets.index[0])


def test_nan_distance_rejected():
    d = blocky_distance()
    d.iloc[0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cl.cluster_from_distance(d)
