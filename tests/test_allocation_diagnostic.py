"""Tests for the allocation *diagnostic*.

These check that the reference weight functions are correct enough to support a
conclusion about whether clustering matters. They are not a substitute for the
allocator project's own tests of its own pinned implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.metrics import diagnose as dg
from radar.structure.distance import correlation_to_distance
from radar.structure.hierarchy import cluster_from_distance
from tests.test_correlation import factor_returns


def covariance(rets: pd.DataFrame) -> pd.DataFrame:
    corr = rets.corr()
    sd = rets.std(ddof=1)
    return corr.mul(sd, axis=0).mul(sd, axis=1)


def test_weights_sum_to_one_and_are_non_negative():
    rets = factor_returns(12, 500, n_blocks=3, market_loading=0.4, block_loading=1.0)
    cov = covariance(rets)
    order = cluster_from_distance(correlation_to_distance(rets.corr())).order

    for weights in (dg.hrp_weights(cov, order), dg.inverse_variance_weights(cov, list(cov.columns))):
        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= 0).all()


def test_inverse_variance_favours_the_quiet_asset():
    rets = factor_returns(3, 1000, seed=2)
    rets["A000"] *= 4.0  # make one asset far more volatile
    weights = dg.inverse_variance_weights(covariance(rets), list(rets.columns))
    assert weights["A000"] == weights.min()


def test_hrp_equals_inverse_variance_for_uncorrelated_equal_variance_assets():
    """With no structure and identical variances there is nothing to allocate on, so the
    two methods must agree -- the baseline against which 'HRP adds nothing' is judged."""
    n = 8
    cov = pd.DataFrame(np.eye(n), index=[f"X{i}" for i in range(n)],
                       columns=[f"X{i}" for i in range(n)])
    order = list(cov.columns)
    hrp = dg.hrp_weights(cov, order)
    ivp = dg.inverse_variance_weights(cov, order)
    np.testing.assert_allclose(hrp.to_numpy(), ivp.to_numpy(), atol=1e-12)


def test_hrp_diverges_from_inverse_variance_when_clusters_are_uneven():
    """HRP's actual claim: it declusters. Given one large correlated block and one
    lonely asset, it should hold back weight from the crowded block."""
    n_big = 9
    tickers = [f"B{i}" for i in range(n_big)] + ["LONE"]
    corr = np.eye(n_big + 1)
    corr[:n_big, :n_big] = 0.95
    np.fill_diagonal(corr, 1.0)
    cov = pd.DataFrame(corr, index=tickers, columns=tickers)

    order = [f"B{i}" for i in range(n_big)] + ["LONE"]
    hrp = dg.hrp_weights(cov, order)
    ivp = dg.inverse_variance_weights(cov, tickers)

    assert hrp["LONE"] > ivp["LONE"]
    assert hrp[:n_big].sum() < ivp[:n_big].sum()


def test_effective_positions_bounds():
    n = 10
    equal = pd.Series(1.0 / n, index=range(n))
    assert dg.effective_positions(equal) == pytest.approx(n)

    concentrated = pd.Series([1.0] + [0.0] * (n - 1), index=range(n))
    assert dg.effective_positions(concentrated) == pytest.approx(1.0)


def test_hrp_is_invariant_to_reversing_the_order():
    """A real symmetry of recursive bisection: reversing the list mirrors every split, so
    each asset lands in the mirrored position and keeps its weight. Worth pinning so a
    future refactor cannot quietly break it."""
    rets = factor_returns(8, 500, n_blocks=2, block_loading=1.0, seed=5)
    cov = covariance(rets)
    forward = dg.hrp_weights(cov, list(cov.columns))
    backward = dg.hrp_weights(cov, list(reversed(list(cov.columns))))
    np.testing.assert_allclose(
        forward.to_numpy(), backward.reindex(forward.index).to_numpy(), atol=1e-12
    )


def test_hrp_weights_depend_on_where_the_order_splits():
    """Bisection splits the ordered list by *position*, so an ordering driven by
    noise-level differences hands the split point over to noise. Rotating the order
    moves assets across the midpoint and changes the answer."""
    rets = factor_returns(8, 500, n_blocks=2, block_loading=1.0, seed=5)
    cov = covariance(rets)
    order = list(cov.columns)
    rotated = order[1:] + order[:1]

    base = dg.hrp_weights(cov, order)
    moved = dg.hrp_weights(cov, rotated)
    assert not np.allclose(base.to_numpy(), moved.reindex(base.index).to_numpy())


def test_allocation_sensitivity_runs_over_a_real_universe(make_series, tmp_path, monkeypatch):
    from radar.data.universe import get_universe

    for i, ticker in enumerate(get_universe("sector_etfs").tickers):
        make_series(ticker, start="2020-01-01", periods=300, seed=i)

    result = dg.allocation_sensitivity(
        "sector_etfs", start="2020-01-01", window=100, step=50
    )
    assert result.n_assets == 11
    assert result.rebalances >= 3
    assert result.frame["max_gap_hrp_ivp"].notna().all()
    assert np.isnan(result.frame["turnover_hrp"].iloc[0])
    assert result.frame["turnover_hrp"].iloc[1:].notna().all()
    assert "turnover" in result.summary()
