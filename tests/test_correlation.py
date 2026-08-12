from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.structure import correlation as corr_mod
from tests.helpers import factor_returns


@pytest.mark.parametrize("estimator", corr_mod.ESTIMATORS)
def test_every_estimator_returns_a_valid_correlation_matrix(estimator):
    rets = factor_returns(30, 400, n_blocks=3, market_loading=0.6, block_loading=0.5)
    result = corr_mod.estimate_correlation(rets, estimator)

    m = result.values
    assert m.shape == (30, 30)
    assert np.allclose(np.diag(m), 1.0)
    assert np.allclose(m, m.T)
    assert (m >= -1.0).all() and (m <= 1.0).all()
    assert result.tickers == list(rets.columns)
    assert result.n_obs == 400
    assert result.q == pytest.approx(30 / 400)


@pytest.mark.parametrize("estimator", corr_mod.ESTIMATORS)
def test_estimators_are_positive_semidefinite(estimator):
    rets = factor_returns(30, 400, n_blocks=3, market_loading=0.6, block_loading=0.5)
    result = corr_mod.estimate_correlation(rets, estimator)
    assert result.diagnostics["min_eigenvalue"] > -1e-8


def test_sample_matches_pandas_corr():
    rets = factor_returns(8, 200, market_loading=0.5)
    result = corr_mod.sample_correlation(rets)
    pd.testing.assert_frame_equal(result.matrix, rets.corr(), atol=1e-12)


def test_estimators_agree_when_data_is_abundant():
    """At q -> 0 there is nothing to clean, so all three must converge."""
    rets = factor_returns(10, 20000, n_blocks=2, market_loading=0.6, block_loading=0.4, seed=3)
    sample = corr_mod.sample_correlation(rets).values
    lw = corr_mod.ledoit_wolf_correlation(rets).values
    rmt = corr_mod.rmt_clipped_correlation(rets).values
    assert np.abs(sample - lw).max() < 0.05
    assert np.abs(sample - rmt).max() < 0.05


# ---------------------------------------------------------------------------
# The regime that motivates the whole module: q large.
# ---------------------------------------------------------------------------


def test_sample_is_singular_when_assets_exceed_observations():
    rets = factor_returns(60, 40, market_loading=0.5)
    result = corr_mod.sample_correlation(rets)
    assert result.q > 1
    assert result.diagnostics["min_eigenvalue"] < 1e-8
    assert result.diagnostics["condition_number"] > 1e6


@pytest.mark.parametrize("estimator", ["ledoit_wolf", "rmt_clipped"])
def test_cleaned_estimators_stay_invertible_where_sample_fails(estimator):
    rets = factor_returns(60, 40, market_loading=0.5)
    result = corr_mod.estimate_correlation(rets, estimator)
    assert result.diagnostics["min_eigenvalue"] > 1e-6
    assert np.isfinite(result.diagnostics["condition_number"])


def test_shrinkage_intensity_is_reported_and_bounded():
    rets = factor_returns(50, 120, market_loading=0.5)
    result = corr_mod.ledoit_wolf_correlation(rets)
    assert 0.0 < result.diagnostics["shrinkage"] <= 1.0


def test_shrinkage_rises_as_the_sample_shrinks():
    wide = corr_mod.ledoit_wolf_correlation(factor_returns(40, 2000, market_loading=0.5, seed=7))
    thin = corr_mod.ledoit_wolf_correlation(factor_returns(40, 60, market_loading=0.5, seed=7))
    assert thin.diagnostics["shrinkage"] > wide.diagnostics["shrinkage"]


# ---------------------------------------------------------------------------
# RMT clipping
# ---------------------------------------------------------------------------


def test_marchenko_pastur_edge_formula():
    assert corr_mod.marchenko_pastur_edge(0.0) == pytest.approx(1.0)
    assert corr_mod.marchenko_pastur_edge(1.0) == pytest.approx(4.0)
    assert corr_mod.marchenko_pastur_edge(0.25) == pytest.approx(2.25)
    assert corr_mod.marchenko_pastur_edge(0.25, 2.0) == pytest.approx(4.5)


def test_pure_noise_is_clipped_toward_the_identity():
    """Independent assets have no real structure; the cleaned matrix should say so."""
    rets = factor_returns(50, 250, market_loading=0.0, block_loading=0.0, seed=11)
    sample = corr_mod.sample_correlation(rets)
    cleaned = corr_mod.rmt_clipped_correlation(rets)

    off_sample = np.abs(sample.values - np.eye(50)).max()
    off_cleaned = np.abs(cleaned.values - np.eye(50)).max()
    assert off_sample > 0.15, "sanity: sampling noise should create spurious correlation"
    assert off_cleaned < off_sample / 3
    assert cleaned.diagnostics["n_factors"] <= 1


def test_real_factors_survive_clipping():
    rets = factor_returns(60, 500, n_blocks=4, market_loading=0.8, block_loading=0.6, seed=5)
    cleaned = corr_mod.rmt_clipped_correlation(rets)
    # One market mode plus four block modes should escape the noise band.
    assert 3 <= cleaned.diagnostics["n_factors"] <= 8
    assert cleaned.diagnostics["largest_eigenvalue"] > cleaned.diagnostics["lambda_plus"]


def test_fitted_sigma_is_below_one_when_a_market_mode_dominates():
    """The point of fitting sigma^2: the market mode removes variance from the bulk, so
    assuming sigma^2 = 1 sets the noise band too high and discards real sector factors."""
    rets = factor_returns(60, 500, n_blocks=4, market_loading=1.2, block_loading=0.6, seed=5)
    fitted = corr_mod.rmt_clipped_correlation(rets, fit_sigma=True)
    naive = corr_mod.rmt_clipped_correlation(rets, fit_sigma=False)

    assert fitted.diagnostics["sigma_squared"] < 1.0
    assert fitted.diagnostics["lambda_plus"] < naive.diagnostics["lambda_plus"]
    assert fitted.diagnostics["n_factors"] >= naive.diagnostics["n_factors"]


def test_clipping_reduces_the_condition_number():
    rets = factor_returns(80, 250, n_blocks=4, market_loading=0.7, block_loading=0.5, seed=9)
    sample = corr_mod.sample_correlation(rets)
    cleaned = corr_mod.rmt_clipped_correlation(rets)
    assert cleaned.diagnostics["condition_number"] < sample.diagnostics["condition_number"]


# ---------------------------------------------------------------------------
# Input validation -- these guard the correctness of everything downstream.
# ---------------------------------------------------------------------------


def test_nans_are_rejected_rather_than_dropped_pairwise():
    rets = factor_returns(5, 100)
    rets.iloc[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        corr_mod.sample_correlation(rets)


def test_constant_column_is_rejected():
    """pandas reports std ~ 1e-18 rather than exactly 0 for a constant column, so the
    check is scaled to the column's magnitude. Otherwise corrcoef emits a NaN row that
    reaches the distance matrix and the tree unflagged."""
    rets = factor_returns(5, 100)
    rets.iloc[:, 1] = 0.01
    assert rets.iloc[:, 1].std(ddof=1) != 0.0  # pin the behaviour this guards against
    with pytest.raises(ValueError, match="Zero-variance"):
        corr_mod.sample_correlation(rets)


def test_barely_varying_column_is_accepted():
    """The tolerance must not swallow a real but quiet series -- a name that traded
    thinly is noisy, not undefined, and dropping it silently would be worse."""
    rets = factor_returns(5, 100)
    rets.iloc[:, 1] = np.linspace(0.0, 1e-6, 100)
    result = corr_mod.sample_correlation(rets)
    assert not result.matrix.isna().any().any()


def test_empty_panel_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        corr_mod.sample_correlation(pd.DataFrame())


def test_unknown_estimator_lists_the_valid_ones():
    rets = factor_returns(5, 100)
    with pytest.raises(ValueError, match="ledoit_wolf"):
        corr_mod.estimate_correlation(rets, "magic")


def test_cov_to_corr_rejects_degenerate_covariance():
    with pytest.raises(ValueError, match="non-positive"):
        corr_mod.cov_to_corr(np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_summary_mentions_the_estimator_and_q():
    rets = factor_returns(20, 300, market_loading=0.5)
    text = corr_mod.rmt_clipped_correlation(rets).summary()
    assert "rmt_clipped" in text and "q=" in text and "lambda+" in text
