from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.metrics import absorption as ab
from tests.test_correlation import factor_returns


def equicorrelated(n: int, rho: float) -> np.ndarray:
    m = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(m, 1.0)
    return m


def test_identity_has_no_absorption_beyond_its_components():
    corr = np.eye(50)
    assert ab.pc1_share(corr) == pytest.approx(1 / 50)
    # 10 of 50 unit eigenvalues.
    assert ab.absorption_ratio(corr) == pytest.approx(10 / 50)
    assert ab.effective_dimension(corr) == pytest.approx(50.0)


def test_perfectly_coupled_market_absorbs_everything():
    corr = equicorrelated(40, 0.999)
    assert ab.pc1_share(corr) > 0.99
    assert ab.absorption_ratio(corr) > 0.99
    assert ab.effective_dimension(corr) == pytest.approx(1.0, abs=0.05)


def test_pc1_share_tracks_mean_correlation_for_an_equicorrelated_matrix():
    """The caveat, pinned as a test: lambda_1 = 1 + (N-1)*rho exactly, so the
    systemic-risk index is close to a monotone transform of average correlation."""
    n = 80
    for rho in (0.1, 0.3, 0.6, 0.9):
        corr = equicorrelated(n, rho)
        expected = (1 + (n - 1) * rho) / n
        assert ab.pc1_share(corr) == pytest.approx(expected, abs=1e-9)
        assert ab.mean_correlation(corr) == pytest.approx(rho, abs=1e-12)
        assert abs(ab.pc1_share(corr) - rho) < 1 / n  # converges to rho as N grows


def test_absorption_ratio_rises_with_coupling():
    values = [ab.absorption_ratio(equicorrelated(60, r)) for r in (0.0, 0.2, 0.5, 0.8)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_effective_dimension_falls_with_coupling():
    values = [ab.effective_dimension(equicorrelated(60, r)) for r in (0.0, 0.2, 0.5, 0.8)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_eigen_spectrum_is_descending_and_sums_to_n():
    corr = factor_returns(30, 500, n_blocks=3, market_loading=0.7, block_loading=0.4).corr()
    spectrum = ab.eigen_spectrum(corr)
    assert len(spectrum) == 30
    assert list(spectrum) == sorted(spectrum, reverse=True)
    assert spectrum.sum() == pytest.approx(30.0)


def test_mean_correlation_matches_manual_offdiagonal_mean():
    corr = factor_returns(12, 400, n_blocks=2, market_loading=0.6).corr()
    values = corr.to_numpy()
    manual = values[np.triu_indices(12, k=1)].mean()
    assert ab.mean_correlation(corr) == pytest.approx(manual)


def test_component_count_follows_the_fifth_rule():
    assert ab.n_components_for(80) == 16
    assert ab.n_components_for(11) == 2
    assert ab.n_components_for(1) == 1


def test_explicit_component_count_is_respected_and_clamped():
    corr = equicorrelated(20, 0.4)
    assert ab.absorption_ratio(corr, n_components=1) == pytest.approx(ab.pc1_share(corr))
    assert ab.absorption_ratio(corr, n_components=999) == pytest.approx(1.0)


def test_non_square_input_is_rejected():
    with pytest.raises(ValueError, match="square"):
        ab.absorption_ratio(np.zeros((3, 5)))


def test_mean_correlation_undefined_for_a_single_asset():
    assert np.isnan(ab.mean_correlation(np.array([[1.0]])))
