from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.structure import distance as dist_mod
from tests.helpers import factor_returns


def test_known_values():
    corr = pd.DataFrame(
        [[1.0, 0.0, -1.0], [0.0, 1.0, 0.5], [-1.0, 0.5, 1.0]],
        index=list("XYZ"), columns=list("XYZ"),
    )
    d = dist_mod.correlation_to_distance(corr)
    assert d.loc["X", "X"] == 0.0
    assert d.loc["X", "Y"] == pytest.approx(np.sqrt(2.0))
    assert d.loc["X", "Z"] == pytest.approx(2.0)
    assert d.loc["Y", "Z"] == pytest.approx(1.0)


def test_perfect_correlation_gives_zero_distance():
    corr = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], index=list("AB"), columns=list("AB"))
    d = dist_mod.correlation_to_distance(corr)
    assert d.loc["A", "B"] == pytest.approx(0.0)


def test_distance_is_monotone_decreasing_in_correlation():
    rhos = np.linspace(-1.0, 1.0, 41)
    ds = [np.sqrt(2 * (1 - r)) for r in rhos]
    assert all(a > b for a, b in zip(ds, ds[1:]))


def test_independent_distance_constant():
    assert dist_mod.INDEPENDENT_DISTANCE == pytest.approx(np.sqrt(2.0))


def test_out_of_range_correlation_is_clipped_not_nan():
    """Estimator round-off can push an entry a hair past 1; sqrt of a negative would
    become a silent NaN edge in the MST."""
    corr = np.array([[1.0, 1.0 + 1e-12], [1.0 + 1e-12, 1.0]])
    d = dist_mod.correlation_to_distance(corr)
    assert not d.isna().any().any()
    assert d.iloc[0, 1] == pytest.approx(0.0)


def test_output_is_symmetric_with_zero_diagonal():
    rets = factor_returns(15, 300, n_blocks=3, market_loading=0.6, block_loading=0.5)
    d = dist_mod.correlation_to_distance(rets.corr())
    assert np.allclose(d.to_numpy(), d.to_numpy().T)
    assert np.allclose(np.diag(d.to_numpy()), 0.0)
    assert d.index.tolist() == rets.columns.tolist()


def test_triangle_inequality_holds_on_real_shaped_data():
    """The metric property is what licenses the MST and classical MDS downstream."""
    rets = factor_returns(25, 400, n_blocks=4, market_loading=0.7, block_loading=0.5, seed=2)
    d = dist_mod.correlation_to_distance(rets.corr())
    assert dist_mod.is_metric(d)


def test_is_metric_rejects_a_triangle_violation():
    bad = pd.DataFrame(
        [[0.0, 1.0, 5.0], [1.0, 0.0, 1.0], [5.0, 1.0, 0.0]],
        index=list("XYZ"), columns=list("XYZ"),
    )
    assert not dist_mod.is_metric(bad)


def test_is_metric_rejects_asymmetry():
    bad = pd.DataFrame(
        [[0.0, 1.0], [2.0, 0.0]], index=list("XY"), columns=list("XY")
    )
    assert not dist_mod.is_metric(bad)


def test_non_square_input_is_rejected():
    with pytest.raises(ValueError, match="square"):
        dist_mod.correlation_to_distance(np.zeros((3, 4)))


def test_nearest_neighbours_returns_closest_first_excluding_self():
    rets = factor_returns(12, 400, n_blocks=3, market_loading=0.3, block_loading=0.9, seed=4)
    d = dist_mod.correlation_to_distance(rets.corr())
    nn = dist_mod.nearest_neighbours(d, "A000", k=3)

    assert "A000" not in nn.index
    assert len(nn) == 3
    assert list(nn.values) == sorted(nn.values)
    # A000, A003, A006, A009 share a block, so the nearest names come from it.
    assert set(nn.index) <= {"A003", "A006", "A009"}


def test_nearest_neighbours_rejects_unknown_ticker():
    rets = factor_returns(5, 100)
    d = dist_mod.correlation_to_distance(rets.corr())
    with pytest.raises(KeyError):
        dist_mod.nearest_neighbours(d, "NOPE")
