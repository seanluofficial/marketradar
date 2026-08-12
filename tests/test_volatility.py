from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.research import backtest as bt
from radar.research import volatility as vol


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-01", periods=1500)
    return pd.DataFrame(
        rng.normal(0.0004, 0.012, size=(1500, 10)),
        index=idx, columns=[f"A{i}" for i in range(10)],
    )


def test_realised_volatility_matches_manual(panel):
    series = panel["A0"]
    computed = vol.realised_volatility(series, 21, 252.0)
    date = panel.index[100]
    manual = series.loc[panel.index[80] : date].std(ddof=1) * np.sqrt(252)
    assert computed.loc[date] == pytest.approx(manual)
    assert computed.iloc[:20].isna().all()


def test_vol_managed_weights_do_not_use_the_future(panel):
    cut = panel.index[900]
    full = vol.volatility_managed_weights(panel, 0.15, 63)
    truncated = vol.volatility_managed_weights(panel.loc[:cut], 0.15, 63)
    shared = truncated.dropna(how="all").index
    pd.testing.assert_frame_equal(
        full.loc[shared].dropna(how="all"), truncated.loc[shared].dropna(how="all")
    )


def test_vol_managed_scales_down_in_turbulence():
    """The mechanism, isolated: a calm regime followed by a violent one must produce
    smaller weights in the violent one."""
    idx = pd.bdate_range("2010-01-01", periods=600)
    rng = np.random.default_rng(2)
    calm = rng.normal(0.0, 0.004, size=(300, 4))
    wild = rng.normal(0.0, 0.040, size=(300, 4))
    panel = pd.DataFrame(np.vstack([calm, wild]), index=idx, columns=list("ABCD"))

    weights = vol.volatility_managed_weights(panel, target_vol=0.15, vol_lookback=63)
    gross = weights.dropna(how="all").sum(axis=1)
    early = gross.loc[: idx[280]].mean()
    late = gross.loc[idx[400] :].mean()
    assert late < early / 2


def test_leverage_cap_is_respected(panel):
    weights = vol.volatility_managed_weights(
        panel, target_vol=10.0, vol_lookback=63, max_leverage=1.5
    )
    gross = weights.dropna(how="all").sum(axis=1)
    assert gross.max() == pytest.approx(1.5)


def test_low_vol_selects_the_quietest_names():
    idx = pd.bdate_range("2010-01-01", periods=400)
    rng = np.random.default_rng(3)
    data = {}
    for i in range(10):
        data[f"A{i}"] = rng.normal(0.0, 0.002 * (i + 1), size=400)
    panel = pd.DataFrame(data, index=idx)

    weights = vol.low_volatility_weights(panel, formation=252).dropna(how="all")
    held = weights.iloc[-1]
    assert held["A0"] > 0 and held["A1"] > 0     # quietest
    assert held["A9"] == 0 and held["A8"] == 0   # loudest
    assert held.sum() == pytest.approx(1.0)


def test_low_vol_long_short_is_dollar_neutral():
    idx = pd.bdate_range("2010-01-01", periods=400)
    rng = np.random.default_rng(4)
    panel = pd.DataFrame(
        {f"A{i}": rng.normal(0.0, 0.002 * (i + 1), size=400) for i in range(10)}, index=idx
    )
    weights = vol.low_volatility_weights(panel, formation=252, long_short=True).dropna(how="all")
    assert weights.iloc[-1].sum() == pytest.approx(0.0)
    assert weights.iloc[-1].abs().sum() == pytest.approx(2.0)


def test_low_vol_weights_do_not_use_the_future(panel):
    cut = panel.index[900]
    full = vol.low_volatility_weights(panel, formation=252)
    truncated = vol.low_volatility_weights(panel.loc[:cut], formation=252)
    shared = truncated.dropna(how="all").index
    pd.testing.assert_frame_equal(
        full.loc[shared].dropna(how="all"), truncated.loc[shared].dropna(how="all")
    )


def test_vol_managed_cuts_risk_when_the_target_is_below_natural_vol(panel):
    """Targeting a level, not minimising: with a target under the portfolio's own vol the
    overlay de-levers. (With a target above it, it levers up to the cap -- which is the
    mechanism working, not a bug.)"""
    natural = bt.buy_and_hold(panel, cost_bps=0.0).stats["vol"]
    managed = vol.volatility_managed_weights(panel, target_vol=natural / 2, vol_lookback=63)
    result = bt.run_backtest(panel, managed, cost_bps=0.0)
    assert result.stats["vol"] < natural


def test_vol_managed_steadies_risk_across_a_regime_change():
    """The actual promise: a steadier risk level through a volatility regime shift."""
    idx = pd.bdate_range("2010-01-01", periods=1200)
    rng = np.random.default_rng(11)
    calm = rng.normal(0.0, 0.005, size=(600, 4))
    wild = rng.normal(0.0, 0.030, size=(600, 4))
    panel = pd.DataFrame(np.vstack([calm, wild]), index=idx, columns=list("ABCD"))

    managed = bt.run_backtest(
        panel, vol.volatility_managed_weights(panel, 0.15, 63), cost_bps=0.0
    )
    baseline = bt.buy_and_hold(panel, cost_bps=0.0)

    def dispersion(returns: pd.Series) -> float:
        return returns.rolling(63).std(ddof=1).std(ddof=1)

    assert dispersion(managed.net_returns) < dispersion(baseline.net_returns)


def test_target_vol_declared_for_every_tested_universe():
    from radar.research.experiment_012 import LOW_VOL_UNIVERSE, VOL_MANAGED_UNIVERSES

    for universe in VOL_MANAGED_UNIVERSES:
        assert universe in vol.TARGET_VOL
    assert LOW_VOL_UNIVERSE[0] in vol.TARGET_VOL
