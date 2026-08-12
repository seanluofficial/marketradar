from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.research import backtest as bt
from radar.research import momentum as mom


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    return pd.DataFrame(
        rng.normal(0.0003, 0.011, size=(1500, 4)),
        index=idx, columns=list("ABCD"),
    )


# ---------------------------------------------------------------------------
# Lookahead. If any of these fail, every result the harness produces is worthless.
# ---------------------------------------------------------------------------


def test_weights_set_today_do_not_earn_todays_return(panel):
    """The one-day gap: a decision made at the close of t earns from t+1."""
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    date = panel.index[100]
    weights.loc[date:, "A"] = 1.0

    result = bt.run_backtest(panel, weights, cost_bps=0.0)
    assert result.gross_returns.loc[date] == pytest.approx(0.0)
    assert result.gross_returns.loc[panel.index[101]] == pytest.approx(panel.loc[panel.index[101], "A"])


def test_momentum_signal_ignores_the_future(panel):
    """Truncating everything after date t must not change the weights at date t."""
    cut = panel.index[1000]
    full = mom.time_series_momentum_weights(panel, lookback=252)
    truncated = mom.time_series_momentum_weights(panel.loc[:cut], lookback=252)

    shared = truncated.dropna(how="all").index
    pd.testing.assert_frame_equal(
        full.loc[shared].dropna(how="all"), truncated.loc[shared].dropna(how="all")
    )


def test_trailing_return_uses_only_past_and_present(panel):
    signal = mom.trailing_return(panel, lookback=10)
    date = panel.index[50]
    manual = panel.loc[panel.index[41] : date, "A"].sum()
    assert signal.loc[date, "A"] == pytest.approx(manual)
    assert signal.iloc[:9].isna().all().all()


def test_positive_control_a_clairvoyant_signal_is_detected(panel):
    """The harness must be *able* to find an edge, or a null result proves nothing.

    Feed it tomorrow's return as the signal. This is cheating by construction and must
    produce an enormous Sharpe -- if it did not, the backtester would be broken in a way
    that silently manufactures null results.
    """
    cheat = (panel.shift(-1) > 0).astype(float) / panel.shape[1]
    result = bt.run_backtest(panel, cheat, cost_bps=0.0)
    assert result.stats["sharpe"] > 5.0


def test_negative_control_a_random_signal_is_not_detected(panel):
    rng = np.random.default_rng(7)
    noise = pd.DataFrame(
        rng.integers(0, 2, size=panel.shape) / panel.shape[1],
        index=panel.index, columns=panel.columns,
    )
    assert abs(bt.run_backtest(panel, noise, cost_bps=0.0).stats["sharpe"]) < 1.0


# ---------------------------------------------------------------------------
# Costs and turnover
# ---------------------------------------------------------------------------


def test_costs_reduce_net_returns(panel):
    weights = mom.time_series_momentum_weights(panel, lookback=126)
    free = bt.run_backtest(panel, weights, cost_bps=0.0)
    dear = bt.run_backtest(panel, weights, cost_bps=100.0)
    assert dear.net_returns.sum() < free.net_returns.sum()
    assert free.gross_returns.equals(dear.gross_returns)


def test_turnover_is_booked_when_weights_change(panel):
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    switch = panel.index[10]
    weights.loc[switch:, "A"] = 1.0

    result = bt.run_backtest(panel, weights, cost_bps=0.0)
    assert result.turnover.loc[switch] == pytest.approx(1.0)
    assert result.turnover.drop(switch).sum() == pytest.approx(0.0)


def test_a_never_trading_strategy_costs_nothing(panel):
    flat = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    result = bt.run_backtest(panel, flat, cost_bps=500.0)
    assert result.net_returns.abs().sum() == pytest.approx(0.0)


def test_breakeven_cost_is_zero_for_a_losing_strategy(panel):
    losing = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    losing["A"] = -1.0  # short an asset with positive drift
    assert bt.breakeven_cost_bps(panel, losing) == 0.0


def test_breakeven_cost_is_positive_for_a_winning_strategy(panel):
    cheat = (panel.shift(-1) > 0).astype(float) / panel.shape[1]
    assert bt.breakeven_cost_bps(cheat.pipe(lambda _: panel), cheat) > 10.0


# ---------------------------------------------------------------------------
# Statistics and partitions
# ---------------------------------------------------------------------------


def test_stats_match_hand_computation(panel):
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    weights["A"] = 1.0
    result = bt.run_backtest(panel, weights, cost_bps=0.0)

    held = panel["A"].shift(0).iloc[1:]  # position is on from day 2
    expected_sharpe = held.mean() / held.std(ddof=1) * np.sqrt(252)
    assert result.stats["sharpe"] == pytest.approx(expected_sharpe, rel=0.02)
    assert result.stats["max_drawdown"] <= 0.0
    assert result.stats["average_exposure"] == pytest.approx(1.0, abs=0.01)


def test_t_statistic_scales_with_sample_length(panel):
    weights = pd.DataFrame(1.0 / 4, index=panel.index, columns=panel.columns)
    short = bt.run_backtest(panel.iloc[:300], weights.iloc[:300], 0.0).stats
    long = bt.run_backtest(panel, weights, 0.0).stats
    # Same underlying process, longer sample -> larger |t| for the same Sharpe sign.
    assert abs(long["t_statistic"]) > abs(short["t_statistic"])


def test_partitions_are_chronological_and_disjoint(panel):
    explore, holdout = bt.split_partitions(panel, 0.7)
    assert len(explore) + len(holdout) == len(panel)
    assert explore.index[-1] < holdout.index[0]
    assert len(explore) == pytest.approx(0.7 * len(panel), abs=1)


def test_buy_and_hold_stays_fully_invested(panel):
    result = bt.buy_and_hold(panel, cost_bps=0.0)
    assert result.stats["average_exposure"] == pytest.approx(1.0, abs=0.02)


def test_single_asset_benchmark_tracks_that_asset(panel):
    result = bt.single_asset(panel, "B", cost_bps=0.0)
    expected = (1.0 + panel["B"].iloc[1:]).prod()
    assert result.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_unknown_benchmark_ticker_raises(panel):
    with pytest.raises(KeyError):
        bt.single_asset(panel, "ZZZ")


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------


def test_momentum_holds_cash_when_nothing_trends():
    """Capital is divided by total assets, not qualifying ones, so a market with no
    trends leaves the book partly in cash rather than concentrating."""
    idx = pd.bdate_range("2015-01-01", periods=400)
    falling = pd.DataFrame(-0.001, index=idx, columns=list("ABCD"))
    weights = mom.time_series_momentum_weights(falling, lookback=252).dropna(how="all")
    assert (weights.sum(axis=1) == 0.0).all()


def test_momentum_goes_fully_invested_when_everything_trends():
    idx = pd.bdate_range("2015-01-01", periods=400)
    rising = pd.DataFrame(0.001, index=idx, columns=list("ABCD"))
    weights = mom.time_series_momentum_weights(rising, lookback=252).dropna(how="all")
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_momentum_rebalances_on_the_declared_cadence(panel):
    weights = mom.time_series_momentum_weights(panel, lookback=252, rebalance_days=21)
    dates = weights.dropna(how="all").index
    gaps = np.diff([panel.index.get_loc(d) for d in dates])
    assert set(gaps) == {21}


def test_concentrated_variant_stays_fully_invested_when_anything_trends():
    idx = pd.bdate_range("2015-01-01", periods=400)
    mixed = pd.DataFrame(-0.001, index=idx, columns=list("ABCD"))
    mixed["A"] = 0.001
    weights = mom.concentrated_momentum_weights(mixed, lookback=252).dropna(how="all")
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.allclose(weights["A"], 1.0)
