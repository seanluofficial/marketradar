"""A small, deliberately pessimistic backtester.

Every design choice here resolves ambiguity *against* the strategy, because the failure
mode of backtesting is not arithmetic error -- it is a thousand small optimistic defaults
compounding into a result you believe.

  * Signals are computed from data up to and including the rebalance date, and the
    resulting positions earn the *next* day's return onward. The one-day gap is enforced
    by a shift, and tested with a deliberately clairvoyant signal that must not profit.
  * Costs are charged on turnover at the moment of trading, per side.
  * Cash earns zero. In a positive-rate environment that understates a long-flat
    strategy's return, which is the safe direction to be wrong in.
  * Nothing is fitted. There are no parameters estimated from the data being tested.

This module tests whether a signal exists. It does not generate trading advice, and a
result here is evidence about a hypothesis, not a reason to deploy capital.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252

#: Rebalance cadence in trading days. Monthly: frequent enough for a momentum signal to
#: act, infrequent enough that costs do not automatically dominate.
REBALANCE_DAYS = 21


@dataclass
class BacktestResult:
    equity: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    cost_bps: float
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        s = self.stats
        return (
            f"Sharpe {s['sharpe']:+.3f}  CAGR {s['cagr']:+.2%}  vol {s['vol']:.2%}  "
            f"maxDD {s['max_drawdown']:.2%}  turnover {s['turnover_per_year']:.1f}x/yr  "
            f"exposure {s['average_exposure']:.0%}"
        )


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Observations per calendar year, measured from the index.

    Hardcoding 252 is correct for equities and wrong for crypto, which trades every
    calendar day -- and annualising crypto at 252 understates its Sharpe by sqrt(365/252),
    about 20%. Measuring it removes a whole class of silent, asset-class-specific error.
    """
    if len(index) < 2:
        return float(TRADING_DAYS)
    span_years = (index[-1] - index[0]).days / 365.25
    if span_years <= 0:
        return float(TRADING_DAYS)
    return float(len(index) / span_years)


def performance_stats(
    net: pd.Series,
    turnover: pd.Series,
    weights: pd.DataFrame,
    periods_per_year: float | None = None,
) -> dict:
    """Standard risk statistics. Sharpe uses a zero risk-free rate, stated not assumed."""
    ppy = periods_per_year if periods_per_year is not None else infer_periods_per_year(net.index)
    if net.empty or net.std(ddof=1) == 0:
        return {
            "sharpe": np.nan, "cagr": np.nan, "vol": np.nan, "max_drawdown": np.nan,
            "turnover_per_year": np.nan, "average_exposure": np.nan, "n_obs": len(net),
            "t_statistic": np.nan, "periods_per_year": ppy,
        }

    equity = (1.0 + net).cumprod()
    years = len(net) / ppy
    vol = float(net.std(ddof=1) * np.sqrt(ppy))
    sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(ppy))
    drawdown = equity / equity.cummax() - 1.0

    return {
        "sharpe": sharpe,
        # t = Sharpe * sqrt(years): the significance of the mean return, which is what the
        # pre-registered threshold is actually about.
        "t_statistic": float(sharpe * np.sqrt(years)),
        "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan,
        "vol": vol,
        "max_drawdown": float(drawdown.min()),
        "turnover_per_year": float(turnover.sum() / years) if years > 0 else np.nan,
        "average_exposure": float(weights.sum(axis=1).mean()),
        "n_obs": int(len(net)),
        "periods_per_year": ppy,
    }


def run_backtest(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_bps: float = 25.0,
) -> BacktestResult:
    """Apply `target_weights` to `returns`, charging `cost_bps` per side on turnover.

    `target_weights` must be indexed like `returns` and stated as of the close of each
    date. They are shifted forward one day before earning anything, so a weight decided
    using information through date t cannot capture date t's return.
    """
    if returns.empty:
        raise ValueError("Return panel is empty.")
    weights = target_weights.reindex(returns.index).ffill().fillna(0.0)
    if list(weights.columns) != list(returns.columns):
        weights = weights.reindex(columns=returns.columns).fillna(0.0)

    held = weights.shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)

    # Turnover is booked on the day the weights change, which is the day the trade happens.
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    turnover.iloc[0] = float(weights.iloc[0].abs().sum())
    costs = turnover * (cost_bps / 10_000.0)

    net = gross - costs
    equity = (1.0 + net).cumprod()

    return BacktestResult(
        equity=equity,
        gross_returns=gross,
        net_returns=net,
        weights=weights,
        turnover=turnover,
        cost_bps=cost_bps,
        stats=performance_stats(net, turnover, held),
    )


def buy_and_hold(returns: pd.DataFrame, cost_bps: float = 25.0) -> BacktestResult:
    """Equal-weight buy-and-hold benchmark, rebalanced on the same monthly cadence.

    Rebalanced rather than drifting so that the comparison isolates the *signal* rather
    than the rebalancing policy.
    """
    weights = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    rebalances = returns.index[::REBALANCE_DAYS]
    weights.loc[rebalances] = 1.0 / returns.shape[1]
    return run_backtest(returns, weights, cost_bps)


def single_asset(returns: pd.DataFrame, ticker: str, cost_bps: float = 25.0) -> BacktestResult:
    """Hold one asset and nothing else -- the benchmark that actually matters in crypto."""
    if ticker not in returns.columns:
        raise KeyError(f"{ticker!r} not in the panel.")
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    weights[ticker] = 1.0
    return run_backtest(returns, weights, cost_bps)


def breakeven_cost_bps(
    returns: pd.DataFrame, target_weights: pd.DataFrame, hi: float = 500.0
) -> float:
    """Cost level at which net Sharpe reaches zero.

    More informative than a Sharpe at one assumed cost: it says how wrong the cost
    assumption has to be before the conclusion flips.
    """
    if run_backtest(returns, target_weights, 0.0).stats["sharpe"] <= 0:
        return 0.0
    lo, high = 0.0, hi
    for _ in range(40):
        mid = 0.5 * (lo + high)
        if run_backtest(returns, target_weights, mid).stats["sharpe"] > 0:
            lo = mid
        else:
            high = mid
    return float(0.5 * (lo + high))


def split_partitions(
    returns: pd.DataFrame, explore_fraction: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological explore/holdout split.

    By date, never at random: a random split would let the model see the future through
    adjacent observations, and the whole point of the holdout is to simulate not having
    it.
    """
    cut = int(len(returns) * explore_fraction)
    return returns.iloc[:cut], returns.iloc[cut:]
