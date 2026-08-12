"""Time-series momentum signals.

Time-series momentum asks whether an asset's *own* past return predicts its own future
return. Distinct from the cross-sectional version -- which asks whether relative rank
predicts relative return -- and tested separately for that reason.

Long-flat by construction: hold the asset when its trailing return is positive, hold
nothing when it is not. That halves turnover relative to a long/short book, which matters
because costs are what defeated the cross-sectional test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radar.research.backtest import REBALANCE_DAYS

LOOKBACKS = {"3m": 63, "6m": 126, "12m": 252}


def trailing_return(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Cumulative log return over the trailing `lookback` observations, inclusive of today.

    Log returns are additive, so this is a rolling sum -- and it uses only rows at or
    before each date, which is what makes the signal usable at the rebalance.
    """
    return returns.rolling(lookback, min_periods=lookback).sum()


def time_series_momentum_weights(
    returns: pd.DataFrame,
    lookback: int = 252,
    rebalance_days: int = REBALANCE_DAYS,
) -> pd.DataFrame:
    """Equal-weight across assets whose trailing return is positive; cash otherwise.

    Weights are stated as of the close of each rebalance date and are NaN in between, so
    the backtester's forward-fill holds the position until the next decision. Capital is
    divided by the *total* asset count rather than the number of qualifying assets, so a
    period when few assets trend leaves the book partly in cash instead of concentrating
    into whatever is left.
    """
    signal = trailing_return(returns, lookback)
    weights = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    eligible = returns.index[lookback - 1 :]
    rebalances = eligible[::rebalance_days]
    n_assets = returns.shape[1]

    for date in rebalances:
        row = signal.loc[date]
        if row.isna().all():
            continue
        weights.loc[date] = (row > 0).astype(float) / n_assets

    return weights


def concentrated_momentum_weights(
    returns: pd.DataFrame,
    lookback: int = 252,
    rebalance_days: int = REBALANCE_DAYS,
) -> pd.DataFrame:
    """Variant that splits capital only across qualifying assets, staying fully invested.

    Not part of the pre-registered panel -- kept for the robustness check the pre-registration
    calls for, and reported separately so it cannot be quietly swapped in as the headline
    if it happens to look better.
    """
    signal = trailing_return(returns, lookback)
    weights = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    eligible = returns.index[lookback - 1 :]
    for date in eligible[::rebalance_days]:
        row = signal.loc[date]
        qualifying = (row > 0)
        count = int(qualifying.sum())
        weights.loc[date] = qualifying.astype(float) / count if count else 0.0

    return weights
