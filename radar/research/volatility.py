"""Risk-based strategies: volatility management and the low-volatility anomaly.

Neither claims returns are predictable. Both rest on volatility being strongly
autocorrelated in a way returns are not -- you cannot forecast tomorrow's return from
today's, but a turbulent week is genuinely informative about next week's turbulence.

That makes these different in kind from experiments 001-011, all of which were
return-prediction claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radar.research.backtest import REBALANCE_DAYS, infer_periods_per_year

#: Declared per universe in HYPOTHESIS_012.md, not fitted.
TARGET_VOL = {"core_equity": 0.15, "cross_asset": 0.10, "crypto_core": 0.50,
              "crypto_majors": 0.50, "sector_etfs": 0.15}

#: Moreira-Muir requires levering up in calm periods; forbidding it outright would test a
#: weaker claim. Financing cost is not modelled, which is optimistic.
MAX_LEVERAGE = 1.5

VOL_LOOKBACKS = {"21d": 21, "63d": 63}
FORMATION_LOOKBACKS = {"63d": 63, "252d": 252}


def realised_volatility(returns: pd.Series, lookback: int, periods_per_year: float) -> pd.Series:
    """Trailing annualised volatility, using only observations at or before each date."""
    return returns.rolling(lookback, min_periods=lookback).std(ddof=1) * np.sqrt(periods_per_year)


def volatility_managed_weights(
    returns: pd.DataFrame,
    target_vol: float,
    vol_lookback: int = 63,
    max_leverage: float = MAX_LEVERAGE,
    rebalance_days: int = REBALANCE_DAYS,
) -> pd.DataFrame:
    """Equal-weight base portfolio, scaled inversely to its own trailing volatility.

    The scale is computed from the *base portfolio's* realised volatility rather than each
    asset's, because the claim concerns portfolio risk, and per-asset scaling would
    silently reintroduce the low-volatility tilt that H1b tests separately.
    """
    ppy = infer_periods_per_year(returns.index)
    n_assets = returns.shape[1]
    base = returns.mean(axis=1)  # equal-weight portfolio return

    vol = realised_volatility(base, vol_lookback, ppy)
    scale = (target_vol / vol).clip(upper=max_leverage)

    weights = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    eligible = returns.index[vol_lookback - 1 :]
    for date in eligible[::rebalance_days]:
        factor = scale.loc[date]
        if not np.isfinite(factor):
            continue
        weights.loc[date] = float(factor) / n_assets
    return weights


def low_volatility_weights(
    returns: pd.DataFrame,
    formation: int = 252,
    quintile: float = 0.2,
    long_short: bool = False,
    rebalance_days: int = REBALANCE_DAYS,
) -> pd.DataFrame:
    """Long the lowest-volatility quintile; optionally short the highest.

    The long-only leg is the honest retail version. The long/short variant does not model
    borrow cost, and high-volatility names are exactly the ones that are expensive or
    impossible to borrow, so it should be read as an upper bound rather than a result.
    """
    ppy = infer_periods_per_year(returns.index)
    vol = returns.rolling(formation, min_periods=formation).std(ddof=1) * np.sqrt(ppy)

    weights = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    n_leg = max(1, int(round(returns.shape[1] * quintile)))

    eligible = returns.index[formation - 1 :]
    for date in eligible[::rebalance_days]:
        row = vol.loc[date].dropna()
        if len(row) < 2 * n_leg:
            continue
        ranked = row.sort_values()
        allocation = pd.Series(0.0, index=returns.columns)
        allocation[ranked.index[:n_leg]] = 1.0 / n_leg
        if long_short:
            allocation[ranked.index[-n_leg:]] = -1.0 / n_leg
        weights.loc[date] = allocation
    return weights
