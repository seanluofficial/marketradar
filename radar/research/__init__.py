"""Hypothesis testing.

This package tests whether signals exist. It does not assert that any do, and a positive
result here is evidence about a hypothesis rather than a reason to deploy capital.
Null results are kept in the repository deliberately -- they are the output.

See research/HYPOTHESIS.md for the pre-registration governing experiment 011.
"""

from radar.research.backtest import (
    BacktestResult,
    breakeven_cost_bps,
    buy_and_hold,
    run_backtest,
    single_asset,
    split_partitions,
)
from radar.research.momentum import (
    LOOKBACKS,
    time_series_momentum_weights,
    trailing_return,
)

__all__ = [
    "LOOKBACKS",
    "BacktestResult",
    "breakeven_cost_bps",
    "buy_and_hold",
    "run_backtest",
    "single_asset",
    "split_partitions",
    "time_series_momentum_weights",
    "trailing_return",
]
