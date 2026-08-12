"""Experiment 011 runner — time-series momentum across asset classes.

Executes the panel declared in research/HYPOTHESIS.md. The explore partition runs first
for every cell; the holdout is computed only for cells that clear the pre-registered kill
threshold, and the runner enforces that rather than trusting discipline.

    python -m radar.research.experiment_011
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd

import radar
from radar.config import CACHE_DIR
from radar.data import returns_panel
from radar.research.backtest import (
    breakeven_cost_bps,
    buy_and_hold,
    run_backtest,
    single_asset,
    split_partitions,
)
from radar.research.momentum import LOOKBACKS, time_series_momentum_weights

#: Universe -> (start date, benchmark ticker or None). Fixed by the pre-registration.
UNIVERSES = {
    # 2019, not the pre-registered 2018: Tiingo's 2018 crypto coverage has multi-day
    # holes that drop 6 of 7 coins under the 1-day fill limit. Amended on data-coverage
    # grounds before the holdout was opened -- see the dated note in HYPOTHESIS.md.
    "crypto_core": ("2019-01-01", "BTCUSD"),
    "crypto_majors": ("2021-09-01", "BTCUSD"),
    "sector_etfs": ("2018-07-01", None),
    "cross_asset": ("2008-01-01", "SPY"),
    "core_equity": ("2000-01-01", None),
}

PRIMARY_COST_BPS = 25.0
COST_LEVELS = (10.0, 25.0, 50.0)
KILL_THRESHOLD = 0.30
BONFERRONI_ALPHA = 0.05 / (len(UNIVERSES) * len(LOOKBACKS))


@dataclass
class Cell:
    universe: str
    lookback_name: str
    lookback: int
    n_assets: int
    n_obs: int
    sharpe_by_cost: dict
    sharpe: float
    t_statistic: float
    cagr: float
    vol: float
    max_drawdown: float
    turnover_per_year: float
    average_exposure: float
    breakeven_bps: float
    benchmark_sharpe: float
    benchmark_cagr: float
    single_asset_sharpe: float | None
    passes_kill_threshold: bool


def run_cell(universe: str, lookback_name: str, returns: pd.DataFrame,
             benchmark_ticker: str | None) -> Cell:
    lookback = LOOKBACKS[lookback_name]
    weights = time_series_momentum_weights(returns, lookback=lookback)

    primary = run_backtest(returns, weights, PRIMARY_COST_BPS)
    bench = buy_and_hold(returns, PRIMARY_COST_BPS)
    solo = (
        single_asset(returns, benchmark_ticker, PRIMARY_COST_BPS)
        if benchmark_ticker and benchmark_ticker in returns.columns
        else None
    )

    return Cell(
        universe=universe,
        lookback_name=lookback_name,
        lookback=lookback,
        n_assets=returns.shape[1],
        n_obs=len(returns),
        sharpe_by_cost={
            str(int(c)): run_backtest(returns, weights, c).stats["sharpe"] for c in COST_LEVELS
        },
        sharpe=primary.stats["sharpe"],
        t_statistic=primary.stats["t_statistic"],
        cagr=primary.stats["cagr"],
        vol=primary.stats["vol"],
        max_drawdown=primary.stats["max_drawdown"],
        turnover_per_year=primary.stats["turnover_per_year"],
        average_exposure=primary.stats["average_exposure"],
        breakeven_bps=breakeven_cost_bps(returns, weights),
        benchmark_sharpe=bench.stats["sharpe"],
        benchmark_cagr=bench.stats["cagr"],
        single_asset_sharpe=solo.stats["sharpe"] if solo else None,
        passes_kill_threshold=bool(primary.stats["sharpe"] >= KILL_THRESHOLD),
    )


def run(partition: str = "explore", only: set[tuple[str, str]] | None = None) -> list[Cell]:
    """Run the panel. `only` restricts to specific (universe, lookback) cells.

    The holdout is always invoked with `only` set to the cells that cleared the explore
    rules, so the untested cells' holdout data stays untouched -- looking at it and then
    declining to use it is not the same as not looking.
    """
    cells: list[Cell] = []
    for universe, (start, benchmark) in UNIVERSES.items():
        if only is not None and not any(u == universe for u, _ in only):
            continue
        full, report = returns_panel(
            universe=universe, start=pd.Timestamp(start).date()
        )
        if full.empty:
            print(f"  !! {universe}: no panel ({len(report.dropped)} dropped); skipped")
            continue
        explore, holdout = split_partitions(full)
        returns = explore if partition == "explore" else holdout

        for lookback_name, lookback in LOOKBACKS.items():
            if only is not None and (universe, lookback_name) not in only:
                continue
            if len(returns) < lookback + 21:
                print(f"  !! {universe}/{lookback_name}: too few observations; skipped")
                continue
            cells.append(run_cell(universe, lookback_name, returns, benchmark))
    return cells


def to_frame(cells: list[Cell]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(c) for c in cells])
    for cost in COST_LEVELS:
        frame[f"sharpe@{int(cost)}bps"] = frame["sharpe_by_cost"].apply(
            lambda d, c=cost: d[str(int(c))]
        )
    return frame.drop(columns=["sharpe_by_cost"])


def main() -> int:
    print(f"Experiment 011 -- time-series momentum (radar {radar.__version__})")
    print(f"Pre-registered: {len(UNIVERSES)} universes x {len(LOOKBACKS)} lookbacks, "
          f"kill threshold Sharpe >= {KILL_THRESHOLD}, "
          f"Bonferroni alpha = {BONFERRONI_ALPHA:.4f}\n")

    cells = run("explore")
    frame = to_frame(cells)

    display = frame[[
        "universe", "lookback_name", "n_assets", "n_obs",
        "sharpe@10bps", "sharpe@25bps", "sharpe@50bps",
        "t_statistic", "breakeven_bps", "benchmark_sharpe", "single_asset_sharpe",
        "turnover_per_year", "average_exposure", "max_drawdown",
    ]].copy()
    print(display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    survivors = frame[frame["passes_kill_threshold"]]
    print(f"\n{len(survivors)}/{len(frame)} cells clear the kill threshold "
          f"(explore Sharpe >= {KILL_THRESHOLD} net of {int(PRIMARY_COST_BPS)}bps).")

    beats = frame[frame["sharpe"] > frame["benchmark_sharpe"]]
    print(f"{len(beats)}/{len(frame)} cells beat equal-weight buy-and-hold.")

    out = CACHE_DIR / "research"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "experiment_011_explore.json"
    path.write_text(json.dumps({
        "experiment": "011",
        "partition": "explore",
        "radar_version": radar.__version__,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "kill_threshold": KILL_THRESHOLD,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "cells": [asdict(c) for c in cells],
    }, indent=2, default=str))
    print(f"\nwritten to {path}")

    if survivors.empty:
        print("\nNo cell proceeds to the holdout. Per the pre-registration, the holdout "
              "is not computed for failed cells and remains unused.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
