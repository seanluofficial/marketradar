"""Experiment 012 runner — volatility management and the low-volatility anomaly.

Executes the panel in research/HYPOTHESIS_012.md. Explore first; the holdout runs only
for cells clearing the kill threshold and the benchmark rule.

    python -m radar.research.experiment_012
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
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
from radar.research.volatility import (
    FORMATION_LOOKBACKS,
    MAX_LEVERAGE,
    TARGET_VOL,
    VOL_LOOKBACKS,
    low_volatility_weights,
    volatility_managed_weights,
)

VOL_MANAGED_UNIVERSES = {
    "core_equity": ("2000-01-01", None),
    "cross_asset": ("2008-01-01", "SPY"),
    "crypto_core": ("2019-01-01", "BTCUSD"),
}
LOW_VOL_UNIVERSE = ("core_equity", "2000-01-01")

PRIMARY_COST_BPS = 25.0
COST_LEVELS = (10.0, 25.0, 50.0)
KILL_THRESHOLD = 0.30

N_CELLS = len(VOL_MANAGED_UNIVERSES) * len(VOL_LOOKBACKS) + len(FORMATION_LOOKBACKS) * 2
WITHIN_ALPHA = 0.05 / N_CELLS
PROGRAMME_CELLS = 15 + N_CELLS
PROGRAMME_ALPHA = 0.05 / PROGRAMME_CELLS
PROGRAMME_CRITICAL_T = 3.09


@dataclass
class Cell:
    hypothesis: str
    universe: str
    variant: str
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
    single_asset_sharpe: float | None
    beats_benchmark: bool
    passes_kill_threshold: bool


def _cell(hypothesis, universe, variant, returns, weights, benchmark_ticker) -> Cell:
    primary = run_backtest(returns, weights, PRIMARY_COST_BPS)
    bench = buy_and_hold(returns, PRIMARY_COST_BPS)
    solo = (
        single_asset(returns, benchmark_ticker, PRIMARY_COST_BPS)
        if benchmark_ticker and benchmark_ticker in returns.columns else None
    )
    hurdle = max(bench.stats["sharpe"], solo.stats["sharpe"] if solo else -np.inf)

    return Cell(
        hypothesis=hypothesis,
        universe=universe,
        variant=variant,
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
        single_asset_sharpe=solo.stats["sharpe"] if solo else None,
        beats_benchmark=bool(primary.stats["sharpe"] > hurdle),
        passes_kill_threshold=bool(primary.stats["sharpe"] >= KILL_THRESHOLD),
    )


def run(partition: str = "explore", only: set[tuple[str, str]] | None = None) -> list[Cell]:
    cells: list[Cell] = []

    for universe, (start, benchmark) in VOL_MANAGED_UNIVERSES.items():
        full, _ = returns_panel(universe=universe, start=pd.Timestamp(start).date())
        if full.empty:
            continue
        explore, holdout = split_partitions(full)
        returns = explore if partition == "explore" else holdout

        for name, lookback in VOL_LOOKBACKS.items():
            variant = f"volmanaged_{name}"
            if only is not None and (universe, variant) not in only:
                continue
            weights = volatility_managed_weights(
                returns, target_vol=TARGET_VOL[universe], vol_lookback=lookback,
                max_leverage=MAX_LEVERAGE,
            )
            cells.append(_cell("H1a", universe, variant, returns, weights, benchmark))

    universe, start = LOW_VOL_UNIVERSE
    full, _ = returns_panel(universe=universe, start=pd.Timestamp(start).date())
    explore, holdout = split_partitions(full)
    returns = explore if partition == "explore" else holdout
    for name, formation in FORMATION_LOOKBACKS.items():
        for long_short in (False, True):
            variant = f"lowvol_{name}_{'ls' if long_short else 'long'}"
            if only is not None and (universe, variant) not in only:
                continue
            weights = low_volatility_weights(returns, formation=formation, long_short=long_short)
            cells.append(_cell("H1b", universe, variant, returns, weights, None))

    return cells


def to_frame(cells: list[Cell]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(c) for c in cells])
    for cost in COST_LEVELS:
        frame[f"sharpe@{int(cost)}"] = frame["sharpe_by_cost"].apply(
            lambda d, c=cost: d[str(int(c))]
        )
    return frame.drop(columns=["sharpe_by_cost"])


def main() -> int:
    print(f"Experiment 012 (radar {radar.__version__})")
    print(f"{N_CELLS} cells. Within-experiment alpha {WITHIN_ALPHA:.4f}; "
          f"programme-wide ({PROGRAMME_CELLS} cells) alpha {PROGRAMME_ALPHA:.4f} "
          f"-> |t| > {PROGRAMME_CRITICAL_T}\n")

    cells = run("explore")
    frame = to_frame(cells)
    print(frame[[
        "hypothesis", "universe", "variant", "n_obs",
        "sharpe@10", "sharpe@25", "sharpe@50", "t_statistic",
        "benchmark_sharpe", "single_asset_sharpe", "turnover_per_year",
        "average_exposure", "max_drawdown",
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    survivors = frame[frame["passes_kill_threshold"] & frame["beats_benchmark"]]
    print(f"\n{int(frame['passes_kill_threshold'].sum())}/{len(frame)} clear the kill "
          f"threshold; {len(survivors)}/{len(frame)} also beat their benchmark.")
    if len(survivors):
        print("Cells reaching the holdout:")
        for _, r in survivors.iterrows():
            print(f"  {r['universe']}/{r['variant']}  Sharpe {r['sharpe']:.3f} "
                  f"vs {r['benchmark_sharpe']:.3f}")
    else:
        print("No cell reaches the holdout; it remains unused for this experiment.")

    out = CACHE_DIR / "research"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "experiment_012_explore.json"
    path.write_text(json.dumps({
        "experiment": "012", "partition": "explore",
        "radar_version": radar.__version__,
        "programme_alpha": PROGRAMME_ALPHA,
        "cells": [asdict(c) for c in cells],
    }, indent=2, default=str))
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
