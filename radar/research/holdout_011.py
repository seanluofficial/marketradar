"""Experiment 011 holdout — run exactly once, on exactly the surviving cells.

Per research/HYPOTHESIS.md rules 3 and 5, a cell reaches the holdout only if it cleared
the explore kill threshold *and* beat its benchmark. Those cells are hardcoded below
rather than recomputed, so that re-running this file cannot quietly widen the set.

    python -m radar.research.holdout_011
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

import radar
from radar.config import CACHE_DIR
from radar.research.experiment_011 import (
    BONFERRONI_ALPHA,
    PRIMARY_COST_BPS,
    to_frame,
    run,
)

#: Fixed by the explore results, recorded here so the holdout set cannot drift.
SURVIVORS = {("core_equity", "6m"), ("core_equity", "12m")}

#: Two-sided normal critical value at the Bonferroni-corrected alpha (0.05 / 15).
BONFERRONI_CRITICAL_T = 2.94


def main() -> int:
    print(f"Experiment 011 holdout (radar {radar.__version__})")
    print(f"Cells reaching the holdout: {sorted(SURVIVORS)}")
    print(f"Corrected significance bar: p < {BONFERRONI_ALPHA:.4f}  (|t| > {BONFERRONI_CRITICAL_T})\n")

    cells = run("holdout", only=SURVIVORS)
    frame = to_frame(cells)

    print(frame[[
        "universe", "lookback_name", "n_obs",
        "sharpe@10bps", "sharpe@25bps", "sharpe@50bps",
        "t_statistic", "breakeven_bps", "benchmark_sharpe",
        "turnover_per_year", "max_drawdown",
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print()
    for cell in cells:
        beats = cell.sharpe > cell.benchmark_sharpe
        significant = abs(cell.t_statistic) > BONFERRONI_CRITICAL_T
        verdict = "SURVIVES" if (beats and significant and cell.sharpe > 0) else "FAILS"
        print(
            f"  {cell.universe}/{cell.lookback_name}: {verdict}  "
            f"Sharpe {cell.sharpe:+.3f} vs benchmark {cell.benchmark_sharpe:+.3f}, "
            f"t = {cell.t_statistic:+.2f}"
        )

    out = CACHE_DIR / "research"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "experiment_011_holdout.json"
    path.write_text(json.dumps({
        "experiment": "011",
        "partition": "holdout",
        "radar_version": radar.__version__,
        "survivors_tested": sorted(SURVIVORS),
        "primary_cost_bps": PRIMARY_COST_BPS,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "cells": [asdict(c) for c in cells],
    }, indent=2, default=str))
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
