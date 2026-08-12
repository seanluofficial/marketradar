"""Experiment 012 holdout — run once, on exactly the cells that cleared explore.

    python -m radar.research.holdout_012
"""

from __future__ import annotations

import json
from dataclasses import asdict

import radar
from radar.config import CACHE_DIR
from radar.research.experiment_012 import (
    PROGRAMME_ALPHA,
    PROGRAMME_CRITICAL_T,
    run,
    to_frame,
)

#: Fixed by the explore results so the set cannot drift on a re-run.
SURVIVORS = {
    ("core_equity", "volmanaged_21d"),
    ("core_equity", "volmanaged_63d"),
    ("core_equity", "lowvol_63d_long"),
    ("core_equity", "lowvol_252d_long"),
}


def main() -> int:
    print(f"Experiment 012 holdout (radar {radar.__version__})")
    print(f"Programme-wide bar: p < {PROGRAMME_ALPHA:.4f}  (|t| > {PROGRAMME_CRITICAL_T})")
    print("Note: core_equity's holdout was already used once in experiment 011, so this "
          "evidence is weaker than a pristine partition would give.\n")

    cells = run("holdout", only=SURVIVORS)
    frame = to_frame(cells)
    print(frame[[
        "hypothesis", "variant", "n_obs", "sharpe@10", "sharpe@25", "sharpe@50",
        "t_statistic", "benchmark_sharpe", "turnover_per_year", "max_drawdown",
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print()
    for cell in cells:
        significant = abs(cell.t_statistic) > PROGRAMME_CRITICAL_T
        verdict = "SURVIVES" if (cell.beats_benchmark and significant) else "FAILS"
        reasons = []
        if not cell.beats_benchmark:
            reasons.append(f"below benchmark {cell.benchmark_sharpe:.3f}")
        if not significant:
            reasons.append(f"|t| = {abs(cell.t_statistic):.2f} < {PROGRAMME_CRITICAL_T}")
        print(f"  {cell.variant}: {verdict}  Sharpe {cell.sharpe:+.3f}"
              + (f"  ({'; '.join(reasons)})" if reasons else ""))

    out = CACHE_DIR / "research"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "experiment_012_holdout.json"
    path.write_text(json.dumps({
        "experiment": "012", "partition": "holdout",
        "radar_version": radar.__version__,
        "survivors_tested": sorted(SURVIVORS),
        "programme_alpha": PROGRAMME_ALPHA,
        "cells": [asdict(c) for c in cells],
    }, indent=2, default=str))
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
