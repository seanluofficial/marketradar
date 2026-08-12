"""Is there enough correlation structure here to be worth exploiting?

A universe-level diagnostic, meant to be run *before* committing to a method that assumes
structure exists. Hierarchical clustering will always return a tree, and an MST will
always return a tree; neither tells you whether the tree means anything. This does.

Five measurements, each with an interpretation attached:

  baseline coupling   median pairwise correlation. High and flat means one dominant
                      factor and little for a clustering method to separate.
  index range         the absorption ratio's spread. A regime overlay needs the index to
                      actually move; an index pinned near its ceiling cannot support a
                      threshold rule.
  tree depth          cophenetic correlation (does the tree represent the distances?) and
                      depth ratio (is there nested structure, or one late merge?).
  cluster stability   adjusted Rand index between consecutive windows. Unstable clusters
                      mean any allocation built on them will churn and pay fees.
  label agreement     MST purity against a prior taxonomy, versus a random baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from radar.data import returns_panel
from radar.data.universe import get_universe
from radar.metrics.absorption import absorption_ratio, effective_dimension, mean_correlation
from radar.structure.hierarchy import cluster_from_distance, cluster_stability
from radar.structure.correlation import DEFAULT_ESTIMATOR, estimate_correlation
from radar.structure.distance import correlation_to_distance
from radar.structure.mst import build_mst, edge_survival, group_purity, normalised_tree_length


@dataclass
class Diagnostic:
    universe: str
    n_assets: int
    n_obs: int
    start: date | None
    end: date | None
    window: int
    estimator: str
    windows: pd.DataFrame
    n_clusters: int
    notes: list[str] = field(default_factory=list)

    def _span(self, column: str) -> tuple[float, float, float]:
        s = self.windows[column].dropna()
        return float(s.min()), float(s.median()), float(s.max())

    def summary(self) -> str:
        w = self.windows
        lines = [
            f"=== {self.universe}: {self.n_assets} assets, {self.n_obs} obs "
            f"{self.start} -> {self.end} ===",
            f"    {len(w)} windows of {self.window} obs, estimator={self.estimator}, "
            f"q = {self.n_assets / self.window:.3f}",
            "",
        ]

        def row(label: str, column: str, fmt: str = "{:.3f}") -> str:
            lo, mid, hi = self._span(column)
            return (f"  {label:<24} " + fmt.format(mid) + "   [" + fmt.format(lo)
                    + " - " + fmt.format(hi) + "]")

        lines += [
            "  metric                   median   [min - max]",
            row("mean correlation", "mean_correlation"),
            row("absorption ratio", "absorption_ratio"),
            row("pc1 share", "pc1_share"),
            row("effective dimension", "effective_dimension", "{:.1f}"),
            row("MST tree length", "tree_length"),
            row("cophenetic corr", "cophenetic"),
            row("depth ratio", "depth_ratio"),
            row("cluster stability (ARI)", "cluster_stability"),
            row("MST edge survival", "edge_survival"),
            row("purity lift vs random", "purity_lift", "{:.2f}"),
            "",
            f"  absorption ratio IQR:  {w['absorption_ratio'].quantile(0.75) - w['absorption_ratio'].quantile(0.25):.3f}"
            "   (a regime threshold needs room to move)",
        ]
        if self.notes:
            lines += ["", "  notes:"] + [f"    - {n}" for n in self.notes]
        return "\n".join(lines)


def structure_diagnostic(
    universe: str,
    start: str | None = None,
    end: str | None = None,
    window: int = 252,
    step: int = 5,
    estimator: str = DEFAULT_ESTIMATOR,
    n_clusters: int | None = None,
) -> Diagnostic:
    """Measure whether `universe` has structure worth exploiting."""
    uni = get_universe(universe)
    rets, report = returns_panel(
        universe=universe,
        start=pd.Timestamp(start).date() if start else None,
        end=pd.Timestamp(end).date() if end else None,
    )
    if rets.empty:
        raise ValueError(f"No usable panel for {universe}; run `radar fetch --universe {universe}`.")
    if len(rets) < window:
        raise ValueError(f"Panel has {len(rets)} observations, fewer than the {window}-day window.")

    n_assets = len(rets.columns)
    k = n_clusters if n_clusters is not None else max(2, min(8, n_assets // 4))

    rows: list[dict] = []
    previous_labels: pd.Series | None = None
    previous_tree = None

    for i in range(0, len(rets) - window + 1, step):
        win = rets.iloc[i : i + window]
        corr = estimate_correlation(win, estimator).matrix
        dist = correlation_to_distance(corr)
        tree = build_mst(dist)
        clus = cluster_from_distance(dist)
        labels = clus.flat_clusters(k)
        purity = group_purity(tree, uni.groups)

        rows.append(
            {
                "window_end": win.index[-1],
                "mean_correlation": mean_correlation(corr),
                "absorption_ratio": absorption_ratio(corr),
                "pc1_share": absorption_ratio(corr, n_components=1),
                "effective_dimension": effective_dimension(corr),
                "tree_length": normalised_tree_length(tree),
                "cophenetic": clus.cophenetic_correlation,
                "depth_ratio": clus.depth_ratio(),
                "cluster_stability": (
                    cluster_stability(previous_labels, labels)
                    if previous_labels is not None else np.nan
                ),
                "edge_survival": (
                    edge_survival(previous_tree, tree) if previous_tree is not None else np.nan
                ),
                "purity": purity["purity"],
                "purity_lift": purity["lift"],
            }
        )
        previous_labels, previous_tree = labels, tree

    windows = pd.DataFrame(rows).set_index("window_end")

    notes = []
    if report.dropped:
        notes.append(f"{len(report.dropped)} members dropped: " +
                     ", ".join(sorted(report.dropped)))
    if uni.caveats:
        notes.append(uni.caveats)

    return Diagnostic(
        universe=universe,
        n_assets=n_assets,
        n_obs=len(rets),
        start=report.start,
        end=report.end,
        window=window,
        estimator=estimator,
        windows=windows,
        n_clusters=k,
        notes=notes,
    )
