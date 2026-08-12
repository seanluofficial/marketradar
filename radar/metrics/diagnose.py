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
from radar.structure.hierarchy import (
    DEFAULT_METHOD,
    cluster_from_distance,
    cluster_stability,
)
from radar.structure.correlation import DEFAULT_ESTIMATOR, estimate_correlation
from radar.structure.distance import correlation_to_distance
from radar.structure.mst import build_mst, edge_survival, group_purity, normalised_tree_length


@dataclass
class AllocationSensitivity:
    """Does the clustering step change the answer, or just the story?

    Deliberately a *diagnostic*, not an allocator. The weight functions below are
    minimal reference implementations whose only job is to measure whether hierarchical
    structure moves the weights relative to naive alternatives. The allocator project
    owns its own production versions, pinned to its own version -- importing these would
    blur the seam between "describe the structure" and "build a portfolio".
    """

    universe: str
    n_assets: int
    rebalances: int
    frame: pd.DataFrame

    def summary(self) -> str:
        f = self.frame
        lines = [
            f"=== {self.universe}: {self.n_assets} assets, {self.rebalances} rebalances ===",
            "",
            "  agreement between methods (weight vectors)",
            f"    max |HRP - inverse-var|   median {f['max_gap_hrp_ivp'].median():.4f}"
            f"   [{f['max_gap_hrp_ivp'].min():.4f} - {f['max_gap_hrp_ivp'].max():.4f}]",
            f"    max |HRP - equal weight|  median {f['max_gap_hrp_ew'].median():.4f}",
            f"    corr(HRP, inverse-var)    median {f['corr_hrp_ivp'].median():.3f}",
            "",
            "  concentration (effective number of positions, max = n_assets)",
            f"    HRP           {f['eff_n_hrp'].median():.2f}",
            f"    inverse-var   {f['eff_n_ivp'].median():.2f}",
            f"    equal weight  {self.n_assets:.2f}",
            "",
            "  turnover per rebalance (sum of absolute weight changes)",
            f"    HRP           median {f['turnover_hrp'].median():.4f}"
            f"   p90 {f['turnover_hrp'].quantile(0.9):.4f}",
            f"    inverse-var   median {f['turnover_ivp'].median():.4f}"
            f"   p90 {f['turnover_ivp'].quantile(0.9):.4f}",
            f"    ratio HRP/IVP {f['turnover_hrp'].median() / f['turnover_ivp'].median():.2f}x",
        ]
        return "\n".join(lines)


def inverse_variance_weights(cov: pd.DataFrame, items: list[str]) -> pd.Series:
    """Naive risk parity: weight inversely to variance, ignoring correlation entirely."""
    variances = pd.Series(np.diag(cov.loc[items, items]), index=items)
    weights = 1.0 / variances
    return weights / weights.sum()


def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    w = inverse_variance_weights(cov, items).to_numpy()
    return float(w @ cov.loc[items, items].to_numpy() @ w)


def hrp_weights(cov: pd.DataFrame, order: list[str]) -> pd.Series:
    """Recursive bisection over the seriated order (López de Prado 2016, stage 3).

    Note what this actually does: it splits the *ordered list* in half by position, not
    the dendrogram by topology. So the quality of the result depends entirely on the
    ordering being meaningful. If seriation is driven by noise-level differences, the
    split point is arbitrary and the weights inherit that instability.
    """
    weights = pd.Series(1.0, index=order)
    clusters = [list(order)]
    while clusters:
        split = []
        for cluster in clusters:
            if len(cluster) > 1:
                mid = len(cluster) // 2
                split += [cluster[:mid], cluster[mid:]]
        clusters = split
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left, var_right = _cluster_variance(cov, left), _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
    return weights


def effective_positions(weights: pd.Series) -> float:
    """Inverse Herfindahl: how many positions the portfolio effectively holds."""
    return float(1.0 / np.square(weights.to_numpy()).sum())


def allocation_sensitivity(
    universe: str,
    start: str | None = None,
    end: str | None = None,
    window: int = 252,
    step: int = 21,
    estimator: str = "ledoit_wolf",
    method: str = DEFAULT_METHOD,
) -> AllocationSensitivity:
    """Compare HRP, inverse-variance and equal weights across rolling rebalances.

    `step` defaults to 21 observations -- roughly monthly -- because turnover is only
    meaningful relative to a realistic rebalance schedule.
    """
    rets, _ = returns_panel(
        universe=universe,
        start=pd.Timestamp(start).date() if start else None,
        end=pd.Timestamp(end).date() if end else None,
    )
    if rets.empty:
        raise ValueError(f"No usable panel for {universe}; run `radar fetch --universe {universe}`.")
    if len(rets) < window:
        raise ValueError(f"Panel has {len(rets)} observations, fewer than the {window}-day window.")

    tickers = list(rets.columns)
    n_assets = len(tickers)
    equal = pd.Series(1.0 / n_assets, index=tickers)

    rows: list[dict] = []
    previous: dict[str, pd.Series] = {}

    for i in range(0, len(rets) - window + 1, step):
        win = rets.iloc[i : i + window]
        corr = estimate_correlation(win, estimator).matrix
        sd = win.std(ddof=1)
        cov = corr.mul(sd, axis=0).mul(sd, axis=1)

        clus = cluster_from_distance(correlation_to_distance(corr), method=method)
        hrp = hrp_weights(cov, clus.order).reindex(tickers)
        ivp = inverse_variance_weights(cov, tickers)

        rows.append(
            {
                "window_end": win.index[-1],
                "max_gap_hrp_ivp": float((hrp - ivp).abs().max()),
                "max_gap_hrp_ew": float((hrp - equal).abs().max()),
                "corr_hrp_ivp": float(np.corrcoef(hrp, ivp)[0, 1]),
                "eff_n_hrp": effective_positions(hrp),
                "eff_n_ivp": effective_positions(ivp),
                "max_weight_hrp": float(hrp.max()),
                "turnover_hrp": (
                    float((hrp - previous["hrp"]).abs().sum()) if "hrp" in previous else np.nan
                ),
                "turnover_ivp": (
                    float((ivp - previous["ivp"]).abs().sum()) if "ivp" in previous else np.nan
                ),
            }
        )
        previous = {"hrp": hrp, "ivp": ivp}

    frame = pd.DataFrame(rows).set_index("window_end")
    return AllocationSensitivity(
        universe=universe, n_assets=n_assets, rebalances=len(frame), frame=frame
    )


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
