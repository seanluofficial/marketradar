"""Precomputed rolling-window artifacts.

Everything the app displays is computed here, offline, and written to parquet. The
deployed app loads arrays and renders; it never estimates and never needs an API key.
That is what makes the time scrubber feel instant, and it is also what makes a result
reproducible: an artifact carries the spec and package version that produced it.

**What is and is not stored.** Scalar metrics and MST edges are stored per window. Full
correlation matrices are not: 80x80 floats across ~1300 windows is ~66 MB per estimator,
and recomputing a single window's matrix on demand takes about a millisecond. So the
artifact holds the time series, and the selected frame's matrix is rebuilt live.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import radar
from radar.config import ARTIFACT_DIR
from radar.data import returns_panel
from radar.data.universe import get_universe
from radar.metrics.absorption import (
    absorption_ratio,
    effective_dimension,
    mean_correlation,
    n_components_for,
    pc1_share,
)
from radar.structure.correlation import DEFAULT_ESTIMATOR, estimate_correlation
from radar.structure.distance import correlation_to_distance
from radar.viz.layout import chained_layouts, layout_drift
from radar.structure.mst import (
    build_mst,
    edge_survival,
    group_purity,
    hub,
    normalised_tree_length,
    to_frame,
)


@dataclass(frozen=True)
class ArtifactSpec:
    """Everything needed to reproduce an artifact byte for byte."""

    universe: str = "core_equity"
    start: str = "2000-01-01"
    end: str | None = None
    window: int = 252
    step: int = 5
    estimator: str = DEFAULT_ESTIMATOR

    @property
    def name(self) -> str:
        return f"{self.universe}_{self.estimator}_w{self.window}_s{self.step}"


@dataclass
class Artifact:
    """A loaded artifact: per-window metrics, MST edges, layouts, and provenance."""

    spec: ArtifactSpec
    windows: pd.DataFrame
    edges: pd.DataFrame
    layouts: pd.DataFrame = field(default_factory=pd.DataFrame)
    returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: dict = field(default_factory=dict)

    @property
    def window_ends(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.windows.index)

    def edges_at(self, window_end) -> pd.DataFrame:
        """MST edges for one frame."""
        stamp = pd.Timestamp(window_end)
        return self.edges[self.edges["window_end"] == stamp].drop(columns=["window_end"])

    def layout_at(self, window_end) -> pd.DataFrame:
        """Node positions for one frame, indexed by ticker."""
        stamp = pd.Timestamp(window_end)
        frame = self.layouts[self.layouts["window_end"] == stamp]
        return frame.set_index("ticker")[["x", "y"]]

    def correlation_at(self, window_end) -> pd.DataFrame:
        """Rebuild one window's correlation matrix from the artifact's own returns.

        The artifact ships the return panel it was built from (a few MB) rather than the
        correlation matrices (tens of MB), so a deployment needs the artifact directory
        and nothing else -- no price cache, no API key, no network.
        """
        if self.returns.empty:
            raise ValueError(
                "This artifact has no bundled returns; rebuild it with `radar build`."
            )
        stamp = pd.Timestamp(window_end)
        if stamp not in self.returns.index:
            raise KeyError(f"{stamp.date()} is not a trading day in the panel.")
        position = self.returns.index.get_loc(stamp)
        if position + 1 < self.spec.window:
            raise ValueError(
                f"Not enough history before {stamp.date()} for a {self.spec.window}-day window."
            )
        window = self.returns.iloc[position + 1 - self.spec.window : position + 1]
        return estimate_correlation(window, self.spec.estimator).matrix


def artifact_dir(spec: ArtifactSpec) -> Path:
    return ARTIFACT_DIR / spec.name


def build_artifact(
    spec: ArtifactSpec = ArtifactSpec(), progress: bool = True
) -> Artifact:
    """Compute every rolling window for `spec` and write it to disk."""
    universe = get_universe(spec.universe)
    groups = universe.groups

    rets, report = returns_panel(
        universe=spec.universe,
        start=pd.Timestamp(spec.start).date(),
        end=pd.Timestamp(spec.end).date() if spec.end else None,
    )
    if rets.empty:
        raise ValueError(f"No usable return panel for {spec.universe}; run `radar fetch`.")
    if len(rets) < spec.window:
        raise ValueError(
            f"Panel has {len(rets)} observations, fewer than the {spec.window}-day window."
        )

    starts = range(0, len(rets) - spec.window + 1, spec.step)
    rows: list[dict] = []
    edge_frames: list[pd.DataFrame] = []
    trees: dict = {}
    previous_tree = None

    for count, i in enumerate(starts, start=1):
        win = rets.iloc[i : i + spec.window]
        window_end = win.index[-1]

        result = estimate_correlation(win, spec.estimator)
        corr = result.matrix
        tree = build_mst(correlation_to_distance(corr))
        purity = group_purity(tree, groups)

        rows.append(
            {
                "window_end": window_end,
                "window_start": win.index[0],
                "n_assets": result.n_assets,
                "n_obs": result.n_obs,
                "q": result.q,
                "mean_correlation": mean_correlation(corr),
                "absorption_ratio": absorption_ratio(corr),
                "pc1_share": pc1_share(corr),
                "effective_dimension": effective_dimension(corr),
                "tree_length": normalised_tree_length(tree),
                "edge_survival": (
                    edge_survival(previous_tree, tree) if previous_tree is not None else np.nan
                ),
                "purity": purity["purity"],
                "purity_lift": purity["lift"],
                "hub": hub(tree),
                "n_factors": result.diagnostics.get("n_factors", np.nan),
                "lambda_plus": result.diagnostics.get("lambda_plus", np.nan),
                "sigma_squared": result.diagnostics.get("sigma_squared", np.nan),
                "shrinkage": result.diagnostics.get("shrinkage", np.nan),
                "condition_number": result.diagnostics["condition_number"],
            }
        )

        frame = to_frame(tree)
        frame.insert(0, "window_end", window_end)
        edge_frames.append(frame)
        trees[window_end] = tree
        previous_tree = tree

        if progress and count % 100 == 0:
            print(f"  {count}/{len(starts)} windows ({window_end.date()})", flush=True)

    windows = pd.DataFrame(rows).set_index("window_end").sort_index()
    edges = pd.concat(edge_frames, ignore_index=True)

    if progress:
        print("  chaining layouts...", flush=True)
    layouts = chained_layouts(trees)
    windows["layout_drift"] = layout_drift(layouts).reindex(windows.index)

    meta = {
        "spec": asdict(spec),
        "radar_version": radar.__version__,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "n_windows": len(windows),
        "assets": list(rets.columns),
        "group_label": universe.group_label,
        "groups": {t: groups[t] for t in rets.columns if t in groups},
        "absorption_components": n_components_for(len(rets.columns)),
        "panel": {
            "requested": len(report.requested),
            "retained": report.n_assets,
            "dropped": report.dropped,
            "start": str(report.start),
            "end": str(report.end),
            "n_obs": report.n_obs,
            "forward_filled": report.filled_values,
        },
        "caveats": universe.caveats,
    }

    out = artifact_dir(spec)
    out.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out / "windows.parquet")
    edges.to_parquet(out / "edges.parquet")
    layouts.to_parquet(out / "layouts.parquet")
    rets.to_parquet(out / "returns.parquet")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    return Artifact(
        spec=spec, windows=windows, edges=edges, layouts=layouts, returns=rets, meta=meta
    )


def load_artifact(spec: ArtifactSpec = ArtifactSpec()) -> Artifact:
    """Read a previously built artifact from disk."""
    out = artifact_dir(spec)
    if not (out / "windows.parquet").exists():
        raise FileNotFoundError(
            f"No artifact at {out}. Build it with `radar build --universe {spec.universe}`."
        )
    windows = pd.read_parquet(out / "windows.parquet")
    edges = pd.read_parquet(out / "edges.parquet")
    layout_path = out / "layouts.parquet"
    layouts = pd.read_parquet(layout_path) if layout_path.exists() else pd.DataFrame()
    returns_path = out / "returns.parquet"
    rets = pd.read_parquet(returns_path) if returns_path.exists() else pd.DataFrame()
    meta = json.loads((out / "meta.json").read_text())
    return Artifact(
        spec=spec, windows=windows, edges=edges, layouts=layouts, returns=rets, meta=meta
    )


def correlation_at(spec: ArtifactSpec, window_end) -> pd.DataFrame:
    """Rebuild one window's correlation matrix on demand.

    The deliberate counterpart to not storing matrices in the artifact: a single window
    costs about a millisecond, so the app recomputes the selected frame rather than
    carrying tens of megabytes it will almost never read.
    """
    rets, _ = returns_panel(
        universe=spec.universe,
        start=pd.Timestamp(spec.start).date(),
        end=pd.Timestamp(spec.end).date() if spec.end else None,
    )
    stamp = pd.Timestamp(window_end)
    if stamp not in rets.index:
        raise KeyError(f"{stamp.date()} is not a trading day in the panel.")
    position = rets.index.get_loc(stamp)
    if position + 1 < spec.window:
        raise ValueError(f"Not enough history before {stamp.date()} for a {spec.window}-day window.")
    win = rets.iloc[position + 1 - spec.window : position + 1]
    return estimate_correlation(win, spec.estimator).matrix
