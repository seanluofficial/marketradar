"""Matplotlib figures for the app. No estimation happens here -- these read the artifact."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from radar.viz.events import events_between

#: Qualitative palette, ordered so adjacent sectors stay distinguishable.
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
    "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#B0754C",
]


def group_colours(groups: dict[str, str]) -> dict[str, str]:
    names = sorted(set(groups.values()))
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(names)}


def network_figure(
    edges: pd.DataFrame,
    layout: pd.DataFrame,
    groups: dict[str, str],
    title: str = "",
    highlight: str | None = None,
) -> Figure:
    """The MST, drawn at precomputed warm-started positions.

    Edge width encodes proximity: short distances (strong correlation) are drawn heavier,
    so a crisis tree looks visibly denser as well as more compact.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    colours = group_colours(groups)

    if not edges.empty:
        widths = 1.0 + 2.5 * (1.0 - edges["distance"] / max(edges["distance"].max(), 1e-9))
        for (_, edge), width in zip(edges.iterrows(), widths):
            src, dst = edge["source"], edge["target"]
            if src not in layout.index or dst not in layout.index:
                continue
            ax.plot(
                [layout.loc[src, "x"], layout.loc[dst, "x"]],
                [layout.loc[src, "y"], layout.loc[dst, "y"]],
                color="#B0B0B0", linewidth=float(width), zorder=1, alpha=0.8,
            )

    for ticker, row in layout.iterrows():
        group = groups.get(ticker, "")
        is_focus = ticker == highlight
        ax.scatter(
            row["x"], row["y"],
            s=300 if is_focus else 130,
            color=colours.get(group, "#8C8C8C"),
            edgecolor="black" if is_focus else "white",
            linewidth=2.0 if is_focus else 0.8,
            zorder=3,
        )
        ax.annotate(
            ticker, (row["x"], row["y"]),
            fontsize=8 if not is_focus else 10,
            fontweight="bold" if is_focus else "normal",
            ha="center", va="center", zorder=4,
        )

    handles = [
        Line2D([], [], marker="o", linestyle="", color=colour, label=name, markersize=8)
        for name, colour in colours.items()
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False, ncol=2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def index_figure(
    windows: pd.DataFrame,
    current=None,
    annotate: bool = True,
) -> Figure:
    """Absorption ratio with mean correlation overlaid.

    The two are plotted together deliberately: they correlate at ~0.85, so showing the
    index alone would overstate how much independent information it carries.
    """
    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(windows.index, windows["absorption_ratio"], color="#C44E52",
            linewidth=1.6, label="Absorption ratio (top 20% of eigenvectors)")
    ax.plot(windows.index, windows["mean_correlation"], color="#4C72B0",
            linewidth=1.2, alpha=0.8, label="Mean pairwise correlation")
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of variance / correlation")

    if annotate:
        for event in events_between(windows.index[0], windows.index[-1]):
            stamp = pd.Timestamp(event.date)
            ax.axvline(stamp, color="#8C8C8C", linestyle=":", linewidth=0.9, zorder=0)
            ax.annotate(
                event.label, (stamp, 0.02), rotation=90, fontsize=7,
                color="#555555", ha="right", va="bottom",
            )

    if current is not None:
        ax.axvline(pd.Timestamp(current), color="black", linewidth=1.8, alpha=0.9)

    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def stability_figure(windows: pd.DataFrame, current=None) -> Figure:
    """Edge survival and layout drift: how much of the picture is real.

    Without this panel a viewer cannot tell whether a reorganising tree reflects the
    market or the estimator. Included at the same prominence as the index for that reason.
    """
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.plot(windows.index, windows["edge_survival"], color="#55A868", linewidth=1.3,
            label="MST edge survival (vs previous window)")
    ax.axhline(1.0, color="#B0B0B0", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction retained")

    if "layout_drift" in windows.columns and windows["layout_drift"].notna().any():
        twin = ax.twinx()
        twin.plot(windows.index, windows["layout_drift"], color="#8172B3",
                  linewidth=1.0, alpha=0.7, label="mean node displacement")
        twin.set_ylabel("layout drift")
        twin.legend(loc="lower right", fontsize=8, frameon=False)

    if current is not None:
        ax.axvline(pd.Timestamp(current), color="black", linewidth=1.8, alpha=0.9)

    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def heatmap_figure(corr: pd.DataFrame, order: list[str], title: str = "") -> Figure:
    """Correlation matrix reordered by the clustering -- the quasi-diagonal view.

    Block structure along the diagonal is the visual form of "there are clusters here".
    A uniform wash means there are not.
    """
    ordered = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(ordered.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_yticklabels(order, fontsize=6)
    ax.set_title(title, fontsize=11)
    fig.colorbar(image, ax=ax, shrink=0.8, label="correlation")
    fig.tight_layout()
    return fig


def spectrum_figure(spectrum: np.ndarray, lambda_plus: float | None = None) -> Figure:
    """Eigenvalue spectrum against the Marchenko-Pastur noise edge.

    Eigenvalues to the left of the line are statistically indistinguishable from noise.
    Seeing how few sit to the right is the most direct way to understand why the raw
    sample correlation matrix should not be trusted.
    """
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(1, len(spectrum) + 1), spectrum, color="#4C72B0", width=0.9)
    if lambda_plus is not None and np.isfinite(lambda_plus):
        ax.axhline(lambda_plus, color="#C44E52", linestyle="--", linewidth=1.2,
                   label=f"MP noise edge = {lambda_plus:.2f}")
        ax.legend(fontsize=8, frameon=False)
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("eigenvalue (log)")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig
