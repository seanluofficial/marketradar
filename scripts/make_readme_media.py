"""Render the README's hero animation from a committed artifact.

Reads the artifact and nothing else -- no network, no price cache -- so the media in the
README is reproducible from what is in the repository.

    python scripts/make_readme_media.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar.metrics.rolling import ArtifactSpec, load_artifact  # noqa: E402
from radar.viz.plots import PALETTE, group_colours  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "media"

#: COVID rather than 2008: the collapse and the partial recovery both fall inside a
#: two-year span, so a short loop shows the structure moving in both directions.
START, END = "2019-06-01", "2021-06-01"
FRAME_STRIDE = 2          # artifact windows are weekly; every other one -> ~2 weeks
FPS = 8


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    artifact = load_artifact(ArtifactSpec())
    windows, groups = artifact.windows, artifact.meta["groups"]
    colours = group_colours(groups)

    ends = artifact.window_ends
    selected = [d for d in ends if pd.Timestamp(START) <= d <= pd.Timestamp(END)][::FRAME_STRIDE]
    print(f"{len(selected)} frames, {selected[0].date()} -> {selected[-1].date()}")

    fig, (ax_net, ax_idx) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    fig.patch.set_facecolor("white")

    # The index panel is static except for the marker, so draw it once.
    ax_idx.plot(windows.index, windows["absorption_ratio"], color="#C44E52", lw=1.5,
                label="Absorption ratio")
    ax_idx.plot(windows.index, windows["mean_correlation"], color="#4C72B0", lw=1.1,
                alpha=0.8, label="Mean correlation")
    ax_idx.set_xlim(pd.Timestamp("2001-01-01"), windows.index[-1])
    ax_idx.set_ylim(0, 1)
    ax_idx.legend(loc="upper left", fontsize=8, frameon=False)
    ax_idx.grid(alpha=0.2)
    ax_idx.tick_params(labelsize=8)
    marker = ax_idx.axvline(selected[0], color="black", lw=1.8)

    def draw(stamp):
        ax_net.clear()
        layout = artifact.layout_at(stamp)
        edges = artifact.edges_at(stamp)
        widest = max(edges["distance"].max(), 1e-9)

        for _, e in edges.iterrows():
            src, dst = e["source"], e["target"]
            if src in layout.index and dst in layout.index:
                ax_net.plot(
                    [layout.loc[src, "x"], layout.loc[dst, "x"]],
                    [layout.loc[src, "y"], layout.loc[dst, "y"]],
                    color="#B8BCC4",
                    lw=0.7 + 2.0 * (1.0 - e["distance"] / widest),
                    zorder=1,
                )
        ax_net.scatter(
            layout["x"], layout["y"],
            c=[colours.get(groups.get(t, ""), "#8C8C8C") for t in layout.index],
            s=60, zorder=3, edgecolor="white", linewidth=0.6,
        )
        ax_net.set_xticks([])
        ax_net.set_yticks([])
        for spine in ax_net.spines.values():
            spine.set_visible(False)

        # Redrawn every frame because clear() wipes it. Without this the colours are
        # decoration; with it, the viewer can see the sectors separating and fusing.
        ax_net.legend(
            handles=[
                Line2D([], [], marker="o", ls="", color=colour, label=name, markersize=5)
                for name, colour in colours.items()
            ],
            loc="upper left", fontsize=6.2, frameon=False, ncol=2,
            handletextpad=0.3, columnspacing=0.9, labelspacing=0.35,
        )

        row = windows.loc[stamp]
        ax_net.set_title(
            f"{stamp.date()}   ·   absorption {row['absorption_ratio']:.3f}   ·   "
            f"{row['effective_dimension']:.1f} independent directions",
            fontsize=10.5,
        )
        marker.set_xdata([stamp, stamp])
        return ()

    anim = FuncAnimation(fig, draw, frames=selected, blit=False)
    gif = OUT / "structure-covid.gif"
    anim.save(gif, writer=PillowWriter(fps=FPS), dpi=76,
              savefig_kwargs={"facecolor": "white"})
    plt.close(fig)
    print(f"wrote {gif}  ({gif.stat().st_size / 1e6:.2f} MB)")

    # A static frame at the stress peak, for contexts that do not animate.
    peak = windows["absorption_ratio"].idxmax()
    fig, (ax_net, ax_idx) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    fig.patch.set_facecolor("white")
    ax_idx.plot(windows.index, windows["absorption_ratio"], color="#C44E52", lw=1.5,
                label="Absorption ratio")
    ax_idx.plot(windows.index, windows["mean_correlation"], color="#4C72B0", lw=1.1,
                alpha=0.8, label="Mean correlation")
    ax_idx.set_ylim(0, 1)
    ax_idx.legend(loc="lower left", fontsize=8, frameon=False)
    ax_idx.grid(alpha=0.2)
    ax_idx.tick_params(labelsize=8)
    marker = ax_idx.axvline(peak, color="black", lw=1.8)
    draw(peak)
    fig.tight_layout()
    png = OUT / "structure-peak.png"
    fig.savefig(png, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"wrote {png}  ({png.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    build()
