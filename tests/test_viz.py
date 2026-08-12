from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from radar.viz import events, layout
from radar.viz import plots


def make_tree(n: int = 8, seed: int = 0) -> nx.Graph:
    rng = np.random.default_rng(seed)
    graph = nx.random_labeled_tree(n, seed=int(rng.integers(1e6)))
    graph = nx.relabel_nodes(graph, {i: f"T{i}" for i in graph.nodes()})
    for _, _, data in graph.edges(data=True):
        data["weight"] = float(rng.uniform(0.5, 1.5))
    return graph


def test_chained_layouts_cover_every_frame_and_node():
    trees = {pd.Timestamp("2020-01-0%d" % (i + 1)): make_tree(seed=i) for i in range(3)}
    frame = layout.chained_layouts(trees)
    assert set(frame.columns) == {"window_end", "ticker", "x", "y"}
    assert len(frame) == 3 * 8
    assert frame["window_end"].nunique() == 3
    assert frame[["x", "y"]].notna().all().all()


def test_identical_trees_barely_move_between_frames():
    """The whole reason layouts are chained: an unchanged tree must produce an
    unchanged picture, or the animation invents motion that is not in the data."""
    tree = make_tree(seed=3)
    trees = {pd.Timestamp("2020-01-01"): tree, pd.Timestamp("2020-01-02"): tree.copy()}
    drift = layout.layout_drift(layout.chained_layouts(trees))
    assert drift.iloc[-1] < 0.05


def test_cold_layouts_would_scramble_by_comparison():
    """Contrast case: independently seeded layouts of the same tree move far more."""
    tree = make_tree(seed=3)
    a = nx.spring_layout(tree, iterations=layout.INITIAL_ITERATIONS, seed=1)
    b = nx.spring_layout(tree, iterations=layout.INITIAL_ITERATIONS, seed=2)
    cold = np.mean([np.hypot(*(np.array(a[n]) - np.array(b[n]))) for n in tree.nodes()])

    trees = {pd.Timestamp("2020-01-01"): tree, pd.Timestamp("2020-01-02"): tree.copy()}
    warm = layout.layout_drift(layout.chained_layouts(trees)).iloc[-1]
    assert warm < cold


def test_layouts_are_reproducible():
    trees = {pd.Timestamp("2020-01-0%d" % (i + 1)): make_tree(seed=i) for i in range(2)}
    first = layout.chained_layouts(trees)
    second = layout.chained_layouts(trees)
    pd.testing.assert_frame_equal(first, second)


def test_layout_drift_first_frame_is_undefined():
    trees = {pd.Timestamp("2020-01-0%d" % (i + 1)): make_tree(seed=i) for i in range(2)}
    drift = layout.layout_drift(layout.chained_layouts(trees))
    assert np.isnan(drift.iloc[0])


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_events_are_sorted_and_parseable():
    dates = [pd.Timestamp(e.date) for e in events.EVENTS]
    assert dates == sorted(dates)
    assert all(e.label for e in events.EVENTS)


def test_events_between_filters_inclusively():
    picked = events.events_between("2008-01-01", "2008-12-31")
    assert [e.label for e in picked] == ["Lehman"]
    assert events.events_between("1990-01-01", "1991-01-01") == ()


# ---------------------------------------------------------------------------
# Figures -- smoke tests: they must render without raising on real-shaped input.
# ---------------------------------------------------------------------------


@pytest.fixture
def artifact_like():
    idx = pd.date_range("2020-01-01", periods=40, freq="W-FRI")
    rng = np.random.default_rng(0)
    windows = pd.DataFrame(
        {
            "absorption_ratio": rng.uniform(0.5, 0.9, 40),
            "mean_correlation": rng.uniform(0.1, 0.6, 40),
            "edge_survival": rng.uniform(0.6, 1.0, 40),
            "layout_drift": rng.uniform(0.0, 0.2, 40),
        },
        index=idx,
    )
    tree = make_tree(8)
    edges = pd.DataFrame(
        [{"source": u, "target": v, "distance": d["weight"]} for u, v, d in tree.edges(data=True)]
    )
    positions = layout.chained_layouts({idx[0]: tree}).set_index("ticker")[["x", "y"]]
    groups = {f"T{i}": f"G{i % 3}" for i in range(8)}
    return windows, edges, positions, groups


def test_network_figure_renders(artifact_like):
    _, edges, positions, groups = artifact_like
    fig = plots.network_figure(edges, positions, groups, title="t", highlight="T0")
    assert fig.axes


def test_index_and_stability_figures_render(artifact_like):
    windows, *_ = artifact_like
    assert plots.index_figure(windows, current=windows.index[10]).axes
    assert plots.stability_figure(windows, current=windows.index[10]).axes


def test_heatmap_and_spectrum_render():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((300, 10))
    corr = pd.DataFrame(np.corrcoef(data, rowvar=False),
                        index=[f"A{i}" for i in range(10)],
                        columns=[f"A{i}" for i in range(10)])
    assert plots.heatmap_figure(corr, list(corr.columns)).axes
    assert plots.spectrum_figure(np.linalg.eigvalsh(corr)[::-1], lambda_plus=1.4).axes


def test_group_colours_are_stable_and_distinct():
    groups = {"A": "x", "B": "y", "C": "x"}
    colours = plots.group_colours(groups)
    assert colours == plots.group_colours(groups)
    assert colours["x"] != colours["y"]
