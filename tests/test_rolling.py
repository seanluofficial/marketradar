from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from radar.data.universe import get_universe
from radar.metrics import rolling
from radar.structure.correlation import estimate_correlation

SPEC = rolling.ArtifactSpec(
    universe="sector_etfs", start="2020-01-01", window=60, step=20, estimator="ledoit_wolf"
)


@pytest.fixture
def small_artifact(tmp_path, monkeypatch, make_series):
    monkeypatch.setattr(rolling, "ARTIFACT_DIR", tmp_path / "artifacts")
    for i, ticker in enumerate(get_universe("sector_etfs").tickers):
        make_series(ticker, start="2020-01-01", periods=200, seed=i)
    return rolling.build_artifact(SPEC, progress=False)


def test_spec_name_encodes_every_parameter_that_changes_the_output():
    name = SPEC.name
    for part in ("sector_etfs", "ledoit_wolf", "w60", "s20"):
        assert part in name


def test_build_produces_the_expected_number_of_windows(small_artifact):
    # 200 business days -> 199 returns; range(0, 199-60+1, 20) -> 7 windows.
    assert len(small_artifact.windows) == 7
    assert small_artifact.windows.index.is_monotonic_increasing


def test_every_window_has_a_full_spanning_tree(small_artifact):
    n_assets = small_artifact.windows["n_assets"].iloc[0]
    counts = small_artifact.edges.groupby("window_end").size()
    assert (counts == n_assets - 1).all()
    assert len(counts) == len(small_artifact.windows)


def test_metrics_are_in_range(small_artifact):
    w = small_artifact.windows
    assert w["absorption_ratio"].between(0, 1).all()
    assert w["pc1_share"].between(0, 1).all()
    assert w["mean_correlation"].between(-1, 1).all()
    assert w["effective_dimension"].between(1, w["n_assets"].max() + 1e-9).all()
    assert (w["tree_length"] > 0).all()
    assert w["purity"].between(0, 1).all()


def test_edge_survival_is_undefined_for_the_first_window_only(small_artifact):
    survival = small_artifact.windows["edge_survival"]
    assert np.isnan(survival.iloc[0])
    assert survival.iloc[1:].notna().all()
    assert survival.iloc[1:].between(0, 1).all()


def test_meta_records_provenance(small_artifact, tmp_path):
    meta = small_artifact.meta
    assert meta["spec"]["window"] == 60
    assert meta["spec"]["estimator"] == "ledoit_wolf"
    assert meta["radar_version"]
    assert meta["n_windows"] == 7
    assert len(meta["assets"]) == meta["panel"]["retained"]
    assert "caveat" in meta["caveats"].lower() or meta["caveats"]

    on_disk = json.loads((rolling.artifact_dir(SPEC) / "meta.json").read_text())
    assert on_disk["spec"] == meta["spec"]


def test_artifact_roundtrips_through_disk(small_artifact):
    loaded = rolling.load_artifact(SPEC)
    pd.testing.assert_frame_equal(loaded.windows, small_artifact.windows)
    pd.testing.assert_frame_equal(loaded.edges, small_artifact.edges)
    assert loaded.meta["spec"] == small_artifact.meta["spec"]


def test_edges_at_selects_one_frame(small_artifact):
    stamp = small_artifact.window_ends[3]
    frame = small_artifact.edges_at(stamp)
    assert "window_end" not in frame.columns
    assert len(frame) == small_artifact.windows["n_assets"].iloc[0] - 1
    assert frame["distance"].is_monotonic_increasing


def test_loading_a_missing_artifact_points_at_the_build_command(tmp_path, monkeypatch):
    monkeypatch.setattr(rolling, "ARTIFACT_DIR", tmp_path / "nothing")
    with pytest.raises(FileNotFoundError, match="radar build"):
        rolling.load_artifact(SPEC)


def test_correlation_at_matches_a_direct_recomputation(small_artifact, monkeypatch):
    """The artifact stores no matrices; the app rebuilds the selected frame. That
    rebuild must reproduce exactly what the stored metrics were computed from."""
    from radar.data import returns_panel

    stamp = small_artifact.window_ends[4]
    rebuilt = rolling.correlation_at(SPEC, stamp)

    rets, _ = returns_panel(universe="sector_etfs", start=pd.Timestamp("2020-01-01").date())
    position = rets.index.get_loc(stamp)
    direct = estimate_correlation(
        rets.iloc[position + 1 - SPEC.window : position + 1], SPEC.estimator
    ).matrix
    pd.testing.assert_frame_equal(rebuilt, direct)

    # And it must agree with the metric stored for that window.
    from radar.metrics.absorption import mean_correlation

    stored = small_artifact.windows.loc[stamp, "mean_correlation"]
    assert mean_correlation(rebuilt) == pytest.approx(stored)


def test_correlation_at_rejects_a_non_trading_day(small_artifact):
    with pytest.raises(KeyError):
        rolling.correlation_at(SPEC, "2020-01-01")


def test_window_longer_than_the_panel_is_rejected(small_artifact):
    spec = rolling.ArtifactSpec(universe="sector_etfs", start="2020-01-01", window=5000)
    with pytest.raises(ValueError, match="fewer than"):
        rolling.build_artifact(spec, progress=False)


def test_empty_panel_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(rolling, "ARTIFACT_DIR", tmp_path / "artifacts")
    with pytest.raises(ValueError, match="radar fetch"):
        rolling.build_artifact(SPEC, progress=False)
