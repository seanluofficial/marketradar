"""Market Structure Radar -- Streamlit app.

Run with:  streamlit run radar/app/main.py

Reads precomputed artifacts and rebuilds only the one selected window's correlation
matrix. No API key required, no network access, nothing fitted at view time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

if __package__ is None:  # allow `streamlit run radar/app/main.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from radar.config import ARTIFACT_DIR
from radar.metrics.absorption import eigen_spectrum
from radar.metrics.rolling import ArtifactSpec, correlation_at, load_artifact
from radar.structure.distance import correlation_to_distance, nearest_neighbours
from radar.structure.hierarchy import cluster_from_distance
from radar.viz.plots import (
    heatmap_figure,
    index_figure,
    network_figure,
    spectrum_figure,
    stability_figure,
)

st.set_page_config(page_title="Market Structure Radar", layout="wide")


def available_artifacts() -> list[str]:
    if not ARTIFACT_DIR.exists():
        return []
    return sorted(p.name for p in ARTIFACT_DIR.iterdir() if (p / "windows.parquet").exists())


def spec_from_name(name: str) -> ArtifactSpec:
    """Artifact directory names encode the spec: <universe>_<estimator>_w<W>_s<S>."""
    body, window, step = name.rsplit("_w", 1)[0], *name.rsplit("_w", 1)[1].split("_s")
    for estimator in ("rmt_clipped", "ledoit_wolf", "sample"):
        if body.endswith("_" + estimator):
            return ArtifactSpec(
                universe=body[: -(len(estimator) + 1)],
                estimator=estimator,
                window=int(window),
                step=int(step),
            )
    raise ValueError(f"Cannot parse artifact name {name!r}")


@st.cache_resource(show_spinner="Loading artifact...")
def get_artifact(name: str):
    spec = spec_from_name(name)
    return spec, load_artifact(spec)


@st.cache_data(show_spinner="Rebuilding window...")
def get_correlation(name: str, window_end: pd.Timestamp) -> pd.DataFrame:
    return correlation_at(spec_from_name(name), window_end)


names = available_artifacts()
if not names:
    st.error(
        f"No artifacts found in {ARTIFACT_DIR}.\n\n"
        "Build one first:\n\n"
        "```\nradar fetch --universe all\nradar build --start 2000-01-01\n```"
    )
    st.stop()

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Market Structure Radar")
name = st.sidebar.selectbox("Artifact", names)
spec, artifact = get_artifact(name)
windows = artifact.windows
ends = artifact.window_ends

st.sidebar.caption(
    f"{len(windows)} windows of {spec.window} trading days, "
    f"estimator `{spec.estimator}`, step {spec.step}d"
)

index = st.sidebar.slider(
    "Window ending", 0, len(ends) - 1, len(ends) - 1,
    format="",
)
current = ends[index]
row = windows.loc[current]

st.sidebar.markdown(f"### {current.date()}")
st.sidebar.metric("Absorption ratio", f"{row['absorption_ratio']:.3f}")
st.sidebar.metric("Effective dimension", f"{row['effective_dimension']:.1f}",
                  help="Independent directions the market is moving in. "
                       "Falls toward 1 as everything couples.")
st.sidebar.metric("Mean correlation", f"{row['mean_correlation']:.3f}")
st.sidebar.metric("MST tree length", f"{row['tree_length']:.3f}")
st.sidebar.metric(
    "Edge survival", "n/a" if pd.isna(row["edge_survival"]) else f"{row['edge_survival']:.3f}",
    help="Fraction of MST edges retained from the previous window. Low values mean the "
         "picture changed; they do not mean the market did.",
)

focus = st.sidebar.selectbox("Inspect asset", ["(none)"] + list(artifact.meta["assets"]))
focus = None if focus == "(none)" else focus

# -------------------------------------------------------------------------------- body
st.title("Market Structure Radar")
st.caption(
    "Descriptive risk analytics. Nothing here predicts returns -- it shows how assets "
    "have been moving together, and how that has changed."
)

groups = artifact.meta["groups"]
tab_network, tab_index, tab_matrix, tab_honesty = st.tabs(
    ["Network", "Systemic risk", "Matrix & spectrum", "How to read this"]
)

with tab_network:
    left, right = st.columns([3, 1])
    with left:
        if artifact.layouts.empty:
            st.warning("This artifact predates stored layouts. Rebuild with `radar build`.")
        else:
            st.pyplot(
                network_figure(
                    artifact.edges_at(current),
                    artifact.layout_at(current),
                    groups,
                    title=f"Minimum spanning tree, {spec.window}d window ending {current.date()}",
                    highlight=focus,
                ),
                clear_figure=True,
            )
    with right:
        st.markdown("**Hub**")
        st.write(row["hub"])
        st.markdown("**Sector agreement**")
        st.write(f"purity {row['purity']:.2f} ({row['purity_lift']:.1f}x random)")
        st.caption(
            "Sector labels are used only to score the tree, never to build it."
        )
        if focus:
            corr = get_correlation(name, current)
            neighbours = nearest_neighbours(correlation_to_distance(corr), focus, 6)
            st.markdown(f"**{focus} nearest**")
            st.dataframe(
                pd.DataFrame({
                    "distance": neighbours.round(3),
                    "group": [groups.get(t, "") for t in neighbours.index],
                }),
                use_container_width=True,
            )

with tab_index:
    st.pyplot(index_figure(windows, current=current), clear_figure=True)
    st.caption(
        "Absorption ratio (Kritzman, Page & Turkington 2010) with mean pairwise "
        "correlation overlaid. On this data the two correlate at 0.85 -- the index is "
        "largely, though not entirely, a restatement of average correlation, and it is "
        "shown that way rather than presented as an independent construction. Event "
        "labels are annotations only; no metric uses them."
    )
    st.pyplot(stability_figure(windows, current=current), clear_figure=True)
    st.caption(
        "Between windows five trading days apart -- sharing 247 of 252 observations -- "
        "median edge survival is 0.873. Roughly 13% of the tree turns over from "
        "estimation noise alone, so not every reorganisation you see is a real one."
    )

with tab_matrix:
    corr = get_correlation(name, current)
    clus = cluster_from_distance(correlation_to_distance(corr))
    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(
            heatmap_figure(corr, clus.order, f"Correlation, clustered order, {current.date()}"),
            clear_figure=True,
        )
    with col_b:
        st.pyplot(
            spectrum_figure(eigen_spectrum(corr), row.get("lambda_plus")),
            clear_figure=True,
        )
        st.metric("Cophenetic correlation", f"{clus.cophenetic_correlation:.3f}",
                  help="How faithfully the tree represents the distances.")
        st.metric("Depth ratio", f"{clus.depth_ratio():.3f}",
                  help="Share of tree height spent on the final merge. Near 1 means one "
                       "late merge dominates and there is little nested structure.")
        if pd.notna(row.get("n_factors")):
            st.metric("Eigenvalues above the noise edge", f"{int(row['n_factors'])} of "
                                                          f"{len(corr)}")

with tab_honesty:
    panel = artifact.meta["panel"]
    st.subheader("What this is measured on")
    st.write(
        f"**{panel['retained']} of {panel['requested']}** requested assets, "
        f"{panel['n_obs']} observations from {panel['start']} to {panel['end']}. "
        f"q = N/T = **{row['q']:.3f}** at the {spec.window}-day window."
    )
    if panel["dropped"]:
        st.write("**Dropped, with reasons:**")
        st.dataframe(
            pd.DataFrame(
                {"ticker": list(panel["dropped"]), "reason": list(panel["dropped"].values())}
            ),
            use_container_width=True, hide_index=True,
        )
    if panel["forward_filled"]:
        st.caption(
            f"{panel['forward_filled']} missing price cells were forward-filled, which "
            "assigns a zero return to a halted day."
        )

    st.subheader("Caveats")
    st.write(artifact.meta["caveats"])
    st.markdown(
        "- **This is not a signal.** The metrics describe the past window. None of them "
        "has been tested for predictive power, and the project makes no such claim.\n"
        "- **Overlapping windows** make every series autocorrelated by construction, so "
        "consecutive readings are not independent evidence.\n"
        "- **Absorption ratio and tree length correlate at −0.99** on this data. They are "
        "near-duplicates as scalar indices; the tree adds topology, not a second opinion.\n"
        "- **Estimator choice matters more than it looks.** Compare artifacts built with "
        "`sample`, `ledoit_wolf` and `rmt_clipped` to see how much of the structure is "
        "an artefact of estimation noise."
    )
    st.subheader("Provenance")
    st.json(
        {
            "spec": artifact.meta["spec"],
            "radar_version": artifact.meta["radar_version"],
            "built_at": artifact.meta["built_at"],
            "absorption_components": artifact.meta["absorption_components"],
        }
    )
