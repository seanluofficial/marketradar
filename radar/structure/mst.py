"""Minimum spanning tree of the distance matrix (Mantegna 1999).

The MST keeps the N-1 strongest links that connect every asset exactly once. It is the
hero visual: sectors organise themselves into branches with no labels supplied, and in a
crisis the tree collapses toward a single hub as everything correlates with everything.

It is also unstable, and this module treats that as a measurement problem rather than
something to hide. `edge_survival` (Onnela et al. 2003) reports how much of the tree
persists between consecutive windows, which separates genuine structural change from
estimation noise -- without it, a scrubber animation makes sampling error look like a
regime shift.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_mst(distance: pd.DataFrame) -> nx.Graph:
    """Minimum spanning tree over the complete graph of pairwise distances.

    Built through networkx rather than scipy.sparse.csgraph deliberately: the sparse
    routines encode "no edge" as a zero weight, so a pair of assets with correlation 1
    (distance exactly 0) would have its edge silently dropped. Identical or near-identical
    series are exactly the case where that matters, and N ~ 100 is far too small for the
    performance difference to justify the trap.
    """
    if distance.empty:
        raise ValueError("Distance matrix is empty.")
    if distance.shape[0] != distance.shape[1]:
        raise ValueError(f"Distance matrix must be square, got {distance.shape}.")
    if distance.isna().any().any():
        raise ValueError("Distance matrix contains NaNs.")

    graph = nx.Graph()
    graph.add_nodes_from(distance.columns)
    labels = list(distance.columns)
    values = distance.to_numpy(dtype=float)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            graph.add_edge(labels[i], labels[j], weight=float(values[i, j]))

    return nx.minimum_spanning_tree(graph, weight="weight", algorithm="kruskal")


def edge_set(mst: nx.Graph) -> set[frozenset]:
    """Undirected edges as an order-independent set, for comparing two trees."""
    return {frozenset((u, v)) for u, v in mst.edges()}


def edge_survival(previous: nx.Graph, current: nx.Graph) -> float:
    """Fraction of `previous`'s edges still present in `current` (Onnela et al. 2003).

    1.0 means the tree did not move; low values mean the topology churned. Interpreting a
    scrubber animation without this number is guesswork, because MST edges turn over
    substantially from pure estimation noise even in a placid market.

    Computed over the intersection of the two node sets, so a universe that gains or
    loses a member between windows does not register as spurious churn.
    """
    shared = set(previous.nodes()) & set(current.nodes())
    if len(shared) < 2:
        return float("nan")

    prev_edges = {e for e in edge_set(previous) if e <= shared}
    curr_edges = {e for e in edge_set(current) if e <= shared}
    if not prev_edges:
        return float("nan")
    return len(prev_edges & curr_edges) / len(prev_edges)


def normalised_tree_length(mst: nx.Graph) -> float:
    """Mean MST edge length -- overall market coupling in one number.

    Falls when everything correlates (crisis) and rises when the market decouples. A
    useful cross-check on the absorption ratio: the two are built from the same matrix
    but the tree length depends only on the strongest N-1 links.
    """
    weights = [d["weight"] for _, _, d in mst.edges(data=True)]
    return float(np.mean(weights)) if weights else float("nan")


def degrees(mst: nx.Graph) -> pd.Series:
    """Node degree, descending. The top of this list is the tree's hub."""
    return pd.Series(dict(mst.degree())).sort_values(ascending=False)


def hub(mst: nx.Graph) -> str:
    """Most connected node. In stress this is typically a broad-market proxy or a
    large-cap financial -- the thing everything else is hanging off."""
    return str(degrees(mst).index[0])


def group_purity(mst: nx.Graph, groups: dict[str, str]) -> dict:
    """How much of the tree's structure lines up with known sector labels.

    This quantifies the project's central descriptive claim -- that sector structure
    emerges from returns alone. `purity` is the share of MST edges joining two nodes of
    the same group; `baseline` is what that share would be if the same number of edges
    were drawn at random from all pairs; `lift` is the ratio.

    The labels are used only to *score* the tree, never to build it.
    """
    edges = [(u, v) for u, v in mst.edges() if u in groups and v in groups]
    if not edges:
        return {"purity": float("nan"), "baseline": float("nan"), "lift": float("nan"),
                "n_edges": 0}

    same = sum(1 for u, v in edges if groups[u] == groups[v])
    purity = same / len(edges)

    labelled = [g for n, g in groups.items() if n in set(mst.nodes())]
    counts = pd.Series(labelled).value_counts()
    total_pairs = len(labelled) * (len(labelled) - 1) / 2
    same_pairs = float((counts * (counts - 1) / 2).sum())
    baseline = same_pairs / total_pairs if total_pairs else float("nan")

    return {
        "purity": float(purity),
        "baseline": float(baseline),
        "lift": float(purity / baseline) if baseline else float("nan"),
        "n_edges": len(edges),
    }


def to_frame(mst: nx.Graph) -> pd.DataFrame:
    """MST edges as a tidy frame, shortest first. What the viz layer consumes."""
    rows = [{"source": u, "target": v, "distance": d["weight"]}
            for u, v, d in mst.edges(data=True)]
    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
