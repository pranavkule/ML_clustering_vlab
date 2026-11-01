"""Plotly figures for clustering experiments."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

CLUSTER_COLORS = [
    "#1f4e79",
    "#2e75b6",
    "#548235",
    "#b07d1a",
    "#8c3b1e",
    "#6b4c9a",
    "#0f766e",
    "#9b1c31",
    "#4b5563",
    "#7c6f10",
]

UNASSIGNED_COLOR = "#9ca3af"
NOISE_COLOR = "#111827"
FONT_FAMILY = '"Segoe UI", "Helvetica Neue", Arial, sans-serif'


def cluster_color(label: int) -> str:
    if label < 0:
        return NOISE_COLOR
    return CLUSTER_COLORS[label % len(CLUSTER_COLORS)]


def _base_layout(title: str, *, height: int = 470) -> dict:
    return {
        "title": {
            "text": title,
            "x": 0.0,
            "xanchor": "left",
            "font": {"size": 15, "color": "#1f2933", "family": FONT_FAMILY},
        },
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"family": FONT_FAMILY, "size": 12, "color": "#374151"},
        "margin": {"l": 58, "r": 24, "t": 58, "b": 92},
        "height": height,
        "autosize": True,
        "hoverlabel": {
            "bgcolor": "#ffffff",
            "bordercolor": "#d1d5db",
            "font": {"family": FONT_FAMILY, "size": 12, "color": "#1f2933"},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 11, "color": "#4b5563"},
            "bgcolor": "rgba(0,0,0,0)",
            "traceorder": "normal",
        },
        "xaxis": {
            "title": {"text": "Feature 1", "font": {"size": 12}},
            "showgrid": True,
            "gridcolor": "#eef0f3",
            "zeroline": False,
            "linecolor": "#d1d5db",
            "ticks": "outside",
            "tickcolor": "#d1d5db",
        },
        "yaxis": {
            "title": {"text": "Feature 2", "font": {"size": 12}},
            "showgrid": True,
            "gridcolor": "#eef0f3",
            "zeroline": False,
            "linecolor": "#d1d5db",
            "ticks": "outside",
            "tickcolor": "#d1d5db",
        },
    }


def plot_raw_dataset(X: np.ndarray, title: str = "Original dataset (unlabelled)") -> go.Figure:
    X = np.asarray(X, dtype=float)
    indices = np.arange(len(X))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            name="Data points",
            marker={"color": UNASSIGNED_COLOR, "size": 7, "opacity": 0.8},
            customdata=indices,
            hovertemplate=(
                "Point %{customdata}<br>"
                "Feature 1: %{x:.2f}<br>"
                "Feature 2: %{y:.2f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(**_base_layout(title))
    return fig


def plot_kmeans_clusters(
    X: np.ndarray,
    labels: np.ndarray | None,
    centroids: np.ndarray | None,
    title: str = "K-Means clustering result",
    *,
    previous_centroids: np.ndarray | None = None,
    show_centroids: bool = True,
) -> go.Figure:
    X = np.asarray(X, dtype=float)
    fig = go.Figure()

    if labels is None:
        fig.add_trace(
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode="markers",
                name="Unassigned points",
                marker={"color": UNASSIGNED_COLOR, "size": 7, "opacity": 0.8},
                customdata=np.arange(len(X)),
                hovertemplate=(
                    "Point %{customdata}<br>"
                    "Feature 1: %{x:.2f}<br>"
                    "Feature 2: %{y:.2f}<br>"
                    "Status: not yet assigned<extra></extra>"
                ),
                showlegend=False,
            )
        )
    else:
        labels = np.asarray(labels, dtype=int)
        for cluster_label in sorted(int(v) for v in np.unique(labels) if v >= 0):
            mask = labels == cluster_label
            point_indices = np.flatnonzero(mask)
            color = cluster_color(cluster_label)
            customdata = np.column_stack([point_indices, np.full(len(point_indices), cluster_label)])
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode="markers",
                    name=f"Cluster {cluster_label}",
                    marker={
                        "color": color,
                        "size": 7,
                        "opacity": 0.82,
                        "line": {"width": 0.6, "color": "#ffffff"},
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "Cluster %{customdata[1]}<br>"
                        "Point %{customdata[0]}<br>"
                        "Feature 1: %{x:.2f}<br>"
                        "Feature 2: %{y:.2f}<extra></extra>"
                    ),
                )
            )

    if show_centroids and centroids is not None:
        centroids = np.asarray(centroids, dtype=float)
        if previous_centroids is not None:
            previous = np.asarray(previous_centroids, dtype=float)
            if previous.shape == centroids.shape and not np.allclose(previous, centroids, atol=1e-9):
                line_x: list[float | None] = []
                line_y: list[float | None] = []
                for old, new in zip(previous, centroids):
                    line_x.extend([old[0], new[0], None])
                    line_y.extend([old[1], new[1], None])
                fig.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode="lines",
                        name="Centroid shift",
                        line={"color": "#9ca3af", "width": 1.5, "dash": "dot"},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=previous[:, 0],
                        y=previous[:, 1],
                        mode="markers",
                        name="Previous centroid",
                        marker={
                            "color": "rgba(0,0,0,0)",
                            "size": 11,
                            "symbol": "diamond-open",
                            "line": {"width": 1.5, "color": "#9ca3af"},
                        },
                        hovertemplate=(
                            "Previous centroid<br>"
                            "Feature 1: %{x:.2f}<br>"
                            "Feature 2: %{y:.2f}<extra></extra>"
                        ),
                    )
                )

        centroid_numbers = np.arange(len(centroids))
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers",
                name="Centroid",
                marker={
                    "color": "#111827",
                    "size": 13,
                    "symbol": "x",
                    "line": {"width": 1.5, "color": "#ffffff"},
                },
                customdata=centroid_numbers,
                hovertemplate=(
                    "Centroid %{customdata}<br>"
                    "Feature 1: %{x:.2f}<br>"
                    "Feature 2: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(**_base_layout(title))
    return fig


def plot_dbscan_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    core_mask: np.ndarray,
    title: str = "DBSCAN clustering result",
) -> go.Figure:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    core_mask = np.asarray(core_mask, dtype=bool)
    fig = go.Figure()

    core_indices = np.flatnonzero((labels != -1) & core_mask)
    border_indices = np.flatnonzero((labels != -1) & (~core_mask))
    noise_indices = np.flatnonzero(labels == -1)

    if len(core_indices):
        core_colors = [cluster_color(int(labels[i])) for i in core_indices]
        customdata = np.column_stack([core_indices, labels[core_indices]])
        fig.add_trace(
            go.Scatter(
                x=X[core_indices, 0],
                y=X[core_indices, 1],
                mode="markers",
                name="Core points (filled)",
                marker={
                    "color": core_colors,
                    "size": 8,
                    "symbol": "circle",
                    "line": {"width": 0.7, "color": "#ffffff"},
                },
                customdata=customdata,
                hovertemplate=(
                    "Point %{customdata[0]}<br>"
                    "Cluster %{customdata[1]}<br>"
                    "Type: Core<br>"
                    "Feature 1: %{x:.2f}<br>"
                    "Feature 2: %{y:.2f}<extra></extra>"
                ),
            )
        )

    if len(border_indices):
        border_colors = [cluster_color(int(labels[i])) for i in border_indices]
        customdata = np.column_stack([border_indices, labels[border_indices]])
        fig.add_trace(
            go.Scatter(
                x=X[border_indices, 0],
                y=X[border_indices, 1],
                mode="markers",
                name="Border points (open)",
                marker={
                    "color": "rgba(0,0,0,0)",
                    "size": 9,
                    "symbol": "circle-open",
                    "line": {"width": 2.0, "color": border_colors},
                },
                customdata=customdata,
                hovertemplate=(
                    "Point %{customdata[0]}<br>"
                    "Cluster %{customdata[1]}<br>"
                    "Type: Border<br>"
                    "Feature 1: %{x:.2f}<br>"
                    "Feature 2: %{y:.2f}<extra></extra>"
                ),
            )
        )

    if len(noise_indices):
        fig.add_trace(
            go.Scatter(
                x=X[noise_indices, 0],
                y=X[noise_indices, 1],
                mode="markers",
                name="Noise points (x)",
                marker={
                    "color": NOISE_COLOR,
                    "size": 8,
                    "symbol": "x",
                    "line": {"width": 1.0, "color": NOISE_COLOR},
                },
                customdata=noise_indices,
                hovertemplate=(
                    "Point %{customdata}<br>"
                    "Type: Noise (label -1)<br>"
                    "Feature 1: %{x:.2f}<br>"
                    "Feature 2: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(**_base_layout(title))
    if not core_indices.size and not border_indices.size and not noise_indices.size:
        fig.add_annotation(
            text="No points to display.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": "#6b7280"},
        )
    return fig
