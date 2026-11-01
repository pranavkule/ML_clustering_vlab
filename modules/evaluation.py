"""Clustering evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class SilhouetteReport:
    """Result of an attempted Silhouette Score calculation."""

    score: float | None
    available: bool
    reason: str | None
    n_evaluated: int
    n_clusters: int
    n_noise: int
    excluded_noise: bool
    metric: str = "euclidean"

    def display_value(self) -> str:
        return f"{self.score:.3f}" if self.available and self.score is not None else "Not available"


def compute_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    exclude_noise: bool = True,
    metric: str = "euclidean",
) -> SilhouetteReport:
    """Compute the Silhouette Score, returning an explicit unavailable state.

    The standard Silhouette Score is defined only when there are at least two
    clusters and fewer clusters than evaluated samples.  When ``exclude_noise``
    is true, points with label ``-1`` (DBSCAN noise) are removed before the
    calculation and the number removed is reported.
    """

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    if X.ndim != 2:
        return SilhouetteReport(
            score=None,
            available=False,
            reason="Silhouette is unavailable because the feature matrix is not two-dimensional.",
            n_evaluated=0,
            n_clusters=0,
            n_noise=0,
            excluded_noise=exclude_noise,
            metric=metric,
        )
    if labels.ndim != 1 or len(labels) != len(X):
        return SilhouetteReport(
            score=None,
            available=False,
            reason="Silhouette is unavailable because labels and samples do not match.",
            n_evaluated=0,
            n_clusters=0,
            n_noise=0,
            excluded_noise=exclude_noise,
            metric=metric,
        )
    if not np.issubdtype(labels.dtype, np.integer):
        return SilhouetteReport(
            score=None,
            available=False,
            reason="Silhouette is unavailable because cluster labels must be integers.",
            n_evaluated=0,
            n_clusters=0,
            n_noise=0,
            excluded_noise=exclude_noise,
            metric=metric,
        )

    total_samples = len(labels)
    if exclude_noise:
        keep = labels != -1
        X_eval = X[keep]
        labels_eval = labels[keep]
        n_noise = int(total_samples - keep.sum())
    else:
        X_eval = X
        labels_eval = labels
        n_noise = 0

    n_evaluated = int(len(labels_eval))
    if n_evaluated == 0:
        return SilhouetteReport(
            score=None,
            available=False,
            reason=(
                "Silhouette is unavailable because every point was labelled as noise "
                "and there are no clustered points to evaluate."
            ),
            n_evaluated=0,
            n_clusters=0,
            n_noise=n_noise,
            excluded_noise=exclude_noise,
            metric=metric,
        )

    unique_labels = np.unique(labels_eval)
    n_clusters = int(len(unique_labels))

    if n_clusters < 2:
        if n_clusters == 0:
            reason = "Silhouette is unavailable because there are no clusters to compare."
        else:
            reason = (
                "Silhouette is unavailable because the result contains only one cluster. "
                "At least two clusters are required for a meaningful comparison."
            )
        return SilhouetteReport(
            score=None,
            available=False,
            reason=reason,
            n_evaluated=n_evaluated,
            n_clusters=n_clusters,
            n_noise=n_noise,
            excluded_noise=exclude_noise,
            metric=metric,
        )

    if n_clusters >= n_evaluated:
        return SilhouetteReport(
            score=None,
            available=False,
            reason=(
                "Silhouette is unavailable because there are as many clusters as evaluated "
                "points. The score requires fewer clusters than samples."
            ),
            n_evaluated=n_evaluated,
            n_clusters=n_clusters,
            n_noise=n_noise,
            excluded_noise=exclude_noise,
            metric=metric,
        )

    if n_evaluated < 2:
        return SilhouetteReport(
            score=None,
            available=False,
            reason="Silhouette is unavailable because fewer than two points remain for evaluation.",
            n_evaluated=n_evaluated,
            n_clusters=n_clusters,
            n_noise=n_noise,
            excluded_noise=exclude_noise,
            metric=metric,
        )

    try:
        score = float(silhouette_score(X_eval, labels_eval, metric=metric))
    except Exception as exc:  # pragma: no cover - defensive for unusual metrics
        return SilhouetteReport(
            score=None,
            available=False,
            reason=f"Silhouette could not be computed for this configuration: {exc}",
            n_evaluated=n_evaluated,
            n_clusters=n_clusters,
            n_noise=n_noise,
            excluded_noise=exclude_noise,
            metric=metric,
        )

    if not np.isfinite(score):
        return SilhouetteReport(
            score=None,
            available=False,
            reason="Silhouette could not be computed because the result was not a finite number.",
            n_evaluated=n_evaluated,
            n_clusters=n_clusters,
            n_noise=n_noise,
            excluded_noise=exclude_noise,
            metric=metric,
        )

    return SilhouetteReport(
        score=score,
        available=True,
        reason=None,
        n_evaluated=n_evaluated,
        n_clusters=n_clusters,
        n_noise=n_noise,
        excluded_noise=exclude_noise,
        metric=metric,
    )
