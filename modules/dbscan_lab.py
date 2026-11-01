"""DBSCAN execution and point-type identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from modules.evaluation import SilhouetteReport


@dataclass
class DBSCANResult:
    """Outcome of a Scikit-learn DBSCAN run."""

    labels: np.ndarray
    core_mask: np.ndarray
    n_clusters: int
    n_noise: int
    n_core: int
    n_border: int
    eps: float
    min_samples: int
    metric: str

    @property
    def n_clustered(self) -> int:
        return int(len(self.labels) - self.n_noise)

    @property
    def noise_fraction(self) -> float:
        if len(self.labels) == 0:
            return 0.0
        return float(self.n_noise / len(self.labels))


def run_dbscan(
    X: np.ndarray,
    eps: float = 0.40,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> DBSCANResult:
    """Fit DBSCAN and classify core, border, and noise points from the model."""

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("DBSCAN requires a two-dimensional feature matrix with two features.")
    if len(X) < 2:
        raise ValueError("DBSCAN requires at least two data points.")
    if not np.all(np.isfinite(X)):
        raise ValueError("The dataset contains non-finite values and cannot be clustered.")
    if not np.isfinite(eps) or float(eps) <= 0:
        raise ValueError("Epsilon (eps) must be a positive number.")
    if int(min_samples) < 1:
        raise ValueError("MinPts (min_samples) must be at least 1.")
    if int(min_samples) > len(X):
        raise ValueError("MinPts cannot exceed the number of data points.")
    if metric not in {"euclidean", "manhattan", "cosine"}:
        raise ValueError("Unsupported distance metric.")

    model = DBSCAN(
        eps=float(eps),
        min_samples=int(min_samples),
        metric=metric,
        algorithm="auto",
    )
    labels = np.asarray(model.fit_predict(X), dtype=int)

    core_mask = np.zeros(len(X), dtype=bool)
    core_mask[np.asarray(model.core_sample_indices_, dtype=int)] = True

    unique_labels = set(int(label) for label in np.unique(labels))
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))
    n_core = int(np.sum(core_mask))
    n_border = int(np.sum((labels != -1) & (~core_mask)))

    return DBSCANResult(
        labels=labels,
        core_mask=core_mask,
        n_clusters=int(n_clusters),
        n_noise=n_noise,
        n_core=n_core,
        n_border=n_border,
        eps=float(eps),
        min_samples=int(min_samples),
        metric=metric,
    )


def dbscan_observation(
    result: DBSCANResult,
    silhouette: SilhouetteReport,
    dataset_name: str,
) -> str:
    """Produce a short observation grounded in the fitted DBSCAN result."""

    parts = [
        f"DBSCAN found {result.n_clusters} cluster(s) using eps = {result.eps:.2f} and "
        f"MinPts = {result.min_samples}: {result.n_core} core point(s), "
        f"{result.n_border} border point(s), and {result.n_noise} noise point(s)."
    ]

    if result.n_clusters == 0:
        parts.append(
            "Every point was labelled as noise. This usually means eps is too small for the local "
            "density or MinPts is too high."
        )
    elif result.n_clusters == 1:
        parts.append(
            "All clustered points were assigned to a single density-connected group. This can happen "
            "when eps is large enough to connect separate dense regions, or when the data genuinely "
            "contains one dense group."
        )

    if result.n_noise == 0:
        parts.append("No points were labelled as noise on this run.")
    elif result.noise_fraction >= 0.20:
        parts.append(
            f"Noise accounts for {result.noise_fraction * 100:.1f}% of all points, suggesting that "
            "the density threshold is strict for this dataset."
        )
    else:
        parts.append(
            f"Noise accounts for {result.noise_fraction * 100:.1f}% of all points."
        )

    if silhouette.available and silhouette.score is not None:
        parts.append(
            f"The Silhouette Score of {silhouette.score:.3f} was computed after excluding "
            f"{silhouette.n_noise} noise point(s) from the evaluation."
        )
    elif silhouette.reason:
        parts.append(silhouette.reason)

    if dataset_name in {"Moons", "Circles"} and result.n_clusters >= 2:
        parts.append(
            "Because DBSCAN follows density connectivity rather than centroid distance, it can "
            "group points along non-convex shapes when eps and MinPts suit the local density."
        )

    if result.n_clusters >= 2 and result.n_noise > 0:
        parts.append(
            "The clusters shown are the density-connected regions; DBSCAN does not compute "
            "centroids as part of its standard algorithm."
        )

    return " ".join(parts)
