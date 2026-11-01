"""K-Means execution and an explicit educational step-by-step implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from modules.evaluation import SilhouetteReport


@dataclass
class KMeansResult:
    """Outcome of a Scikit-learn K-Means run."""

    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    n_iter: int
    n_clusters_requested: int
    n_clusters_obtained: int
    init_method: str
    random_state: int
    max_iter: int
    converged: bool | None
    tolerance: float = 1e-4


def _validate_kmeans_inputs(
    X: np.ndarray,
    n_clusters: int,
    max_iter: int,
    random_state: int,
    tol: float,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("K-Means requires a two-dimensional feature matrix with two features.")
    if len(X) < 2:
        raise ValueError("K-Means requires at least two data points.")
    if not np.all(np.isfinite(X)):
        raise ValueError("The dataset contains non-finite values and cannot be clustered.")
    if not (1 <= int(n_clusters) <= len(X)):
        raise ValueError("K must be between 1 and the number of data points.")
    if int(max_iter) < 1:
        raise ValueError("Maximum iterations must be at least 1.")
    if int(random_state) < 0:
        raise ValueError("The random seed must be a non-negative integer.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("The convergence tolerance must be a positive number.")
    return X


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 3,
    init_method: str = "k-means++",
    max_iter: int = 300,
    random_state: int = 42,
    *,
    n_init: int = 10,
    tol: float = 1e-4,
) -> KMeansResult:
    """Run Scikit-learn K-Means and return the fitted result.

    Scikit-learn is configured with ``n_init`` independent initialisations and
    keeps the solution with the lowest inertia.  This is intentionally different
    from the single-initialisation educational stepper in this module.
    """

    X = _validate_kmeans_inputs(X, n_clusters, max_iter, random_state, tol)
    if init_method not in {"k-means++", "random"}:
        raise ValueError("Initialization must be either 'k-means++' or 'random'.")

    model = KMeans(
        n_clusters=int(n_clusters),
        init=init_method,
        n_init=int(n_init),
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=int(random_state),
        algorithm="lloyd",
    )
    labels = model.fit_predict(X)
    labels = np.asarray(labels, dtype=int)
    n_obtained = int(len(np.unique(labels)))
    converged = bool(model.n_iter_ < int(max_iter)) if int(max_iter) > 1 else None

    return KMeansResult(
        labels=labels,
        centroids=np.asarray(model.cluster_centers_, dtype=float),
        inertia=float(model.inertia_),
        n_iter=int(model.n_iter_),
        n_clusters_requested=int(n_clusters),
        n_clusters_obtained=n_obtained,
        init_method=init_method,
        random_state=int(random_state),
        max_iter=int(max_iter),
        converged=converged,
        tolerance=float(tol),
    )


def _squared_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.sum(diff * diff, axis=2)


class KMeansStepper:
    """A single-initialisation Lloyd K-Means implementation for teaching.

    The class exposes assignment and update as separate operations so that a
    student can inspect point assignments before the centroids move.  Euclidean
    distance is used throughout.  Empty clusters are re-seeded with the sample
    farthest from its currently assigned centroid, which prevents a centroid
    from becoming permanently unused.
    """

    def __init__(
        self,
        X: np.ndarray,
        n_clusters: int = 3,
        init_method: str = "k-means++",
        max_iter: int = 300,
        random_state: int = 42,
        *,
        tol: float = 1e-6,
    ) -> None:
        self.X = _validate_kmeans_inputs(X, n_clusters, max_iter, random_state, tol)
        if init_method not in {"k-means++", "random"}:
            raise ValueError("Initialization must be either 'k-means++' or 'random'.")

        self.n_clusters = int(n_clusters)
        self.init_method = init_method
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.tol = float(tol)

        self.rng = np.random.default_rng(self.random_state)
        self.centroids = self._initialize_centroids()
        self.initial_centroids = self.centroids.copy()
        self.previous_centroids: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.inertia: float | None = None
        self.iteration = 0
        self.phase = "assign"
        self.converged = False
        self.centroid_shift: float | None = None
        self.empty_cluster_fix = False

    @property
    def signature(self) -> tuple[int, str, int, int, float]:
        return (
            self.n_clusters,
            self.init_method,
            self.max_iter,
            self.random_state,
            self.tol,
        )

    @property
    def n_clusters_obtained(self) -> int:
        if self.labels is None:
            return self.n_clusters
        return int(len(np.unique(self.labels)))

    def _initialize_centroids(self) -> np.ndarray:
        X = self.X
        n_samples = len(X)
        if self.init_method == "random":
            indices = self.rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()

        centers = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        first_index = int(self.rng.integers(n_samples))
        centers[0] = X[first_index]
        chosen = {first_index}
        closest_sq = np.sum((X - centers[0]) ** 2, axis=1)

        for cluster_index in range(1, self.n_clusters):
            total = float(closest_sq.sum())
            if total <= 1e-12:
                candidates = [i for i in range(n_samples) if i not in chosen]
                if not candidates:
                    candidates = list(range(n_samples))
                next_index = int(candidates[int(self.rng.integers(len(candidates)))])
            else:
                probabilities = closest_sq / total
                next_index = int(self.rng.choice(n_samples, p=probabilities))
            centers[cluster_index] = X[next_index]
            chosen.add(next_index)
            new_closest_sq = np.sum((X - centers[cluster_index]) ** 2, axis=1)
            closest_sq = np.minimum(closest_sq, new_closest_sq)

        return centers

    def _inertia_for(self, labels: np.ndarray, centroids: np.ndarray) -> float:
        distances = _squared_distances(self.X, centroids)
        return float(distances[np.arange(len(self.X)), labels].sum())

    def assign_step(self) -> None:
        """Assign every point to its nearest current centroid."""

        if self.phase != "assign" or self.converged:
            return
        distances = _squared_distances(self.X, self.centroids)
        self.labels = np.argmin(distances, axis=1).astype(int)
        self.inertia = float(distances[np.arange(len(self.X)), self.labels].sum())
        self.previous_centroids = None
        self.phase = "update"

    def update_step(self) -> None:
        """Move each centroid to the mean of its assigned points."""

        if self.phase != "update" or self.labels is None or self.converged:
            return

        old_centroids = self.centroids.copy()
        new_centroids = old_centroids.copy()
        distances = _squared_distances(self.X, old_centroids)
        self.empty_cluster_fix = False

        for cluster_index in range(self.n_clusters):
            members = self.labels == cluster_index
            if np.any(members):
                new_centroids[cluster_index] = self.X[members].mean(axis=0)
            else:
                assigned_distances = distances[np.arange(len(self.X)), self.labels]
                farthest_index = int(np.argmax(assigned_distances))
                new_centroids[cluster_index] = self.X[farthest_index]
                self.empty_cluster_fix = True

        shifts = np.linalg.norm(new_centroids - old_centroids, axis=1)
        self.centroid_shift = float(shifts.max()) if len(shifts) else 0.0
        self.previous_centroids = old_centroids
        self.centroids = new_centroids
        self.iteration += 1
        self.inertia = self._inertia_for(self.labels, self.centroids)

        if self.centroid_shift <= self.tol or self.iteration >= self.max_iter:
            self.converged = True
            self.phase = "converged"
        else:
            self.phase = "assign"

    def run_to_convergence(self) -> None:
        """Continue alternating assignment and update until convergence."""

        guard = 0
        maximum_steps = (self.max_iter * 2) + 5
        while not self.converged and guard < maximum_steps:
            if self.phase == "assign":
                self.assign_step()
            elif self.phase == "update":
                self.update_step()
            else:
                break
            guard += 1

    def state_summary(self) -> str:
        if self.phase == "assign":
            return "Centroids are initialized. Next step: assign each point to its nearest centroid."
        if self.phase == "update":
            return "Points are assigned. Next step: move each centroid to the mean of its points."
        if self.converged:
            return (
                f"Converged after {self.iteration} iteration(s) using a centroid-shift "
                f"tolerance of {self.tol:g} or the iteration limit."
            )
        return "Ready."


def kmeans_observation(
    result: KMeansResult,
    silhouette: SilhouetteReport,
    dataset_name: str,
) -> str:
    """Produce a short observation grounded in the fitted result."""

    parts = [
        f"K-Means formed {result.n_clusters_obtained} cluster(s) after "
        f"{result.n_iter} iteration(s), with inertia {result.inertia:.2f}."
    ]

    if result.n_clusters_requested == 1:
        parts.append("With K = 1 all points belong to one cluster, so cluster separation is not defined.")
    elif silhouette.available and silhouette.score is not None:
        score = silhouette.score
        if score >= 0.60:
            parts.append(
                f"The Silhouette Score of {score:.3f} indicates well-separated clusters on this run."
            )
        elif score >= 0.30:
            parts.append(
                f"The Silhouette Score of {score:.3f} indicates moderate separation, with some "
                "overlap between clusters."
            )
        elif score >= 0.0:
            parts.append(
                f"The Silhouette Score of {score:.3f} suggests weak separation; many points are "
                "near cluster boundaries."
            )
        else:
            parts.append(
                f"The Silhouette Score of {score:.3f} is negative, suggesting that some points may "
                "be assigned to unsuitable clusters."
            )
    elif silhouette.reason:
        parts.append(silhouette.reason)

    if result.n_clusters_obtained < result.n_clusters_requested:
        parts.append(
            f"The fitted result has {result.n_clusters_obtained} non-empty cluster(s) even though "
            f"K = {result.n_clusters_requested}; some clusters did not retain any points."
        )

    if result.converged is False:
        parts.append(
            "The maximum iteration limit was reached before the centroids stabilized; increasing "
            "the limit may change the result."
        )

    if dataset_name in {"Moons", "Circles"}:
        parts.append(
            "K-Means partitions points by distance to centroids, so crescent or concentric "
            "structures can be split into compact regions rather than followed as continuous shapes."
        )

    return " ".join(parts)
