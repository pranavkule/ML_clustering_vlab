"""Synthetic dataset generation helpers for the clustering virtual lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

DATASET_NAMES = ["Blobs", "Moons", "Circles", "Noisy blobs", "Custom random"]

GROUP_DATASETS = {"Blobs", "Noisy blobs", "Custom random"}

SPREAD_CONFIG: dict[str, tuple[str, float, float, float, float]] = {
    "Blobs": ("Cluster spread (standard deviation)", 0.1, 3.0, 0.1, 0.9),
    "Moons": ("Noise", 0.0, 0.30, 0.01, 0.08),
    "Circles": ("Noise", 0.0, 0.30, 0.01, 0.08),
    "Noisy blobs": ("Cluster spread (standard deviation)", 0.1, 3.0, 0.1, 0.9),
    "Custom random": ("Cluster spread (standard deviation)", 0.1, 4.0, 0.1, 1.2),
}


@dataclass
class GeneratedDataset:
    """A two-dimensional dataset and the settings used to create it."""

    X: np.ndarray
    y_true: np.ndarray
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    signature: tuple[Any, ...] = ()
    version: int = 0

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_groups(self) -> int | None:
        return self.params.get("n_groups")


def spread_config(dataset_name: str) -> tuple[str, float, float, float, float]:
    """Return (label, min, max, step, default) for the current dataset type."""

    return SPREAD_CONFIG[dataset_name]


def _validate_common(
    dataset_name: str,
    n_samples: int,
    noise: float,
    n_groups: int,
    random_state: int,
) -> None:
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset type: {dataset_name!r}.")
    if int(n_samples) < 2:
        raise ValueError("The dataset must contain at least two samples.")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("Dataset noise or spread must be a non-negative number.")
    if int(n_groups) < 1:
        raise ValueError("The number of generated groups must be at least one.")
    if int(n_groups) > int(n_samples):
        raise ValueError("The number of generated groups cannot exceed the number of samples.")
    if int(random_state) < 0:
        raise ValueError("The random seed must be a non-negative integer.")


def generate_dataset(
    dataset_name: str,
    n_samples: int = 320,
    noise: float = 0.9,
    n_groups: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate a deterministic two-feature dataset.

    ``noise`` is interpreted as the cluster standard deviation for blob-style
    datasets and as the additive noise level for the moons and circles datasets.
    """

    _validate_common(dataset_name, n_samples, noise, n_groups, random_state)
    n_samples = int(n_samples)
    n_groups = int(n_groups)
    random_state = int(random_state)
    noise = float(noise)

    if dataset_name == "Blobs":
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=n_groups,
            cluster_std=noise,
            random_state=random_state,
            shuffle=True,
        )
        params = {
            "n_samples": n_samples,
            "n_groups": n_groups,
            "cluster_std": noise,
            "random_state": random_state,
        }
    elif dataset_name == "Noisy blobs":
        n_main = max(n_groups, int(round(n_samples * 0.90)))
        X_main, y_main = make_blobs(
            n_samples=n_main,
            n_features=2,
            centers=n_groups,
            cluster_std=noise,
            random_state=random_state,
            shuffle=True,
        )
        n_extra = n_samples - n_main
        rng = np.random.default_rng(random_state + 17)
        if n_extra > 0:
            lower = X_main.min(axis=0) - 1.0
            upper = X_main.max(axis=0) + 1.0
            X_extra = rng.uniform(lower, upper, size=(n_extra, 2))
            X = np.vstack([X_main, X_extra])
            y = np.concatenate([y_main, np.full(n_extra, -1, dtype=int)])
            order = rng.permutation(len(X))
            X, y = X[order], y[order]
        else:
            X, y = X_main, y_main
        params = {
            "n_samples": n_samples,
            "n_groups": n_groups,
            "cluster_std": noise,
            "background_noise_fraction": 0.10,
            "random_state": random_state,
        }
    elif dataset_name == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        params = {
            "n_samples": n_samples,
            "n_groups": 2,
            "noise": noise,
            "random_state": random_state,
        }
    elif dataset_name == "Circles":
        X, y = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=0.5,
            random_state=random_state,
        )
        params = {
            "n_samples": n_samples,
            "n_groups": 2,
            "noise": noise,
            "factor": 0.5,
            "random_state": random_state,
        }
    else:  # Custom random
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=n_groups,
            cluster_std=noise,
            center_box=(-12.0, 12.0),
            random_state=random_state,
            shuffle=True,
        )
        params = {
            "n_samples": n_samples,
            "n_groups": n_groups,
            "cluster_std": noise,
            "center_box": (-12.0, 12.0),
            "random_state": random_state,
        }

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("Internal error: generated datasets must have exactly two features.")
    if not np.all(np.isfinite(X)):
        raise ValueError("The generated dataset contains non-finite values.")
    return X, y, params


def dataset_signature(
    dataset_name: str,
    n_samples: int,
    noise: float,
    n_groups: int,
    random_state: int,
) -> tuple[Any, ...]:
    """Build a comparable signature for the dataset controls."""

    return (
        dataset_name,
        int(n_samples),
        round(float(noise), 6),
        int(n_groups),
        int(random_state),
    )
