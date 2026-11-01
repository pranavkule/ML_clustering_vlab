from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import datasets, quiz
from modules.dbscan_lab import run_dbscan
from modules.evaluation import compute_silhouette
from modules.kmeans_lab import KMeansStepper, run_kmeans

app = FastAPI(title="Clustering Visualizer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _matrix(values: list[list[float]]) -> np.ndarray:
    try:
        X = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Data must contain numeric coordinates.") from exc
    if X.ndim != 2 or X.shape[1] != 2 or len(X) == 0:
        raise HTTPException(status_code=422, detail="Data must be a non-empty N-by-2 matrix.")
    if not np.all(np.isfinite(X)):
        raise HTTPException(status_code=422, detail="Data contains non-finite coordinates.")
    return X


class DatasetRequest(BaseModel):
    name: str
    samples: int = Field(default=320, ge=2, le=5000)
    noise: float = Field(default=0.08, ge=0.0, le=4.0)
    groups: int = Field(default=3, ge=1, le=8)
    seed: int = Field(default=42, ge=0)


class ExperimentData(BaseModel):
    X: list[list[float]]


class KMeansRequest(ExperimentData):
    k: int = Field(default=3, ge=1, le=20)
    init: Literal["k-means++", "random"] = "k-means++"
    max_iter: int = Field(default=300, ge=1, le=1000)
    seed: int = Field(default=42, ge=0)


class DBSCANRequest(ExperimentData):
    eps: float = Field(default=0.4, gt=0.0, le=100.0)
    min_samples: int = Field(default=5, ge=1, le=5000)
    metric: Literal["euclidean", "manhattan", "cosine"] = "euclidean"


class CompareRequest(BaseModel):
    X: list[list[float]]
    kmeans: dict
    dbscan: dict


class EvaluateRequest(BaseModel):
    X: list[list[float]]
    labels: list[int]
    exclude_noise: bool = True
    metric: str = "euclidean"


class StepCreateRequest(KMeansRequest):
    pass


class StepRequest(BaseModel):
    experiment_id: str
    action: Literal["initialize", "assign", "update", "finish"]


class QuizAnswers(BaseModel):
    answers: list[int | None]

    @field_validator("answers")
    @classmethod
    def validate_length(cls, value: list[int | None]) -> list[int | None]:
        if len(value) != quiz.QUIZ_LENGTH:
            raise ValueError(f"Exactly {quiz.QUIZ_LENGTH} answers are required.")
        return value


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/datasets")
def dataset_options() -> dict[str, list[str]]:
    return {"datasets": datasets.DATASET_NAMES}


@app.post("/api/datasets/generate")
def generate_dataset(request: DatasetRequest) -> dict:
    try:
        X, y_true, params = datasets.generate_dataset(
            request.name, request.samples, request.noise, request.groups, request.seed
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "X": X.tolist(),
        "true_labels": y_true.tolist(),
        "params": params,
        "name": request.name,
    }


def _silhouette_dict(report) -> dict:
    return {
        "score": _safe_float(report.score),
        "available": report.available,
        "reason": report.reason,
        "n_evaluated": report.n_evaluated,
        "n_clusters": report.n_clusters,
        "n_noise": report.n_noise,
        "excluded_noise": report.excluded_noise,
        "metric": report.metric,
    }


@app.post("/api/kmeans/run")
def run_kmeans_endpoint(request: KMeansRequest) -> dict:
    X = _matrix(request.X)
    try:
        result = run_kmeans(X, request.k, request.init, request.max_iter, request.seed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    silhouette = compute_silhouette(X, result.labels, exclude_noise=False)
    return {
        "labels": result.labels.tolist(),
        "centroids": result.centroids.tolist(),
        "inertia": result.inertia,
        "n_iter": result.n_iter,
        "n_clusters_obtained": result.n_clusters_obtained,
        "converged": result.converged,
        "silhouette": _silhouette_dict(silhouette),
    }


@app.post("/api/dbscan/run")
def run_dbscan_endpoint(request: DBSCANRequest) -> dict:
    X = _matrix(request.X)
    try:
        result = run_dbscan(X, request.eps, request.min_samples, request.metric)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    silhouette = compute_silhouette(
        X, result.labels, exclude_noise=True, metric=request.metric
    )
    return {
        "labels": result.labels.tolist(),
        "core_mask": result.core_mask.tolist(),
        "n_clusters": result.n_clusters,
        "n_noise": result.n_noise,
        "n_core": result.n_core,
        "n_border": result.n_border,
        "silhouette": _silhouette_dict(silhouette),
    }


@app.post("/api/compare")
def compare_endpoint(request: CompareRequest) -> dict:
    X = _matrix(request.X)
    try:
        km = request.kmeans
        db = request.dbscan
        kmeans_result = run_kmeans(
            X,
            int(km.get("k", 3)),
            str(km.get("init", "k-means++")),
            int(km.get("max_iter", 300)),
            int(km.get("seed", 42)),
        )
        dbscan_result = run_dbscan(
            X,
            float(db.get("eps", 0.4)),
            int(db.get("min_samples", 5)),
            str(db.get("metric", "euclidean")),
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    km_silhouette = compute_silhouette(X, kmeans_result.labels, exclude_noise=False)
    db_silhouette = compute_silhouette(
        X, dbscan_result.labels, exclude_noise=True, metric=str(db.get("metric", "euclidean"))
    )
    return {
        "kmeans": {
            "labels": kmeans_result.labels.tolist(),
            "centroids": kmeans_result.centroids.tolist(),
            "inertia": kmeans_result.inertia,
            "n_clusters": kmeans_result.n_clusters_obtained,
            "n_iter": kmeans_result.n_iter,
            "silhouette": _silhouette_dict(km_silhouette),
        },
        "dbscan": {
            "labels": dbscan_result.labels.tolist(),
            "core_mask": dbscan_result.core_mask.tolist(),
            "n_clusters": dbscan_result.n_clusters,
            "n_noise": dbscan_result.n_noise,
            "n_core": dbscan_result.n_core,
            "n_border": dbscan_result.n_border,
            "silhouette": _silhouette_dict(db_silhouette),
        },
    }


@app.post("/api/evaluate")
def evaluate_endpoint(request: EvaluateRequest) -> dict:
    X = _matrix(request.X)
    labels = np.asarray(request.labels)
    if labels.ndim != 1 or len(labels) != len(X):
        raise HTTPException(status_code=422, detail="Labels must contain one integer per point.")
    report = compute_silhouette(
        X, labels, exclude_noise=request.exclude_noise, metric=request.metric
    )
    return _silhouette_dict(report)


_steppers: dict[str, KMeansStepper] = {}
_step_X: dict[str, np.ndarray] = {}


def _step_state(experiment_id: str, stepper: KMeansStepper) -> dict:
    return {
        "experiment_id": experiment_id,
        "phase": stepper.phase,
        "centroids": stepper.centroids.tolist(),
        "initial_centroids": stepper.initial_centroids.tolist(),
        "previous_centroids": (
            stepper.previous_centroids.tolist() if stepper.previous_centroids is not None else None
        ),
        "labels": stepper.labels.tolist() if stepper.labels is not None else None,
        "inertia": _safe_float(stepper.inertia),
        "iteration": stepper.iteration,
        "centroid_shift": _safe_float(stepper.centroid_shift),
        "converged": stepper.converged,
        "empty_cluster_fix": stepper.empty_cluster_fix,
        "summary": stepper.state_summary(),
    }


@app.post("/api/kmeans/step")
def kmeans_step_endpoint(request: StepCreateRequest | StepRequest) -> dict:
    if isinstance(request, StepRequest):
        experiment_id = request.experiment_id
        stepper = _steppers.get(experiment_id)
        if stepper is None:
            raise HTTPException(status_code=404, detail="Step-by-step experiment was not found.")
        if request.action == "assign":
            stepper.assign_step()
        elif request.action == "update":
            stepper.update_step()
        elif request.action == "finish":
            stepper.run_to_convergence()
        return _step_state(experiment_id, stepper)

    X = _matrix(request.X)
    try:
        stepper = KMeansStepper(X, request.k, request.init, request.max_iter, request.seed, tol=1e-6)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    experiment_id = uuid.uuid4().hex
    _steppers[experiment_id] = stepper
    _step_X[experiment_id] = X
    return _step_state(experiment_id, stepper)


@app.get("/api/quiz")
def quiz_endpoint() -> dict:
    return {
        "questions": [
            {"question": item.question, "options": list(item.options)}
            for item in quiz.QUESTIONS
        ]
    }


@app.post("/api/quiz/submit")
def quiz_submit_endpoint(request: QuizAnswers) -> dict:
    try:
        score, flags = quiz.score_quiz(request.answers)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    percentage = round(score / quiz.QUIZ_LENGTH * 100, 1)
    return {
        "score": score,
        "total": quiz.QUIZ_LENGTH,
        "percentage": percentage,
        "grade": quiz.grade_for_score(score),
        "flags": flags,
        "review": [
            {
                "question": item.question,
                "options": list(item.options),
                "answer_index": item.answer_index,
                "explanation": item.explanation,
            }
            for item in quiz.QUESTIONS
        ],
    }
