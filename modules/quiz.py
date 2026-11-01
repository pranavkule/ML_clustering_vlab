"""Quiz questions, answer validation, scoring, and explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class QuizQuestion:
    question: str
    options: tuple[str, str, str, str]
    answer_index: int
    explanation: str


QUESTIONS: tuple[QuizQuestion, ...] = (
    QuizQuestion(
        question="Which statement best describes unsupervised learning?",
        options=(
            "It learns from labelled input-output pairs.",
            "It looks for structure or patterns in unlabelled data.",
            "It evaluates a model using classification accuracy.",
            "It always predicts a target variable from the input features.",
        ),
        answer_index=1,
        explanation=(
            "Unsupervised learning works with data that has no target labels. Clustering is an "
            "unsupervised task because it groups samples using only their features."
        ),
    ),
    QuizQuestion(
        question="Which quantity does K-Means attempt to minimize?",
        options=(
            "The number of noise points.",
            "The Silhouette Score.",
            "The within-cluster sum of squared distances, also called inertia.",
            "The epsilon neighbourhood radius.",
        ),
        answer_index=2,
        explanation=(
            "The K-Means objective is J = sum of squared distances from each point to its assigned "
            "centroid. This quantity is called inertia or the within-cluster sum of squares."
        ),
    ),
    QuizQuestion(
        question="In K-Means, what does the parameter K specify?",
        options=(
            "The number of features in the dataset.",
            "The maximum number of iterations.",
            "The number of clusters to form.",
            "The number of independent initializations.",
        ),
        answer_index=2,
        explanation=(
            "K is the number of centroids and therefore the number of clusters requested from "
            "K-Means. It must be chosen before the algorithm runs."
        ),
    ),
    QuizQuestion(
        question="During the K-Means update step, each centroid is recalculated as:",
        options=(
            "The mean of the points currently assigned to its cluster.",
            "The point farthest from the centroid.",
            "The median of the entire dataset.",
            "A new point chosen at random from the dataset.",
        ),
        answer_index=0,
        explanation=(
            "After assignment, each centroid moves to the coordinate-wise mean of the points in its "
            "cluster. The assignment and update steps repeat until convergence."
        ),
    ),
    QuizQuestion(
        question="What is DBSCAN primarily designed to do?",
        options=(
            "Partition the data into exactly K compact clusters.",
            "Find dense regions of points and label sparse points as noise.",
            "Predict a continuous target from labelled examples.",
            "Minimize the within-cluster sum of squares.",
        ),
        answer_index=1,
        explanation=(
            "DBSCAN is a density-based clustering algorithm. It grows clusters from dense "
            "neighbourhoods and can leave isolated points as noise rather than forcing every point "
            "into a cluster."
        ),
    ),
    QuizQuestion(
        question="In DBSCAN, what does eps (epsilon) define?",
        options=(
            "The radius of the neighbourhood around a point.",
            "The maximum number of clusters to discover.",
            "The minimum number of points in the whole dataset.",
            "The number of iterations before the algorithm stops.",
        ),
        answer_index=0,
        explanation=(
            "eps is the neighbourhood radius. For each point, DBSCAN counts how many samples lie "
            "within distance eps to estimate the local density."
        ),
    ),
    QuizQuestion(
        question="In DBSCAN, what does MinPts (min_samples) control?",
        options=(
            "The minimum number of points in an eps-neighbourhood for a point to be a core point.",
            "The minimum number of clusters that must be found.",
            "The maximum number of noise points that may be produced.",
            "The minimum distance allowed between two cluster centroids.",
        ),
        answer_index=0,
        explanation=(
            "A point is a core point when its eps-neighbourhood contains at least MinPts samples, "
            "including the point itself. Larger values generally require denser regions."
        ),
    ),
    QuizQuestion(
        question="Which statement correctly distinguishes DBSCAN core, border, and noise points?",
        options=(
            "A core point has at least MinPts points within eps; a border point is within eps of a "
            "core point but is not itself a core point; a noise point is neither.",
            "A core point has fewer than MinPts neighbours; a border point has no neighbours; noise "
            "points have the most neighbours.",
            "Every point within eps of another point is a core point.",
            "Border points are labelled -1, while noise points belong to clusters.",
        ),
        answer_index=0,
        explanation=(
            "Core points satisfy the density requirement. Border points are reachable from a core "
            "point but do not satisfy the density requirement themselves. Noise points are neither "
            "core nor border points and receive the label -1."
        ),
    ),
    QuizQuestion(
        question="A Silhouette Score close to +1 generally indicates that:",
        options=(
            "Points are well matched to their own cluster and far from other clusters.",
            "Points lie close to the boundaries between clusters.",
            "The clustering result contains only one cluster.",
            "Every point was labelled as noise.",
        ),
        answer_index=0,
        explanation=(
            "The Silhouette Score compares a point's average distance to its own cluster (a) with "
            "its smallest average distance to another cluster (b). A value near +1 indicates "
            "compact, well-separated clusters."
        ),
    ),
    QuizQuestion(
        question="Which statement best distinguishes K-Means from DBSCAN?",
        options=(
            "K-Means requires K and favours compact centroid-based clusters, while DBSCAN finds "
            "density-connected clusters and can label noise.",
            "K-Means can label noise points, while DBSCAN requires K to be chosen in advance.",
            "Both algorithms require the number of clusters to be specified before running.",
            "Both algorithms always produce identical clusters on the same dataset.",
        ),
        answer_index=0,
        explanation=(
            "K-Means is a partitioning method driven by centroids and requires K. DBSCAN is "
            "density-based, discovers the number of clusters from the data, can produce "
            "non-convex clusters, and can label outliers as noise."
        ),
    ),
)

QUIZ_LENGTH = len(QUESTIONS)


def validate_quiz() -> None:
    """Raise an informative error if the quiz data is malformed."""

    if QUIZ_LENGTH != 10:
        raise ValueError(f"The quiz must contain exactly 10 questions, found {QUIZ_LENGTH}.")
    for index, question in enumerate(QUESTIONS, start=1):
        if len(question.options) != 4:
            raise ValueError(f"Question {index} must have exactly four options.")
        if not (0 <= question.answer_index < 4):
            raise ValueError(f"Question {index} has an invalid answer key.")
        if len(set(question.options)) != 4:
            raise ValueError(f"Question {index} contains duplicate options.")


def score_quiz(answers: Sequence[int | None]) -> tuple[int, list[bool]]:
    """Score answers against the answer key.

    Returns the integer score and a correctness flag for every question.
    """

    if len(answers) != QUIZ_LENGTH:
        raise ValueError("The number of answers must match the number of questions.")

    flags: list[bool] = []
    for index, (answer, question) in enumerate(zip(answers, QUESTIONS), start=1):
        if answer is None:
            flags.append(False)
            continue
        if not isinstance(answer, (int,)) or not (0 <= int(answer) < 4):
            raise ValueError(f"Answer {index} must be an option index from 0 to 3.")
        flags.append(int(answer) == question.answer_index)

    return int(sum(flags)), flags


def grade_for_score(score: int) -> str:
    if score >= 9:
        return "Excellent"
    if score >= 7:
        return "Good"
    if score >= 5:
        return "Satisfactory"
    return "Needs Improvement"


validate_quiz()
