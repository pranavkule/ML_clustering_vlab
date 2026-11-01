import unittest

import numpy as np

from modules.datasets import generate_dataset
from modules.dbscan_lab import run_dbscan
from modules.evaluation import compute_silhouette
from modules.kmeans_lab import KMeansStepper, run_kmeans
from modules.quiz import QUESTIONS, QUIZ_LENGTH, grade_for_score, score_quiz, validate_quiz


class DatasetTests(unittest.TestCase):
    def test_supported_datasets_are_deterministic_and_two_dimensional(self):
        for name in ("Blobs", "Moons", "Circles"):
            with self.subTest(name=name):
                first = generate_dataset(name, n_samples=120, noise=0.08, n_groups=3, random_state=7)
                second = generate_dataset(name, n_samples=120, noise=0.08, n_groups=3, random_state=7)
                self.assertEqual(first[0].shape, (120, 2))
                np.testing.assert_array_equal(first[0], second[0])
                np.testing.assert_array_equal(first[1], second[1])


class KMeansTests(unittest.TestCase):
    def setUp(self):
        self.X, _, _ = generate_dataset("Blobs", n_samples=180, noise=0.7, n_groups=3, random_state=11)

    def test_runs_on_supported_datasets_with_multiple_settings(self):
        for name, n_groups in (("Blobs", 3), ("Moons", 2), ("Circles", 2)):
            X, _, _ = generate_dataset(name, n_samples=180, noise=0.08, n_groups=n_groups, random_state=11)
            for k in (2, 3):
                if k > len(X):
                    continue
                for seed in (0, 29):
                    with self.subTest(dataset=name, k=k, seed=seed):
                        result = run_kmeans(X, n_clusters=k, max_iter=300, random_state=seed)
                        self.assertEqual(result.labels.shape, (len(X),))
                        self.assertEqual(result.centroids.shape[1], 2)
                        self.assertGreaterEqual(result.inertia, 0.0)
                        self.assertGreaterEqual(result.n_iter, 1)

    def test_stepper_assignment_update_and_convergence(self):
        stepper = KMeansStepper(self.X, n_clusters=3, max_iter=100, random_state=11, tol=1e-8)
        self.assertEqual(stepper.phase, "assign")
        initial = stepper.centroids.copy()
        stepper.run_to_convergence()
        self.assertTrue(stepper.converged)
        self.assertIsNotNone(stepper.labels)
        self.assertEqual(set(np.unique(stepper.labels)), {0, 1, 2})
        self.assertFalse(np.array_equal(initial, stepper.centroids))
        self.assertGreaterEqual(stepper.iteration, 1)


class DBSCANTests(unittest.TestCase):
    def test_core_border_noise_and_cluster_counts(self):
        X = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [0.1, 0.1],
                [0.2, 0.0],
                [4.0, 4.0],
            ],
            dtype=float,
        )
        result = run_dbscan(X, eps=0.25, min_samples=3)
        self.assertIn(-1, set(result.labels))
        self.assertGreaterEqual(result.n_clusters, 1)
        self.assertEqual(result.n_core + result.n_border + result.n_noise, len(X))
        cluster_labels = set(result.labels) - {-1}
        self.assertEqual(result.n_clusters, len(cluster_labels))
        self.assertTrue(np.any(result.core_mask))

    def test_supported_shapes(self):
        for name, noise in (("Blobs", 0.7), ("Moons", 0.08), ("Circles", 0.08)):
            X, _, _ = generate_dataset(name, n_samples=180, noise=noise, n_groups=3, random_state=11)
            for eps in (0.15, 0.35):
                with self.subTest(dataset=name, eps=eps):
                    result = run_dbscan(X, eps=eps, min_samples=5)
                    self.assertEqual(len(result.labels), len(X))
                    self.assertEqual(
                        int(np.sum(result.labels == -1)),
                        result.n_noise,
                    )


class SilhouetteTests(unittest.TestCase):
    def test_single_cluster_and_all_noise(self):
        X = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        report = compute_silhouette(X, np.zeros(3, dtype=int))
        self.assertFalse(report.available)
        self.assertIn("one cluster", report.reason)

        report = compute_silhouette(X, np.full(3, -1, dtype=int), exclude_noise=True)
        self.assertFalse(report.available)
        self.assertIn("every point", report.reason.lower())
        self.assertEqual(report.n_noise, 3)

    def test_every_point_separate_cluster(self):
        X = np.arange(8, dtype=float).reshape(4, 2)
        report = compute_silhouette(X, np.arange(4))
        self.assertFalse(report.available)
        self.assertIn("as many clusters as evaluated points", report.reason)

    def test_dbscan_noise_exclusion(self):
        X = np.array([[0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [0.1, 0.1], [5.0, 5.0]])
        labels = np.array([0, 0, 0, 0, -1])
        report = compute_silhouette(X, labels, exclude_noise=True)
        self.assertFalse(report.available)
        self.assertTrue(report.excluded_noise)
        self.assertEqual(report.n_noise, 1)
        self.assertEqual(report.n_evaluated, 4)

    def test_valid_report_comes_from_real_labels(self):
        X = np.array([[0.0, 0.0], [0.0, 0.1], [4.0, 4.0], [4.0, 4.1]])
        report = compute_silhouette(X, np.array([0, 0, 1, 1]))
        self.assertTrue(report.available)
        self.assertIsInstance(report.score, float)
        self.assertGreaterEqual(report.score, -1.0)
        self.assertLessEqual(report.score, 1.0)


class QuizTests(unittest.TestCase):
    def test_quiz_integrity(self):
        validate_quiz()
        self.assertEqual(QUIZ_LENGTH, 10)
        for question in QUESTIONS:
            self.assertEqual(len(question.options), 4)

    def test_scoring_and_grade(self):
        answers = [question.answer_index for question in QUESTIONS]
        score, flags = score_quiz(answers)
        self.assertEqual(score, 10)
        self.assertTrue(all(flags))
        self.assertEqual(grade_for_score(10), "Excellent")

        score, flags = score_quiz([0] * QUIZ_LENGTH)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, QUIZ_LENGTH)
        self.assertEqual(len(flags), QUIZ_LENGTH)


if __name__ == "__main__":
    unittest.main()
