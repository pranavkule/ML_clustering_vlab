from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dataset_and_kmeans_endpoints():
    generated = client.post("/api/datasets/generate", json={
        "name": "Blobs", "samples": 60, "noise": 0.5, "groups": 3, "seed": 7,
    })
    assert generated.status_code == 200
    data = generated.json()["X"]

    result = client.post("/api/kmeans/run", json={
        "X": data, "k": 3, "init": "k-means++", "max_iter": 100, "seed": 7,
    })
    assert result.status_code == 200
    assert len(result.json()["labels"]) == 60
    assert result.json()["inertia"] >= 0


def test_dbscan_and_quiz_serialization():
    generated = client.post("/api/datasets/generate", json={
        "name": "Moons", "samples": 80, "noise": 0.08, "groups": 2, "seed": 5,
    })
    assert generated.status_code == 200
    data = generated.json()["X"]
    result = client.post("/api/dbscan/run", json={
        "X": data, "eps": 0.25, "min_samples": 5, "metric": "euclidean",
    })
    assert result.status_code == 200
    body = result.json()
    assert body["n_core"] + body["n_border"] + body["n_noise"] == 80

    quiz = client.get("/api/quiz")
    assert quiz.status_code == 200
    assert len(quiz.json()["questions"]) == 10
    for question in quiz.json()["questions"]:
        assert len(question["options"]) == 4
        assert "answer_index" not in question
