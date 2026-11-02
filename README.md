# Clustering Visualizer

**An Interactive Laboratory for Partitioning and Density-Based Algorithms**

A React and FastAPI virtual laboratory for K-Means and DBSCAN clustering. The backend preserves the original real Scikit-learn implementations; the React interface uses Plotly.js for interactive two-dimensional visualizations.

## Aim

Understand, implement, visualize, and compare K-Means and DBSCAN clustering using reproducible datasets and the Silhouette Score.

## Learning Objectives

1. Understand unsupervised learning and clustering.
2. Explain K-Means initialization, assignment, centroid updates, and convergence.
3. Understand DBSCAN's eps, MinPts, core, border, and noise-point concepts.
4. Explore parameter effects.
5. Compare partitioning and density-based methods.
6. Interpret the Silhouette Score and its validity conditions.

## Features

- Five-section academic virtual lab: Aim, Theory, Procedure, Simulation, and Quiz Based Assessment.
- Deterministic synthetic datasets: Blobs, Moons, Circles, Noisy blobs, and a configurable random dataset.
- Real K-Means clustering with cluster assignments, centroids, inertia, iteration count, and Silhouette Score.
- An explicit step-by-step K-Means mode for observing centroid initialization, point assignment, centroid updates, and convergence.
- Real DBSCAN clustering with core points, border points, noise points, cluster counts, and Silhouette Score.
- Correct handling of invalid Silhouette Score configurations, including single-cluster and all-noise results.
- Side-by-side K-Means vs. DBSCAN comparison on identical data.
- Data-driven experimental observations based on the fitted results.
- Ten-question interactive assessment with scoring, grade, answer review, and explanations.
- Session-state preservation of datasets, parameters, results, step-by-step state, and quiz progress.

## Technology Stack

- React and Vite
- JavaScript, HTML, and CSS
- Plotly.js
- Python and FastAPI
- NumPy, Pandas, and Scikit-learn

## Project Structure

```text
clustering-visualizer/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── modules/
├── frontend/
│   ├── package.json
│   └── src/
├── modules/
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

### Module Responsibilities

- `backend/main.py`: FastAPI application, Pydantic requests, REST endpoints, and JSON serialization.
- `frontend/src`: React components, API service, experiment hook, Plotly charts, and academic CSS.
- `modules/datasets.py`: deterministic two-feature synthetic dataset generation.
- `modules/kmeans_lab.py`: Scikit-learn K-Means execution and the custom step-by-step Lloyd implementation.
- `modules/dbscan_lab.py`: Scikit-learn DBSCAN execution and core/border/noise identification.
- `modules/evaluation.py`: Silhouette Score calculation with explicit invalid-state handling.
- `modules/quiz.py`: question bank, answer key, scoring, grade calculation, and explanations.

## Installation

Python 3.11 or newer and Node.js 18 or newer are recommended.

### Backend

From the project root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

FastAPI is available at `http://localhost:8000`, with interactive API documentation at `http://localhost:8000/docs`.

### Frontend

In a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

Open `http://localhost:5173`.

The Vite development server proxies `/api` requests to FastAPI at port 8000.

Set `VITE_API_BASE_URL` to the deployed Render API URL, for example:

```text
VITE_API_BASE_URL=https://your-service.onrender.com
```

Do not add a trailing slash. A local `.env.local` is ignored by Git.

## Production Deployment

### Render backend

The repository root includes `render.yaml`. Render should use the repository root and Python runtime. The configured build and start commands are:

```text
Build command: python -m pip install -r backend/requirements.txt
Start command: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
Health check: /api/health
```

Configure this backend environment variable after obtaining the deployed Vercel origin:

```text
FRONTEND_ORIGIN=https://your-project.vercel.app
```

Multiple trusted origins can be comma-separated. Trailing slashes are accepted and normalized.

### Vercel frontend

Create the Vercel project with:

```text
Root directory: frontend
Framework preset: Vite
Build command: npm run build
Output directory: dist
```

Add the following production environment variable:

```text
VITE_API_BASE_URL=https://your-render-service.onrender.com
```

The frontend configuration file [frontend/vercel.json](frontend/vercel.json) routes refreshes and direct navigation to the React application.

## Navigation Sections

1. **Aim** — presents the laboratory objective, introduction, and learning objectives.
2. **Theory** — explains unsupervised learning, K-Means, DBSCAN, point types, the comparison between the algorithms, and the Silhouette Score.
3. **Procedure** — gives a general experiment workflow, separate K-Means and DBSCAN procedures, and an observation template.
4. **Simulation** — provides the interactive dataset generation, clustering experiments, step-by-step K-Means mode, DBSCAN point-type visualization, comparison mode, metrics, and observations.
5. **Quiz Based Assessment** — contains ten multiple-choice questions with immediate scoring and explanation review after submission.

## K-Means and DBSCAN in Brief

**K-Means** is a partitioning algorithm. It represents each of K clusters by a centroid, assigns every point to the nearest centroid, and repeatedly moves each centroid to the mean of its assigned points. It minimizes the within-cluster sum of squared distances, also called inertia. K-Means is fast and easy to interpret, but K must be selected in advance, the clusters are assumed to be reasonably compact, and every point must be assigned to a cluster.

**DBSCAN** is a density-based algorithm. It uses a neighbourhood radius `eps` and a minimum density `MinPts`. A point is a core point when its eps-neighbourhood contains at least MinPts samples. A border point is within eps of a core point but is not itself a core point. A noise point is neither. DBSCAN discovers the number of clusters automatically, can find irregularly shaped clusters, and can label outliers as noise. Its results depend strongly on the density structure of the data and on the chosen `eps` and `MinPts` values. DBSCAN does not compute cluster centroids as part of its standard algorithm.

## Conducting the Experiments

1. Open **Simulation** and choose a dataset type.
2. Set the number of samples, noise or cluster spread, number of groups where applicable, and a random seed.
3. Click **Generate Dataset** to create a deterministic dataset from those settings.
4. Select **K-Means** and configure K, initialization, maximum iterations, and seed.
5. Click **Run Clustering** and inspect the original dataset, the cluster assignments, the centroids, the inertia, and the Silhouette Score.
6. Open the **Step-by-step K-Means** section to initialize centroids and advance through assignment and update operations.
7. Select **DBSCAN**, configure eps, MinPts, and the distance metric, and run the experiment.
8. Inspect core points, border points, and noise points. Note that the number of clusters excludes noise.
9. Change one parameter at a time and rerun. The interface marks results that do not match the current controls.
10. Enable **Comparison mode** to run K-Means and DBSCAN on exactly the same data and review the side-by-side plots and comparison table.

## Silhouette Score

The Silhouette Score compares how close a point is to its own cluster with how close it is to the nearest other cluster:

```text
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

- `a(i)` is the mean distance from point i to the other points in its own cluster.
- `b(i)` is the smallest mean distance from point i to the points of another cluster.
- Values close to +1 generally indicate compact, well-separated clusters.
- Values near 0 indicate points near cluster boundaries.
- Negative values can indicate potentially inappropriate assignments.

The score is not defined for every result. It requires at least two clusters and fewer clusters than evaluated samples. The application reports the score as unavailable, together with a reason, for single-cluster results, all-noise DBSCAN results, and results in which every point forms its own cluster.

For DBSCAN, the application excludes noise points from the Silhouette Score and reports how many points were excluded. K-Means uses all points. Because the two scores may be computed on different point sets, they should not be treated as a direct like-for-like ranking. The Silhouette Score is also a diagnostic rather than proof that a clustering is correct; it can favour compact convex clusters.

## Testing and Verification

Python compilation and the existing plus API regression tests:

```bash
python -m compileall modules backend tests
python -m unittest discover -s tests -v
```

Frontend production build:

```bash
cd frontend
npm run build
```

## Known Limitations

- The application intentionally supports only two-feature synthetic datasets for direct visualization.
- The step-by-step K-Means mode uses one initialization, while the standard Scikit-learn result uses `n_init=10`; the two results can therefore differ.
- K-Means and DBSCAN results are sensitive to their parameters and to the scale/density of the data.
- A single DBSCAN `eps`/`MinPts` pair may not suit datasets containing very different density levels.
- Silhouette Scores are not directly comparable across different evaluated point sets.
- Experiment state is maintained by the React application and step-by-step state is held temporarily by the backend; results are not persisted to a database or file.
- The sidebar uses a fixed width on desktop and adapts for narrow screens.
- `cosine` distance is available for DBSCAN, but distance metrics should be selected with care for the dataset.

## Possible Future Enhancements

- Upload custom CSV datasets with two selected features.
- Add an elbow-method view for selecting K.
- Add a k-distance graph to support eps selection.
- Add additional internal evaluation measures such as the Davies–Bouldin index.
- Export experiment results and figures.
- Support higher-dimensional data with projection or feature selection.
- Add an instructor mode with additional guided exercises.
