# 🔬 Clustering Visualizer – Machine Learning Virtual Lab

An interactive web-based virtual laboratory for learning, implementing, visualizing, and comparing **K-Means and DBSCAN clustering algorithms**.

Built using **React, FastAPI, Python, Scikit-learn, and Plotly.js**, this project provides a hands-on environment to explore unsupervised machine learning through interactive simulations, customizable datasets, performance metrics, and quiz-based assessments.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [Features](#-features)
- [Algorithms Implemented](#-algorithms-implemented)
- [Available Datasets](#-available-datasets)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [API Endpoints](#-api-endpoints)
- [Evaluation Metrics](#-evaluation-metrics)
- [Testing](#-testing)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)

---

## 📖 Overview

Clustering is an important technique in unsupervised machine learning used to group similar data points without predefined labels.

Understanding clustering algorithms theoretically can be challenging, particularly when learning how parameters, centroids, and data distributions affect the final results.

**Clustering Visualizer** addresses this challenge by providing an interactive virtual laboratory where students can:

- Generate and explore different datasets.
- Execute K-Means and DBSCAN clustering algorithms.
- Observe clustering results through interactive visualizations.
- Understand the internal working of K-Means step by step.
- Experiment with different algorithm parameters.
- Compare the performance and behavior of both algorithms.
- Evaluate clustering quality using the Silhouette Score.
- Test their understanding through an interactive quiz.

The application combines machine learning computation with an intuitive educational interface.

## 🎯 Objectives

1. Understand the fundamentals of unsupervised learning and clustering.
2. Study the working principles of K-Means and DBSCAN algorithms.
3. Visualize cluster formation and understand the effect of algorithm parameters.
4. Explore the differences between partitioning-based and density-based clustering.
5. Evaluate clustering quality using appropriate performance metrics.
6. Provide an interactive platform for practical machine learning experimentation.

## ✨ Features

### 1. Five-Section Virtual Laboratory

The application follows a structured academic laboratory format:

| Section | Description |
|---|---|
| Aim | Presents the experiment objective and learning objectives. |
| Theory | Explains clustering concepts, algorithms, and evaluation metrics. |
| Procedure | Provides step-by-step experimental instructions. |
| Simulation | Allows interactive execution and visualization of clustering algorithms. |
| Quiz-Based Assessment | Evaluates understanding through multiple-choice questions. |

### 2. Interactive Dataset Generation

Generate reproducible synthetic datasets with configurable parameters.

- Select different dataset types.
- Configure the number of samples.
- Adjust noise and cluster spread.
- Set the number of groups where applicable.
- Specify a random seed for reproducible experiments.

### 3. K-Means Clustering

- Execute actual Scikit-learn K-Means clustering.
- Configure the number of clusters (K).
- Select centroid initialization methods.
- Set maximum iterations.
- Visualize cluster assignments and centroids.
- View inertia, iteration count, and Silhouette Score.

**Step-by-Step K-Means Simulation**

An additional custom implementation of Lloyd's algorithm allows students to observe:

1. Centroid initialization.
2. Assignment of data points to the nearest centroid.
3. Centroid recalculation.
4. Convergence checking.
5. Final cluster formation.

### 4. DBSCAN Clustering

Explore density-based clustering through configurable parameters.

- Adjust epsilon (`eps`).
- Configure minimum samples (`MinPts`).
- Select distance metrics: Euclidean, Manhattan, and Cosine.
- Identify core points, border points, and noise points.
- Visualize clusters and outliers.
- Evaluate clustering results using the Silhouette Score.

### 5. K-Means vs DBSCAN Comparison

Compare both algorithms using the same generated dataset.

- Side-by-side clustering visualizations.
- Comparison of cluster assignments.
- Evaluation metrics.
- Observation of differences in cluster formation.
- Understanding how each algorithm handles noise and irregularly shaped clusters.

### 6. Clustering Evaluation

Evaluate clustering quality using the Silhouette Score with appropriate handling of invalid scoring conditions.

### 7. Interactive Quiz Assessment

- Ten multiple-choice questions.
- Four options per question.
- Automatic score calculation.
- Percentage and grade calculation.
- Correct-answer review.
- Explanations for submitted answers.

---

## 🧠 Algorithms Implemented

### K-Means Clustering

K-Means is a partitioning-based unsupervised learning algorithm that divides data points into K clusters.

**Working:**

1. Select K initial centroids.
2. Assign each data point to its nearest centroid.
3. Recalculate centroids using the mean of assigned points.
4. Repeat assignment and update steps until convergence or the maximum iteration limit is reached.

**Key characteristics:**
- Requires the number of clusters beforehand.
- Uses centroids to represent clusters.
- Minimizes within-cluster sum of squared distances (inertia).
- Sensitive to initialization and outliers.

### DBSCAN Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups data points based on their density.

**Key parameters:**

| Parameter | Description |
|---|---|
| Epsilon (`eps`) | Radius of the neighbourhood around a point. |
| MinPts (`min_samples`) | Minimum number of samples required to form a dense neighbourhood. |

**Point classifications:**

- **Core Point:** Has at least MinPts samples within its epsilon neighbourhood, including itself.
- **Border Point:** Lies within the neighbourhood of a core point but does not independently satisfy the density requirement.
- **Noise Point:** Does not belong to any discovered cluster.

**Key characteristics:**
- Does not require the number of clusters beforehand.
- Can identify irregularly shaped clusters.
- Can detect noise and outliers.
- Sensitive to parameter selection and varying data densities.

### Algorithm Comparison

| Parameter | K-Means | DBSCAN |
|---|---|---|
| Approach | Partitioning-based | Density-based |
| Number of clusters | Required beforehand | Determined automatically |
| Cluster representation | Centroids | Dense regions |
| Noise detection | No explicit noise class | Yes |
| Cluster shapes | Best suited to compact clusters | Supports irregular shapes |
| Main parameters | K, initialization | Epsilon, MinPts |

---

## 📊 Available Datasets

The application supports five synthetic dataset configurations.

| Dataset | Description |
|---|---|
| Blobs | Well-separated, approximately spherical clusters. |
| Moons | Two interleaving crescent-shaped clusters. |
| Circles | Concentric circular clusters. |
| Noisy Blobs | Blob-shaped clusters with additional random noise. |
| Custom Random | Configurable randomly generated blob clusters. |

These datasets help demonstrate how different clustering algorithms behave with different data distributions.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Frontend | React, JavaScript, HTML, CSS |
| Build Tool | Vite |
| Visualization | Plotly.js |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn |
| Data Processing | NumPy, Pandas |
| API Validation | Pydantic |
| Testing | Python unittest |

---

## 🏗️ Project Architecture

The application follows a frontend-backend architecture.

```text
              ┌──────────────────────┐
              │      React UI        │
              │                      │
              │  Dataset Selection   │
              │  Algorithm Controls  │
              │  Interactive Charts  │
              │  Quiz Assessment     │
              └──────────┬───────────┘
                         │
                    REST API
                         │
              ┌──────────▼───────────┐
              │     FastAPI          │
              │                      │
              │  Request Validation  │
              │  API Endpoints       │
              │  Response Handling   │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │   Python ML Modules  │
              │                      │
              │  Dataset Generation  │
              │  K-Means              │
              │  DBSCAN               │
              │  Evaluation           │
              │  Quiz Management      │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  JSON Responses      │
              │  Clusters & Metrics  │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │ Plotly Visualization │
              └──────────────────────┘
```

### How It Works

1. The user selects a dataset and configures experiment parameters through the React interface.
2. React sends an HTTP request to the FastAPI backend.
3. FastAPI validates the request using Pydantic.
4. Python modules generate datasets and execute the selected clustering algorithm.
5. The evaluation module calculates clustering metrics.
6. The backend returns the results in JSON format.
7. React processes the response and displays interactive visualizations using Plotly.js.

---

## 📁 Project Structure

```text
ML_clustering_vlab/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── modules/
│
├── frontend/
│   ├── package.json
│   └── src/
│
├── modules/
│   ├── datasets.py
│   ├── kmeans_lab.py
│   ├── dbscan_lab.py
│   ├── evaluation.py
│   └── quiz.py
│
├── tests/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

### Module Responsibilities

| File | Responsibility |
|---|---|
| `backend/main.py` | FastAPI application, API endpoints, request validation, and response handling. |
| `modules/datasets.py` | Synthetic dataset generation. |
| `modules/kmeans_lab.py` | Standard and step-by-step K-Means implementations. |
| `modules/dbscan_lab.py` | DBSCAN execution and point classification. |
| `modules/evaluation.py` | Silhouette Score calculation and validation. |
| `modules/quiz.py` | Quiz questions, answer validation, scoring, and explanations. |
| `frontend/src/` | React components, API integration, interactive charts, and UI. |
| `tests/` | Backend and API regression tests. |

---

## 🚀 Installation and Setup

### Prerequisites

Make sure the following are installed:

- Python 3.11 or newer
- Node.js 18 or newer
- npm
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/pranavkule/ML_clustering_vlab.git

cd ML_clustering_vlab
```

### Step 2: Set Up the Backend

Create a Python virtual environment:

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies from the project root:

```bash
python -m pip install -r requirements.txt

python -m pip install -r backend/requirements.txt
```

### Step 3: Start the Backend Server

```bash
cd backend

python -m uvicorn main:app --reload --port 8000
```

Backend URL:

http://localhost:8000

Interactive API documentation:

http://localhost:8000/docs

### Step 4: Set Up the Frontend

Open a new terminal from the project root:

```bash
cd frontend

npm install
```

### Step 5: Start the Frontend Server

```bash
npm run dev
```

Frontend URL:

http://localhost:5173

**Note:** The Vite development server is configured to proxy `/api` requests to the FastAPI backend running on port 8000. Keep both servers running while using the application.

---

## 🔗 API Endpoints

The backend exposes REST API endpoints for dataset generation, clustering, evaluation, comparison, and quiz assessment.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Check backend availability. |
| GET | `/api/datasets` | Retrieve available dataset configurations. |
| POST | `/api/datasets/generate` | Generate a synthetic dataset. |
| POST | `/api/kmeans/run` | Execute K-Means clustering. |
| POST | `/api/dbscan/run` | Execute DBSCAN clustering. |
| POST | `/api/compare` | Compare K-Means and DBSCAN. |
| POST | `/api/evaluate` | Calculate clustering evaluation metrics. |
| POST | `/api/kmeans/step` | Perform step-by-step K-Means operations. |
| GET | `/api/quiz` | Retrieve quiz questions. |
| POST | `/api/quiz/submit` | Submit and evaluate quiz answers. |

For request and response details, visit the interactive FastAPI documentation at:

http://localhost:8000/docs

---

## 📈 Evaluation Metrics

### Silhouette Score

The application uses the Silhouette Score to measure how well-separated and cohesive the generated clusters are.

The score is calculated as:

\[
S(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
\]

Where:

- `a(i)` is the average distance between a data point and other points in its own cluster.
- `b(i)` is the minimum average distance between that point and points in another cluster.

**Interpretation:**

| Score | Meaning |
|---|---|
| Close to +1 | Well-separated clusters. |
| Close to 0 | Points are near cluster boundaries. |
| Negative | Potentially incorrect cluster assignments. |

**Special handling:**

- The score is unavailable when fewer than two valid clusters exist.
- All-noise DBSCAN results are handled separately.
- DBSCAN noise points are excluded from Silhouette Score calculation, and the excluded count is reported.
- Scores calculated on different sets of points should not be treated as directly comparable.

---

## 🧪 Testing

The project includes Python compilation checks and backend/API regression tests.

Run the following commands from the project root.

### Python Compilation Check

```bash
python -m compileall modules backend tests
```

### Run Backend Tests

```bash
python -m unittest discover -s tests -v
```

### Frontend Production Build

```bash
cd frontend

npm run build
```

---

## ⚠️ Limitations

- Currently supports synthetic two-dimensional datasets for direct visualization.
- Does not currently support custom CSV dataset uploads.
- Step-by-step K-Means uses a single initialization, whereas standard Scikit-learn K-Means uses multiple initializations. Results may differ.
- K-Means and DBSCAN are sensitive to parameter selection and data characteristics.
- DBSCAN may struggle with datasets containing significantly different density levels.
- Silhouette Scores calculated on different evaluated point sets are not directly comparable.
- Experiment results are not permanently stored in a database or file.

---

## 🔮 Future Enhancements

The following improvements can extend the capabilities of the virtual laboratory:

- [ ] Support custom CSV dataset uploads.
- [ ] Add an Elbow Method visualization for selecting the optimal K.
- [ ] Implement a K-distance graph for DBSCAN epsilon selection.
- [ ] Add additional evaluation metrics such as the Davies-Bouldin Index.
- [ ] Enable experiment result and visualization exports.
- [ ] Support higher-dimensional datasets using PCA.
- [ ] Introduce guided exercises and instructor-oriented features.

---

## 🎓 Learning Outcomes

By using this virtual laboratory, students can:

- Understand the practical implementation of unsupervised learning algorithms.
- Visualize the internal working of clustering techniques.
- Analyze the influence of parameters on clustering outcomes.
- Differentiate between partitioning-based and density-based approaches.
- Interpret clustering evaluation metrics.
- Gain practical experience with machine learning and full-stack application development.

---

## 👨‍💻 Author

**Pranav Kule**

GitHub: [@pranavkule](https://github.com/pranavkule)

---

**If you find this project useful, consider giving the repository a ⭐ on GitHub!**
