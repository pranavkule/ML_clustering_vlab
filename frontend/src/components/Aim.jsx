export default function Aim() {
  return (
    <>
      <p className="kicker">Module 4 · Unsupervised Learning</p>
      <h1>Clustering Visualizer</h1>
      <p className="subtitle">An Interactive Laboratory for Partitioning and Density-Based Algorithms</p>
      <div className="rule" />
      <div className="aim-box">
        <strong>Objective</strong>
        <p>Understand, implement, visualize, and compare K-Means and DBSCAN using actual machine-learning computations.</p>
      </div>
      <h2>Introduction</h2>
      <p>Clustering groups observations using feature similarity without predefined target labels. It is widely used for exploration, segmentation, pattern discovery, anomaly detection, and organizing unlabelled data.</p>
      <h2>Learning objectives</h2>
      <ol>
        <li>Explain the principles of unsupervised learning.</li>
        <li>Describe K-Means initialization, assignment, update, and convergence.</li>
        <li>Interpret eps, MinPts, core points, border points, and noise points in DBSCAN.</li>
        <li>Evaluate clusters using the Silhouette Score.</li>
        <li>Compare centroid-based and density-based clustering.</li>
      </ol>
      <h2>Expected outcomes</h2>
      <p>Students should be able to select suitable parameters, interpret actual result metrics, explain invalid Silhouette configurations, and relate cluster structure to algorithm assumptions.</p>
    </>
  )
}
