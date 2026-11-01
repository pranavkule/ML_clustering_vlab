export default function Procedure() {
  return (
    <>
      <h1>Procedure</h1><div className="rule" />
      <h2>K-Means experiment</h2>
      <ol><li>Generate a dataset.</li><li>Choose K, initialization, maximum iterations, and seed.</li><li>Run K-Means and inspect assignments, centroids, inertia, and Silhouette Score.</li><li>Use step-by-step mode to observe initialization, assignment, updates, and convergence.</li></ol>
      <h2>DBSCAN experiment</h2>
      <ol><li>Generate or retain the same dataset.</li><li>Select eps, MinPts, and a distance metric.</li><li>Run DBSCAN and identify core, border, and noise points.</li><li>Observe the Silhouette Score calculated only on non-noise points.</li></ol>
      <h2>Comparison and evaluation</h2>
      <ol><li>Configure both algorithms and run the comparison on the same feature matrix.</li><li>Record cluster counts, noise, inertia, and scores.</li><li>Change one parameter at a time and explain the effect using actual results.</li></ol>
      <h2>Observation table</h2>
      <div className="table-wrap"><table><thead><tr><th>Experiment</th><th>Parameters</th><th>Clusters</th><th>Noise</th><th>Metric</th><th>Observation</th></tr></thead><tbody><tr><td>K-Means</td><td>K / seed</td><td>—</td><td>0</td><td>Inertia / Silhouette</td><td>—</td></tr><tr><td>DBSCAN</td><td>eps / MinPts</td><td>—</td><td>—</td><td>Silhouette</td><td>—</td></tr></tbody></table></div>
      <h2>Conclusion guidelines</h2>
      <p>State which parameter changes altered cluster structure, whether the score supports that result, and why each algorithm behaves differently on the selected dataset. Do not treat the Silhouette Score as proof of correctness.</p>
    </>
  )
}
