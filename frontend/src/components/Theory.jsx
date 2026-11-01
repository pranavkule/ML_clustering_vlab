export default function Theory() {
  return (
    <>
      <h1>Theory</h1><div className="rule" />
      <h2>A. Unsupervised learning and clustering</h2>
      <p>In unsupervised learning, samples have features but no target labels. Clustering assigns similar samples to groups. Cluster identifiers are arbitrary, and validation may require internal measures or domain knowledge.</p>
      <h2>B. K-Means</h2>
      <p>K-Means selects K centroids, assigns each point to the nearest centroid, and recalculates every centroid as the mean of its assigned points. These steps repeat until assignments are stable or the iteration limit is reached.</p>
      <div className="equation">J = Σᵢ ‖xᵢ − μcᵢ‖²</div>
      <p>J is the within-cluster sum of squared Euclidean distances and is reported as inertia. K-Means converges to a local optimum and can be sensitive to initialization, scale, outliers, and K.</p>
      <h2>C. DBSCAN</h2>
      <p>DBSCAN uses an eps-neighbourhood radius and MinPts density threshold. A core point has at least MinPts points, including itself, within eps. A border point is in the neighbourhood of a core point but is not a core point. Other points are noise and receive label −1.</p>
      <p>DBSCAN discovers the number of clusters from density, can follow non-convex shapes, and does not calculate centroids.</p>
      <h2>D. Comparison and evaluation</h2>
      <p>K-Means partitions every point around compact centroids and requires K. DBSCAN can label noise and discover density-connected clusters. The Silhouette Score compares each point's within-cluster distance a(i) with its nearest-cluster distance b(i):</p>
      <div className="equation">s(i) = [b(i) − a(i)] / max(a(i), b(i))</div>
      <p>Values near +1 indicate compact, separated clusters; values near 0 indicate boundary points; negative values indicate potentially inappropriate assignments. At least two clusters and fewer clusters than samples are required. For DBSCAN, this laboratory excludes noise before evaluation.</p>
    </>
  )
}
