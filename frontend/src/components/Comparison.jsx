import ClusteringChart from './ClusteringChart.jsx'

export default function Comparison({ X, result }) {
  if (!result) return null
  return (
    <>
      <h2>Algorithm comparison</h2>
      <div className="chart-grid">
        <ClusteringChart X={X} labels={result.kmeans.labels} centroids={result.kmeans.centroids} title="K-Means" />
        <ClusteringChart X={X} labels={result.dbscan.labels} coreMask={result.dbscan.core_mask} dbscan title="DBSCAN" />
      </div>
      <div className="metrics">
        <div><span>K-Means clusters</span><strong>{result.kmeans.n_clusters}</strong></div>
        <div><span>K-Means inertia</span><strong>{result.kmeans.inertia.toFixed(2)}</strong></div>
        <div><span>K-Means silhouette</span><strong>{result.kmeans.silhouette.available ? result.kmeans.silhouette.score.toFixed(3) : 'Unavailable'}</strong></div>
        <div><span>DBSCAN clusters</span><strong>{result.dbscan.n_clusters}</strong></div>
        <div><span>DBSCAN noise</span><strong>{result.dbscan.n_noise}</strong></div>
        <div><span>DBSCAN silhouette</span><strong>{result.dbscan.silhouette.available ? result.dbscan.silhouette.score.toFixed(3) : 'Unavailable'}</strong></div>
      </div>
    </>
  )
}
