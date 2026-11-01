export default function Evaluation({ report }) {
  if (!report) return (
    <section className="control-section"><h2>Evaluation</h2><p className="helper">Run an experiment to display its evaluated Silhouette Score.</p></section>
  )
  return (
    <section className="control-section">
      <h2>Clustering evaluation</h2>
      <div className="metrics">
        <div><span>Silhouette</span><strong>{report.available ? report.score.toFixed(3) : 'Unavailable'}</strong></div>
        <div><span>Evaluated points</span><strong>{report.n_evaluated}</strong></div>
        <div><span>Clusters</span><strong>{report.n_clusters}</strong></div>
      </div>
      {report.excluded_noise && <p>{report.n_noise} noise point(s) were excluded.</p>}
      {!report.available && <p className="notice">{report.reason}</p>}
    </section>
  )
}
