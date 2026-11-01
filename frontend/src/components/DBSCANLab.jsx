export default function DBSCANLab({ value, onChange }) {
  return (
    <section className="control-section">
      <h2>DBSCAN parameters</h2>
      <div className="control-grid">
        <label>eps<input type="number" min="0.01" max="100" step="0.01" value={value.eps} onChange={(e) => onChange({ ...value, eps: Number(e.target.value) })} /></label>
        <label>MinPts<input type="number" min="1" max="5000" value={value.minSamples} onChange={(e) => onChange({ ...value, minSamples: Number(e.target.value) })} /></label>
        <label>Distance metric<select value={value.metric} onChange={(e) => onChange({ ...value, metric: e.target.value })}><option>euclidean</option><option>manhattan</option><option>cosine</option></select></label>
      </div>
    </section>
  )
}
