export default function KMeansLab({ value, onChange }) {
  return (
    <section className="control-section">
      <h2>K-Means parameters</h2>
      <div className="control-grid">
        <label>K<input type="number" min="1" max="20" value={value.k} onChange={(e) => onChange({ ...value, k: Number(e.target.value) })} /></label>
        <label>Initialization<select value={value.init} onChange={(e) => onChange({ ...value, init: e.target.value })}><option>k-means++</option><option>random</option></select></label>
        <label>Maximum iterations<input type="number" min="1" max="1000" value={value.maxIter} onChange={(e) => onChange({ ...value, maxIter: Number(e.target.value) })} /></label>
        <label>Random seed<input type="number" min="0" value={value.seed} onChange={(e) => onChange({ ...value, seed: Number(e.target.value) })} /></label>
      </div>
    </section>
  )
}
