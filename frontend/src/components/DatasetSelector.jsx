export default function DatasetSelector({ settings, onChange, onGenerate, loading }) {
  const datasets = ['Blobs', 'Moons', 'Circles', 'Noisy blobs', 'Custom random']
  const set = (key, value) => onChange({ ...settings, [key]: value })
  const hasGroups = !['Moons', 'Circles'].includes(settings.name)
  return (
    <section className="control-section">
      <h2>Dataset</h2>
      <div className="control-grid">
        <label>Dataset
          <select value={settings.name} onChange={(e) => set('name', e.target.value)}>
            {datasets.map((name) => <option key={name}>{name}</option>)}
          </select>
        </label>
        <label>Samples
          <input type="number" min="2" max="5000" value={settings.samples} onChange={(e) => set('samples', Number(e.target.value))} />
        </label>
        <label>{['Moons', 'Circles'].includes(settings.name) ? 'Noise' : 'Cluster spread'}
          <input type="number" min="0" max="4" step="0.01" value={settings.noise} onChange={(e) => set('noise', Number(e.target.value))} />
        </label>
        {hasGroups && <label>Generated groups
          <input type="number" min="1" max="8" value={settings.groups} onChange={(e) => set('groups', Number(e.target.value))} />
        </label>}
        <label>Random seed
          <input type="number" min="0" value={settings.seed} onChange={(e) => set('seed', Number(e.target.value))} />
        </label>
      </div>
      <button className="primary" type="button" disabled={loading} onClick={onGenerate}>
        {loading ? 'Generating...' : 'Generate Dataset'}
      </button>
      <p className="helper">Dataset settings take effect only after Generate Dataset is clicked.</p>
    </section>
  )
}
