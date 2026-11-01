import Plot from 'react-plotly.js'

const colors = ['#1f4e79', '#2e75b6', '#548235', '#b07d1a', '#6b4c9a', '#0f766e']

export default function ClusteringChart({ X, labels, centroids, coreMask, title, dbscan = false, stepPrevious }) {
  const traces = []
  const labelValues = labels ? [...new Set(labels.filter((v) => v >= 0))] : []
  labelValues.forEach((label) => {
    const indices = labels.map((v, i) => v === label ? i : -1).filter((i) => i >= 0)
    const coreIndices = dbscan && coreMask ? indices.filter((i) => coreMask[i]) : indices
    const borderIndices = dbscan && coreMask ? indices.filter((i) => !coreMask[i]) : []
    if (coreIndices.length) {
      traces.push({
        x: coreIndices.map((i) => X[i][0]),
        y: coreIndices.map((i) => X[i][1]),
        mode: 'markers',
        type: 'scatter',
        name: dbscan ? `Core / Cluster ${label}` : `Cluster ${label}`,
        marker: { color: colors[label % colors.length], size: 8, symbol: dbscan ? 'circle' : 'circle' },
        hovertemplate: `Cluster ${label}<br>Feature 1: %{x:.3f}<br>Feature 2: %{y:.3f}<extra></extra>`,
      })
    }
    if (borderIndices.length) {
      traces.push({
        x: borderIndices.map((i) => X[i][0]),
        y: borderIndices.map((i) => X[i][1]),
        mode: 'markers',
        type: 'scatter',
        name: `Border / Cluster ${label}`,
        marker: { color: colors[label % colors.length], size: 8, symbol: 'circle-open' },
        hovertemplate: `Border, Cluster ${label}<br>Feature 1: %{x:.3f}<br>Feature 2: %{y:.3f}<extra></extra>`,
      })
    }
  })
  if (dbscan && labels) {
    const noise = labels.map((v, i) => v === -1 ? i : -1).filter((i) => i >= 0)
    if (noise.length) {
      traces.push({
        x: noise.map((i) => X[i][0]), y: noise.map((i) => X[i][1]),
        mode: 'markers', type: 'scatter', name: 'Noise',
        marker: { color: '#111827', size: 9, symbol: 'x' },
        hovertemplate: 'Noise<br>Feature 1: %{x:.3f}<br>Feature 2: %{y:.3f}<extra></extra>',
      })
    }
  }
  if (!labels) {
    traces.push({ x: X.map((p) => p[0]), y: X.map((p) => p[1]), mode: 'markers', type: 'scatter', name: 'Data points', marker: { color: '#9ca3af', size: 8 } })
  }
  if (centroids) {
    if (stepPrevious) {
      stepPrevious.forEach((old, i) => {
        const current = centroids[i]
        traces.push({
          x: [old[0], current[0]], y: [old[1], current[1]], mode: 'lines', type: 'scatter',
          name: i === 0 ? 'Centroid movement' : undefined, showlegend: i === 0,
          line: { color: '#9ca3af', dash: 'dot' }, hoverinfo: 'skip',
        })
      })
    }
    traces.push({
      x: centroids.map((p) => p[0]), y: centroids.map((p) => p[1]), mode: 'markers', type: 'scatter',
      name: 'Centroid', marker: { color: '#111827', size: 13, symbol: 'x' },
      hovertemplate: 'Centroid<br>Feature 1: %{x:.3f}<br>Feature 2: %{y:.3f}<extra></extra>',
    })
  }
  return (
    <Plot
      data={traces}
      useResizeHandler
      style={{ width: '100%', height: '460px' }}
      layout={{
        title: { text: title, x: 0 },
        paper_bgcolor: '#fff', plot_bgcolor: '#fff',
        font: { family: 'Segoe UI, Arial, sans-serif', color: '#374151' },
        margin: { l: 55, r: 20, t: 55, b: 55 },
        legend: { orientation: 'h' },
        xaxis: { title: 'Feature 1', gridcolor: '#eef0f3' },
        yaxis: { title: 'Feature 2', gridcolor: '#eef0f3' },
        hovermode: 'closest',
      }}
      config={{ displayModeBar: false, responsive: true }}
    />
  )
}
