const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(body.detail || `Request failed with status ${response.status}`)
  }
  return body
}

export const api = {
  health: () => request('/api/health'),
  datasets: () => request('/api/datasets'),
  generateDataset: (payload) => request('/api/datasets/generate', { method: 'POST', body: JSON.stringify(payload) }),
  runKMeans: (payload) => request('/api/kmeans/run', { method: 'POST', body: JSON.stringify(payload) }),
  runDBSCAN: (payload) => request('/api/dbscan/run', { method: 'POST', body: JSON.stringify(payload) }),
  compare: (payload) => request('/api/compare', { method: 'POST', body: JSON.stringify(payload) }),
  evaluate: (payload) => request('/api/evaluate', { method: 'POST', body: JSON.stringify(payload) }),
  kmeansStep: (payload) => request('/api/kmeans/step', { method: 'POST', body: JSON.stringify(payload) }),
  quiz: () => request('/api/quiz'),
  submitQuiz: (answers) => request('/api/quiz/submit', { method: 'POST', body: JSON.stringify({ answers }) }),
}
