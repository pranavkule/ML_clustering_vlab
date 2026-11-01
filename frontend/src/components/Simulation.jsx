import { useState } from 'react'
import { api } from '../services/api.js'
import useExperiment from '../hooks/useExperiment.js'
import DatasetSelector from './DatasetSelector.jsx'
import KMeansLab from './KMeansLab.jsx'
import DBSCANLab from './DBSCANLab.jsx'
import ClusteringChart from './ClusteringChart.jsx'
import Comparison from './Comparison.jsx'
import Evaluation from './Evaluation.jsx'

export default function Simulation() {
  const experiment = useExperiment()
  const [datasetSettings, setDatasetSettings] = useState({ name: 'Blobs', samples: 320, noise: 0.9, groups: 3, seed: 42 })
  const [algorithm, setAlgorithm] = useState('K-Means')
  const [comparisonMode, setComparisonMode] = useState(false)
  const [kmeans, setKmeans] = useState({ k: 3, init: 'k-means++', maxIter: 300, seed: 42 })
  const [dbscan, setDbscan] = useState({ eps: 0.4, minSamples: 5, metric: 'euclidean' })
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  const [step, setStep] = useState(null)

  const run = async (fn) => {
    setLoading(true)
    setMessage('')
    try { await fn() } catch (error) { setMessage(error.message) } finally { setLoading(false) }
  }

  const generateDataset = () => run(async () => {
    const data = await api.generateDataset({
      name: datasetSettings.name,
      samples: datasetSettings.samples,
      noise: datasetSettings.noise,
      groups: datasetSettings.groups,
      seed: datasetSettings.seed,
    })
    experiment.setDataset(data)
    experiment.resetResults()
    setStep(null)
    setMessage(`Generated ${data.X.length} two-dimensional points.`)
  })

  const runAlgorithm = () => run(async () => {
    if (!experiment.dataset) throw new Error('Generate a dataset before running an experiment.')
    if (comparisonMode) {
      const result = await api.compare({
        X: experiment.dataset.X,
        kmeans: { k: kmeans.k, init: kmeans.init, max_iter: kmeans.maxIter, seed: kmeans.seed },
        dbscan: { eps: dbscan.eps, min_samples: dbscan.minSamples, metric: dbscan.metric },
      })
      experiment.setComparisonResult(result)
      return
    }
    if (algorithm === 'K-Means') {
      experiment.setKmeansResult(await api.runKMeans({
        X: experiment.dataset.X, k: kmeans.k, init: kmeans.init, max_iter: kmeans.maxIter, seed: kmeans.seed,
      }))
    } else {
      experiment.setDbscanResult(await api.runDBSCAN({
        X: experiment.dataset.X, eps: dbscan.eps, min_samples: dbscan.minSamples, metric: dbscan.metric,
      }))
    }
  })

  const reset = () => { experiment.resetResults(); setStep(null); setMessage('Experiment results cleared.') }

  const initializeStep = () => run(async () => {
    if (!experiment.dataset) throw new Error('Generate a dataset before initializing step-by-step mode.')
    setStep(await api.kmeansStep({
      X: experiment.dataset.X, k: kmeans.k, init: kmeans.init, max_iter: kmeans.maxIter, seed: kmeans.seed,
    }))
  })

  const stepAction = (action) => run(async () => {
    setStep(await api.kmeansStep({ experiment_id: step.experiment_id, action }))
  })

  return (
    <>
      <h1>Simulation</h1><div className="rule" />
      <DatasetSelector settings={datasetSettings} onChange={setDatasetSettings} onGenerate={generateDataset} loading={loading} />
      {!comparisonMode && (
        <div className="segmented">
          {['K-Means', 'DBSCAN'].map((item) => <button type="button" key={item} className={algorithm === item ? 'selected' : ''} onClick={() => setAlgorithm(item)}>{item}</button>)}
        </div>
      )}
      <label className="check-row"><input type="checkbox" checked={comparisonMode} onChange={(e) => setComparisonMode(e.target.checked)} /> Comparison mode: use the same dataset for both algorithms</label>
      {comparisonMode || algorithm === 'K-Means' ? <KMeansLab value={kmeans} onChange={setKmeans} /> : null}
      {comparisonMode || algorithm === 'DBSCAN' ? <DBSCANLab value={dbscan} onChange={setDbscan} /> : null}
      <div className="button-row">
        <button className="primary" type="button" disabled={loading} onClick={runAlgorithm}>{comparisonMode ? 'Run Comparison' : 'Run Clustering'}</button>
        <button type="button" onClick={reset}>Reset Experiment</button>
      </div>
      {message && <p className={message.includes('could not') || message.includes('error') ? 'notice error' : 'notice'}>{message}</p>}

      {experiment.dataset && !comparisonMode && !experiment.kmeansResult && !experiment.dbscanResult && (
        <ClusteringChart X={experiment.dataset.X} title="Generated dataset" />
      )}
      {algorithm === 'K-Means' && experiment.kmeansResult && (
        <>
          <div className="metrics">
            <div><span>Clusters</span><strong>{experiment.kmeansResult.n_clusters_obtained}</strong></div>
            <div><span>Iterations</span><strong>{experiment.kmeansResult.n_iter}</strong></div>
            <div><span>Inertia</span><strong>{experiment.kmeansResult.inertia.toFixed(2)}</strong></div>
            <div><span>Silhouette</span><strong>{experiment.kmeansResult.silhouette.available ? experiment.kmeansResult.silhouette.score.toFixed(3) : 'Unavailable'}</strong></div>
          </div>
          <ClusteringChart X={experiment.dataset.X} labels={experiment.kmeansResult.labels} centroids={experiment.kmeansResult.centroids} title="K-Means clustering result" />
          <Evaluation report={experiment.kmeansResult.silhouette} />
        </>
      )}
      {algorithm === 'DBSCAN' && experiment.dbscanResult && (
        <>
          <div className="metrics">
            <div><span>Clusters</span><strong>{experiment.dbscanResult.n_clusters}</strong></div>
            <div><span>Core points</span><strong>{experiment.dbscanResult.n_core}</strong></div>
            <div><span>Border points</span><strong>{experiment.dbscanResult.n_border}</strong></div>
            <div><span>Noise</span><strong>{experiment.dbscanResult.n_noise}</strong></div>
          </div>
          <ClusteringChart X={experiment.dataset.X} labels={experiment.dbscanResult.labels} coreMask={experiment.dbscanResult.core_mask} dbscan title="DBSCAN clustering result" />
          <Evaluation report={{ ...experiment.dbscanResult.silhouette, excluded_noise: true }} />
        </>
      )}
      {comparisonMode && <Comparison X={experiment.dataset?.X || []} result={experiment.comparisonResult} />}
      {!comparisonMode && algorithm === 'K-Means' && (
        <section className="control-section">
          <h2>Step-by-step K-Means</h2>
          {!step ? <button type="button" onClick={initializeStep}>Initialize centroids</button> : (
            <>
              <p className="helper">{step.summary}</p>
              <div className="metrics"><div><span>Iteration</span><strong>{step.iteration}</strong></div><div><span>Inertia</span><strong>{step.inertia === null ? 'Not computed' : step.inertia.toFixed(2)}</strong></div><div><span>Centroid shift</span><strong>{step.centroid_shift === null ? 'Not computed' : step.centroid_shift.toFixed(4)}</strong></div></div>
              <div className="button-row">
                {!step.converged && step.phase === 'assign' && <button className="primary" type="button" onClick={() => stepAction('assign')}>Assign points</button>}
                {!step.converged && step.phase === 'update' && <button className="primary" type="button" onClick={() => stepAction('update')}>Update centroids</button>}
                {!step.converged && <button type="button" onClick={() => stepAction('finish')}>Run to convergence</button>}
                <button type="button" onClick={initializeStep}>Reset step-by-step</button>
              </div>
              <ClusteringChart X={experiment.dataset.X} labels={step.labels} centroids={step.centroids} stepPrevious={step.previous_centroids} title={`K-Means step ${step.iteration}`} />
            </>
          )}
        </section>
      )}
    </>
  )
}
