import { useState } from 'react'

export default function useExperiment() {
  const [dataset, setDataset] = useState(null)
  const [kmeansResult, setKmeansResult] = useState(null)
  const [dbscanResult, setDbscanResult] = useState(null)
  const [comparisonResult, setComparisonResult] = useState(null)
  const resetResults = () => {
    setKmeansResult(null)
    setDbscanResult(null)
    setComparisonResult(null)
  }
  return { dataset, setDataset, kmeansResult, setKmeansResult, dbscanResult, setDbscanResult, comparisonResult, setComparisonResult, resetResults }
}
