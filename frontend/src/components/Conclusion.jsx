
export default function Conclusion() {
  return (
    <>
      <h1>Conclusion</h1>
      <div className="rule" />

      <h2>Experiment Summary</h2>
      <p>
        This experiment explored unsupervised learning
        through K-Means and DBSCAN clustering algorithms.
        Different synthetic datasets were generated and
        analyzed to understand how clustering works.
      </p>

      <h2>Key Learnings</h2>
      <ul>
        <li>Understood the fundamentals of clustering.</li>
        <li>Observed K-Means initialization, assignment
          and centroid updates.</li>
        <li>Understood DBSCAN and its density-based
          clustering mechanism.</li>
        <li>Identified core, border and noise points.</li>
        <li>Studied the effect of algorithm parameters.</li>
        <li>Evaluated clustering using the
          Silhouette Score.</li>
      </ul>

      <h2>Algorithm Comparison</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>K-Means</th>
              <th>DBSCAN</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Centroid-based</td>
              <td>Density-based</td>
            </tr>
            <tr>
              <td>Requires K</td>
              <td>Does not require K</td>
            </tr>
            <tr>
              <td>Assigns all points</td>
              <td>Identifies noise points</td>
            </tr>
            <tr>
              <td>Suitable for compact clusters</td>
              <td>Supports irregular cluster shapes</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Final Conclusion</h2>
      <p>
        Through this virtual laboratory, the working
        principles of K-Means and DBSCAN were studied
        practically. The experiments demonstrated how
        different datasets and parameter configurations
        influence cluster formation. The Silhouette
        Score provided an additional measure for
        evaluating clustering quality. Both algorithms
        have different strengths and limitations, and
        their suitability depends on the characteristics
        of the dataset.
      </p>

      <div className="aim-box">
        <strong>Learning Outcome</strong>
        <p>
          The experiment helped develop practical
          understanding of clustering algorithms,
          parameter selection, visualization and
          evaluation of unsupervised learning results.
        </p>
      </div>
    </>
  )
}
