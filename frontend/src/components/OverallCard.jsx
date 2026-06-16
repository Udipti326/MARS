function OverallCard({ overall }) {
  return (
    <div className="card overall-card">
      <h2>Overall Result</h2>

      <div className="claim-grid">
        <div>
          <strong>Verdict</strong>
          <p>{overall.verdict}</p>
        </div>

        <div>
          <strong>Confidence</strong>
          <p>{overall.confidence}</p>
        </div>

        <div>
          <strong>Label</strong>
          <p>{overall.label}</p>
        </div>
      </div>
    </div>
  );
}

export default OverallCard;