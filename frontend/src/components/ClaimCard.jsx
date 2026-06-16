function ClaimCard({ claim }) {
  const claimText =
    claim?.claim ||
    claim?.claim_text ||
    claim?.text ||
    claim?.statement ||
    "";

  const verdict = claim?.judge?.verdict || claim?.verdict || "—";
  const confidenceValue =
    claim?.scores?.confidence ??
    claim?.confidence ??
    claim?.judge?.confidence ??
    null;
  const confidence =
    typeof confidenceValue === "number"
      ? confidenceValue.toFixed(3)
      : confidenceValue ?? "—";

  const label = claim?.scores?.label || claim?.label || "—";

  return (
    <div className="card">
      <h3>{claimText || "Claim"}</h3>

      <div className="claim-grid">
        <div>
          <strong>Verdict</strong>
          <p>{verdict}</p>
        </div>

        <div>
          <strong>Confidence</strong>
          <p>{confidence}</p>
        </div>

        <div>
          <strong>Label</strong>
          <p>{label}</p>
        </div>
      </div>
    </div>
  );
}

export default ClaimCard;