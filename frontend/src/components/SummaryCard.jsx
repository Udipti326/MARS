function SummaryCard({ summary }) {
  const summaryText =
    typeof summary === "string"
      ? summary
      : summary?.summary || "No summary available";

  const claims =
    typeof summary === "object"
      ? summary?.claims || []
      : [];

  return (
    <div className="card">
      <h2>Summary</h2>

      <p>{summaryText}</p>

      {claims.length > 0 && (
        <>
          <h3 style={{ marginTop: "20px" }}>Key Claims</h3>

          <ul>
            {claims.map((claim, index) => (
              <li key={index}>{claim}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

export default SummaryCard;