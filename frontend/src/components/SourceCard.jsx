function isValidHttpUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function SourceCard({ source }) {
  const safeUrl = isValidHttpUrl(source?.url || "") ? source.url : "";

  return (
    <div className="card">
      <h3>{source?.title || "Untitled Source"}</h3>

      <p>
        <strong>Type:</strong> {source?.source_type || "unknown"}
      </p>

      <p>
        <strong>Rank:</strong> {source?.rank_score ?? "N/A"}
      </p>

      {safeUrl ? (
        <a href={safeUrl} target="_blank" rel="noreferrer noopener">
          Open Source
        </a>
      ) : (
        <span style={{ color: "#888" }}>No direct source link</span>
      )}
    </div>
  );
}

export default SourceCard;