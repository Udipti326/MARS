import { useState } from "react";

function QueryBox({ onSearch, loading }) {
  const [query, setQuery] = useState("");

  const handleSubmit = () => {
    if (!query.trim()) return;
    onSearch(query);
  };

  return (
    <div className="query-box">
      <input
        type="text"
        placeholder="Enter research query..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <button onClick={handleSubmit} disabled={loading}>
        {loading ? "Running..." : "Research"}
      </button>
    </div>
  );
}

export default QueryBox;