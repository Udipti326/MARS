import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import Header from "../components/Header";
import QueryBox from "../components/QueryBox";
import SummaryCard from "../components/SummaryCard";
import ClaimCard from "../components/ClaimCard";
import SourceCard from "../components/SourceCard";
import OverallCard from "../components/OverallCard";
import ResearchChatModal from "../components/ResearchChatModal";

import { runResearch, saveExpedition } from "../services/api";

function Home() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [data, setData] = useState(null);
  const [savedId, setSavedId] = useState(null);
  const [lastQuery, setLastQuery] = useState("");
  const [chatOpen, setChatOpen] = useState(false);

  const claims = useMemo(() => data?.claims || [], [data]);
  const sources = useMemo(() => data?.sources || [], [data]);

  const expeditionId = data?.expedition_id || savedId;

  const handleSearch = async (query) => {
    try {
      setLoading(true);
      setLastQuery(query);
      const response = await runResearch(query, expeditionId || null);
      setData(response);
      setSavedId(response?.expedition_id || null);
    } catch (err) {
      console.error(err);
      alert("Research failed");
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!data) return;

    try {
      setSaving(true);
      const response = await saveExpedition(data, expeditionId || null);
      setSavedId(response.expedition_id || expeditionId || null);
      alert("Expedition saved.");
    } catch (err) {
      console.error(err);
      alert("Save failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="page">
      <Header />
      <QueryBox onSearch={handleSearch} loading={loading} />

      {data && (
        <>
          <div className="toolbar">
            <button className="button" onClick={handleSave} disabled={saving}>
              {saving ? "Saving..." : "Save Expedition"}
            </button>

            <button className="button button-secondary" onClick={() => setChatOpen(true)}>
              Ask Research Bot
            </button>

            {savedId && (
              <button
                className="button button-secondary"
                onClick={() => navigate(`/expeditions/${savedId}`)}
              >
                Open Expedition
              </button>
            )}
          </div>

          <SummaryCard summary={data.summary} />
          <OverallCard overall={data.overall} />

          <div className="section">
            <h2>Claims</h2>
            {claims.length > 0 ? (
              claims.map((claim, index) => <ClaimCard key={index} claim={claim} />)
            ) : (
              <div className="empty">No claims returned for: {lastQuery}</div>
            )}
          </div>

          <div className="section">
            <h2>Sources</h2>
            {sources.length > 0 ? (
              sources.map((source, index) => <SourceCard key={index} source={source} />)
            ) : (
              <div className="empty">No sources available.</div>
            )}
          </div>

          <ResearchChatModal
            open={chatOpen}
            onClose={() => setChatOpen(false)}
            expeditionId={expeditionId}
            title={data?.title || lastQuery}
          />
        </>
      )}
    </div>
  );
}

export default Home;