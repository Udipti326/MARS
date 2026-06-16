import { useEffect, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";

import { getExpedition } from "../services/api";
import SummaryCard from "../components/SummaryCard";
import OverallCard from "../components/OverallCard";
import ClaimCard from "../components/ClaimCard";
import SourceCard from "../components/SourceCard";
import ResearchChatModal from "../components/ResearchChatModal";

function ExpeditionDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [detail, setDetail] = useState(null);
  const [chatOpen, setChatOpen] = useState(searchParams.get("chat") === "1");

  const loadDetail = async () => {
    try {
      const data = await getExpedition(id);
      setDetail(data);
    } catch (err) {
      console.error(err);
      setDetail(null);
    }
  };

  useEffect(() => {
    loadDetail();
  }, [id]);

  useEffect(() => {
    setChatOpen(searchParams.get("chat") === "1");
  }, [searchParams]);

  if (!detail) {
    return (
      <div className="page">
        <div className="empty">Loading expedition...</div>
      </div>
    );
  }

  const summary = detail.summary || detail.summary_json || {};
  const overall = detail.overall || detail.overall_json || {};
  const claims = detail.claims || [];
  const sources = detail.sources || [];

  return (
    <div className="page">
      <div className="section">
        <h2>{detail.title || detail.root_query}</h2>
        <div className="muted">{detail.root_query}</div>
      </div>

      <div className="toolbar">
        <button className="button" onClick={() => setChatOpen(true)}>
          Chat
        </button>
        <button className="button button-secondary" onClick={() => navigate("/expeditions")}>
          Back
        </button>
      </div>

      <SummaryCard summary={summary} />
      <OverallCard overall={overall} />

      <div className="section">
        <h2>Claims</h2>
        {claims.length > 0 ? (
          claims.map((claim, index) => <ClaimCard key={index} claim={claim} />)
        ) : (
          <div className="empty">No claims stored.</div>
        )}
      </div>

      <div className="section">
        <h2>Sources</h2>
        {sources.length > 0 ? (
          sources.map((source, index) => <SourceCard key={index} source={source} />)
        ) : (
          <div className="empty">No sources stored.</div>
        )}
      </div>

      <ResearchChatModal
        open={chatOpen}
        onClose={() => setChatOpen(false)}
        expeditionId={id}
        title={detail.title || detail.root_query}
      />
    </div>
  );
}

export default ExpeditionDetail;