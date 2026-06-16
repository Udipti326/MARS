import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

import { getCFG, getExpeditions, rebuildCFG } from "../services/api";
import CFGGraph from "../components/CFGGraph";

function CFGPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [expeditions, setExpeditions] = useState([]);
  const [selectedId, setSelectedId] = useState(id || "");
  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [status, setStatus] = useState("");
  const [selectedNode, setSelectedNode] = useState(null);

  useEffect(() => {
    const loadList = async () => {
      try {
        const items = await getExpeditions();
        const list = Array.isArray(items) ? items : [];
        setExpeditions(list);

        if (!selectedId && list.length > 0) {
          setSelectedId(list[0].id);
          navigate(`/CFG/${list[0].id}`);
        }
      } catch (err) {
        console.error(err);
        setExpeditions([]);
      }
    };

    loadList();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (id && id !== selectedId) {
      setSelectedId(id);
    }
  }, [id, selectedId]);

  useEffect(() => {
    let alive = true;

    const loadGraph = async () => {
      if (!selectedId) {
        setGraph(null);
        setStatus("Select an expedition to view its CFG.");
        return;
      }

      try {
        setLoading(true);
        setStatus("");
        const data = await getCFG(selectedId);

        if (!alive) return;

        setGraph(data);
        if (!data?.ready || !data?.nodes?.length) {
          setStatus("CFG not built yet. Click Rebuild CFG.");
        } else {
          setStatus("");
        }
      } catch (err) {
        console.error(err);
        if (alive) {
          setGraph(null);
          setStatus("Failed to load CFG.");
        }
      } finally {
        if (alive) setLoading(false);
      }
    };

    loadGraph();

    return () => {
      alive = false;
    };
  }, [selectedId]);

  const handleRebuild = async () => {
    if (!selectedId) return;

    try {
      setRebuilding(true);
      setStatus("Rebuilding CFG...");
      const data = await rebuildCFG(selectedId);
      setGraph(data);
      setSelectedNode(null);
      setStatus(
        data?.nodes?.length
          ? "CFG rebuilt successfully."
          : "CFG rebuilt, but no concepts were found."
      );
    } catch (err) {
      console.error(err);
      const msg = err?.response?.data?.error || err.message || "CFG rebuild failed";
      setStatus(msg);
      alert(msg);
    } finally {
      setRebuilding(false);
    }
  };

  const nodeById = useMemo(() => {
    const map = new Map();
    for (const node of graph?.nodes || []) {
      map.set(String(node.id), node);
    }
    return map;
  }, [graph]);

  const selectedNodeId = selectedNode ? String(selectedNode.id) : "";

  const connectedLinks = useMemo(() => {
    if (!selectedNode || !Array.isArray(graph?.links)) return [];

    return graph.links
      .map((link) => {
        const sourceId = String(link?.source?.id ?? link?.source);
        const targetId = String(link?.target?.id ?? link?.target);

        if (sourceId !== selectedNodeId && targetId !== selectedNodeId) return null;

        const otherId = sourceId === selectedNodeId ? targetId : sourceId;
        const otherNode = nodeById.get(String(otherId));

        return {
          ...link,
          sourceId,
          targetId,
          otherNode,
          otherId,
        };
      })
      .filter(Boolean)
      .sort((a, b) => Number(b.weight || 0) - Number(a.weight || 0));
  }, [graph, nodeById, selectedNode, selectedNodeId]);

  const curveData = graph?.forgotten_curve || [];
  const trends = graph?.trends || [];
  const learnNext = graph?.learn_next || [];
  const selectedExpedition = expeditions.find((e) => e.id === selectedId);

  const selectedSemanticScore = Number(
    selectedNode?.relevance_score ??
      selectedNode?.score ??
      selectedNode?.confidence ??
      0
  );

  const selectedTypeLabel = selectedNode?.type || "node";
  const selectedLabel = selectedNode?.label || selectedNode?.id || "Node";

  return (
    <div className="page">
      <div className="section">
        <h2>CFG</h2>
        <div className="muted">
          Concept Flow Graph built from your semantic similarity model
        </div>
        <div className="muted">
          Hover to see labels. Click a node to inspect semantic score and related links.
        </div>
        {selectedExpedition && (
          <div className="muted">
            {selectedExpedition.title || selectedExpedition.root_query}
          </div>
        )}
      </div>

      <div className="toolbar">
        <select
          className="input-like"
          value={selectedId}
          onChange={(e) => {
            const next = e.target.value;
            setSelectedId(next);
            setSelectedNode(null);
            navigate(next ? `/CFG/${next}` : "/CFG");
          }}
        >
          <option value="">Select an expedition</option>
          {expeditions.map((exp) => (
            <option key={exp.id} value={exp.id}>
              {exp.title || exp.root_query}
            </option>
          ))}
        </select>

        <button className="button" onClick={handleRebuild} disabled={rebuilding || !selectedId}>
          {rebuilding ? "Rebuilding..." : "Rebuild CFG"}
        </button>
      </div>

      {status ? <div className="empty">{status}</div> : null}

      {loading ? (
        <div className="empty">Loading graph...</div>
      ) : (
        <>
          <CFGGraph
            data={graph}
            selectedNodeId={selectedNodeId}
            onNodeClick={(node) => setSelectedNode(node)}
          />

          <div className="grid" style={{ marginTop: 16 }}>
            <div className="card">
              <h3>What to learn next</h3>
              {learnNext.length ? (
                <ul className="plain-list">
                  {learnNext.map((item, idx) => (
                    <li key={idx}>
                      <strong>{item.name}</strong> — {Number(item.score || 0).toFixed(3)}
                      {item.reason ? <div className="muted">{item.reason}</div> : null}
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="empty">No recommendations yet.</div>
              )}
            </div>

            <div className="card">
              <h3>Research depth</h3>
              {trends.length ? (
                <ul className="plain-list">
                  {trends.map((item, idx) => (
                    <li key={idx}>
                      <strong>{item.name}</strong> — depth{" "}
                      {Number(item.depth || 0).toFixed(3)}
                      <div className="muted">
                        claims: {item.claims} · relations: {item.relations} · score:{" "}
                        {Number(item.relevance_score ?? item.topic_score ?? 0).toFixed(6)}
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="empty">No depth data yet.</div>
              )}
            </div>

            <div className="card">
              <h3>Forgotten curve</h3>
              {curveData.length ? (
                <div style={{ width: "100%", height: 240 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={curveData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="day" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="retention"
                        stroke="#ffffff"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="empty">No curve data.</div>
              )}
            </div>
          </div>
        </>
      )}

      {selectedNode ? (
        <div
          className="cfg-modal-backdrop"
          onClick={() => setSelectedNode(null)}
          role="presentation"
        >
          <div
            className="cfg-modal"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
          >
            <div className="cfg-modal-header">
              <div>
                <div className="cfg-modal-kicker">Selected node</div>
                <h3>{selectedLabel}</h3>
              </div>
              <button className="cfg-close-btn" onClick={() => setSelectedNode(null)}>
                ×
              </button>
            </div>

            <div className="cfg-modal-meta">
              <span className="cfg-pill">{selectedTypeLabel}</span>
              <span className="cfg-pill">
                Semantic score: {selectedSemanticScore.toFixed(6)}
              </span>
            </div>

            <div className="cfg-modal-body">
              <div className="cfg-detail-block">
                <h4>Node details</h4>
                <p>
                  This node is shown without an always-visible label. Hover the node in the graph to
                  preview its name and click it to inspect its semantic score.
                </p>
              </div>

              <div className="cfg-detail-block">
                <h4>How related it is</h4>
                {connectedLinks.length ? (
                  <ul className="cfg-connection-list">
                    {connectedLinks.map((link, idx) => {
                      const other = link.otherNode;
                      const otherName = other?.label || other?.id || link.otherId;
                      return (
                        <li key={idx} className="cfg-connection-row">
                          <div>
                            <strong>{otherName}</strong>
                            <div className="muted">{link.type || "related"}</div>
                          </div>
                          <div className="cfg-score-badge">
                            {Number(link.weight || 0).toFixed(6)}
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <div className="empty">No connected links for this node.</div>
                )}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default CFGPage;