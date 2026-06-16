import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { deleteExpedition, getExpeditions } from "../services/api";
import ExpeditionMenu from "../components/ExpeditionMenu";

function ExpeditionsDashboard() {
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    try {
      setLoading(true);
      const data = await getExpeditions();
      setItems(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error(err);
      setItems([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleDelete = async (id) => {
    const ok = window.confirm("Delete this expedition?");
    if (!ok) return;

    try {
      await deleteExpedition(id);
      await load();
    } catch (err) {
      console.error(err);
      alert("Delete failed");
    }
  };

  return (
    <div className="page">
      <div className="section">
        <h2>Your Expeditions</h2>

        {loading ? (
          <div className="empty">Loading...</div>
        ) : items.length > 0 ? (
          <div className="grid">
            {items.map((item) => (
              <div key={item.id} className="expedition-card expedition-card-row">
                <div className="expedition-info">
                  <div className="expedition-title">{item.title || item.root_query}</div>
                  <div className="muted">{item.root_query}</div>
                  <div className="meta-row">
                    <span>{item.status}</span>
                    <span>{item.claim_count || 0} claims</span>
                    <span>{item.source_count || 0} sources</span>
                  </div>
                </div>

                <ExpeditionMenu
                  onView={() => navigate(`/expeditions/${item.id}`)}
                  onChat={() => navigate(`/expeditions/${item.id}?chat=1`)}
                  onDelete={() => handleDelete(item.id)}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="empty">No expeditions saved yet.</div>
        )}
      </div>
    </div>
  );
}

export default ExpeditionsDashboard;