import { useEffect, useState } from "react";
import { getExpeditionMessages, sendExpeditionMessage } from "../services/api";

function ExpeditionChatPanel({ expeditionId }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const load = async () => {
      if (!open || !expeditionId) return;
      try {
        setLoading(true);
        const data = await getExpeditionMessages(expeditionId);
        setMessages(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error(err);
        setMessages([]);
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [open, expeditionId]);

  const handleSend = async () => {
    if (!text.trim() || !expeditionId) return;

    try {
      setSending(true);
      const reply = await sendExpeditionMessage(expeditionId, text.trim());
      setText("");

      setMessages((prev) => [
        ...prev,
        { role: "user", content: text.trim(), id: `u-${Date.now()}` },
        { role: "assistant", content: reply?.answer || "", id: `a-${Date.now()}` },
      ]);
    } catch (err) {
      console.error(err);
      alert("Chat failed");
    } finally {
      setSending(false);
    }
  };

  if (!expeditionId) return null;

  return (
    <div className="card" style={{ marginTop: "16px" }}>
      <h2>Ask about this expedition</h2>

      {!open ? (
        <button className="button" onClick={() => setOpen(true)}>
          Open Chat
        </button>
      ) : (
        <>
          <div className="chat-box">
            {loading ? (
              <div className="empty">Loading chat...</div>
            ) : messages.length > 0 ? (
              messages.map((msg, idx) => (
                <div key={msg.id || idx} className={`chat-line chat-${msg.role}`}>
                  <strong>{msg.role}:</strong> {msg.content}
                </div>
              ))
            ) : (
              <div className="empty">No chat yet. Ask a question.</div>
            )}
          </div>

          <div className="chat-input-row">
            <input
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Ask a follow-up question..."
            />
            <button className="button" onClick={handleSend} disabled={sending}>
              {sending ? "Sending..." : "Send"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

export default ExpeditionChatPanel;