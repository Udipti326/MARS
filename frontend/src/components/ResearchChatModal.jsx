import { useEffect, useMemo, useRef, useState } from "react";
import { getChatMemory, sendChatMessage } from "../services/api";

function ResearchChatModal({ open, onClose, expeditionId, title }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [loading, setLoading] = useState(false);
  const [latestMeta, setLatestMeta] = useState(null);
  const bottomRef = useRef(null);

  const canChat = useMemo(() => Boolean(expeditionId), [expeditionId]);

  useEffect(() => {
    const load = async () => {
      if (!open || !expeditionId) return;
      try {
        setLoading(true);
        const data = await getChatMemory(expeditionId);
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

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, latestMeta, open]);

  const handleSend = async () => {
    if (!input.trim() || !expeditionId) return;

    const question = input.trim();
    setInput("");
    setSending(true);
    setLatestMeta(null);

    try {
      setMessages((prev) => [
        ...prev,
        {
          id: `local-user-${Date.now()}`,
          role: "user",
          content: question,
        },
      ]);

      const response = await sendChatMessage(expeditionId, question);

      setMessages((prev) => [
        ...prev,
        {
          id: `local-assistant-${Date.now()}`,
          role: "assistant",
          content: response?.answer || "",
        },
      ]);

      setLatestMeta({
        confidence: response?.confidence,
        citations: response?.citations || [],
        web_results: response?.web_results || [],
      });
    } catch (err) {
      console.error(err);
      alert("Chat failed");
    } finally {
      setSending(false);
    }
  };

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="chat-modal" onClick={(e) => e.stopPropagation()}>
        <div className="chat-modal-header">
          <div>
            <div className="chat-modal-title">Research Chat</div>
            <div className="muted">{title || "Expedition context"}</div>
          </div>
          <button className="icon-button" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="chat-modal-body">
          {!canChat ? (
            <div className="empty">No expedition selected.</div>
          ) : loading ? (
            <div className="empty">Loading memory...</div>
          ) : messages.length ? (
            messages.map((msg) => (
              <div key={msg.id} className={`chat-bubble chat-${msg.role}`}>
                <div className="chat-role">{msg.role}</div>
                <div>{msg.content}</div>
              </div>
            ))
          ) : (
            <div className="empty">Ask a question about this expedition.</div>
          )}
          <div ref={bottomRef} />
        </div>

        {latestMeta && (
          <div className="chat-meta">
            <div>
              <strong>Confidence:</strong>{" "}
              {typeof latestMeta.confidence === "number"
                ? latestMeta.confidence.toFixed(3)
                : "N/A"}
            </div>
            {latestMeta.citations?.length > 0 && (
              <div className="chat-meta-block">
                <strong>Citations</strong>
                <ul>
                  {latestMeta.citations.map((c, idx) => (
                    <li key={idx}>
                      <a href={c.url || "#"} target="_blank" rel="noreferrer">
                        {c.title || c.url || `Citation ${idx + 1}`}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        <div className="chat-input-row">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask something about this research..."
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSend();
            }}
          />
          <button className="button" onClick={handleSend} disabled={sending || !canChat}>
            {sending ? "Sending..." : "Send"}
          </button>
        </div>  
      </div>
    </div>
  );
}

export default ResearchChatModal;