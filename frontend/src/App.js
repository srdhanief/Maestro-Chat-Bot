import React, { useState, useRef, useEffect } from "react";
const BACKEND_URL = "http://127.0.0.1:8000/chat";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    setLoading(true);
    const userMsg = { role: "user", content: input };
    setMessages((m) => [...m, userMsg]);
    const payload = { query: input, top_k: 4, char_limit: 1400 };
    setInput("");

    try {
      const res = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const txt = await res.text();
        setMessages((m) => [...m, { role: "bot", content: "Error: " + txt }]);
        setLoading(false);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let botText = "";
      // placeholder bot message to allow streaming updates
      setMessages((m) => [...m, { role: "bot", content: "" }]);
      let botIndex = messages.length;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        botText += chunk;
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: "bot", content: botText };
          return copy;
        });
      }
    } catch (err) {
      setMessages((m) => [...m, { role: "bot", content: "⚠️ Error connecting to backend." }]);
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => setMessages([]);

  return (
    <div className="page">
      <aside className="side">
        <h2>Maestro Chat</h2>
        <p className="small">Offline • Private • RAG</p>
        <div className="meta">
          <div>Docs: 2 PDFs</div>
          <div>Retriever: FAISS (384d)</div>
          <div>Model: GPT4All (CPU)</div>
        </div>
        <button className="btn" onClick={clearChat}>Clear</button>
      </aside>

      <main className="main">
        <div className="header">Maestro Chat Bot</div>

        <div className="messages" id="msgs">
          {messages.length === 0 && !loading && (
            <div className="hint">Ask about Maestro configuration, workflows, validations...</div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role === "user" ? "user" : "bot"}`}>
              <div className="bubble">{m.content}</div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="composer">
          <textarea
            placeholder="Type your Maestro question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={loading}
            rows={2}
          />
          <div className="controls">
            <button className="send" onClick={sendMessage} disabled={loading}>
              {loading ? "Thinking..." : "Send"}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
