import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import {
  AlertCircle,
  Archive,
  BookOpen,
  Bot,
  CheckCircle2,
  ChevronDown,
  Database,
  Copy,
  FileText,
  Globe2,
  History,
  Link,
  Loader2,
  Menu,
  MessageSquarePlus,
  PanelLeftClose,
  PanelLeftOpen,
  Paperclip,
  RefreshCcw,
  Search,
  Send,
  Square,
  Trash2,
  Upload,
  User,
  X,
} from "lucide-react";
import "katex/dist/katex.min.css";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

function cls(...parts) {
  return parts.filter(Boolean).join(" ");
}

function friendlyError(error) {
  const message = error?.message || String(error || "");
  if (/failed to fetch|networkerror|load failed/i.test(message)) {
    return "I could not reach the CiteMind AI backend. Check that the app server is running, then retry.";
  }
  if (/api\.tavily\.com|tavily|name resolution|temporary failure in name resolution|connectionpool/i.test(message)) {
    return "Live web search is temporarily unavailable. Loaded papers and URLs still work; ask about the loaded source or retry web search later.";
  }
  if (/abort/i.test(error?.name || message)) return "Generation stopped.";
  if (/timeout/i.test(message)) return "The request timed out. Retry when the model is available.";
  return message.replace(/^Error:\s*/, "") || "Something went wrong. Please retry.";
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      message = body.detail || body.message || message;
    } catch {
      // Keep the HTTP status message.
    }
    throw new Error(message);
  }
  return response.json();
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

function formatDate(value) {
  try {
    return new Intl.DateTimeFormat(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(value));
  } catch {
    return "";
  }
}

function parseSseBlock(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  const raw = dataLines.join("\n");
  return { event, data: raw ? JSON.parse(raw) : {} };
}

function App() {
  const [sessions, setSessions] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarDrawerOpen, setSidebarDrawerOpen] = useState(false);
  const [sourcePanelOpen, setSourcePanelOpen] = useState(false);
  const [sourcePanelWidth, setSourcePanelWidth] = useState(360);
  const [selectedTurn, setSelectedTurn] = useState(null);
  const [sessionQuery, setSessionQuery] = useState("");
  const [notice, setNotice] = useState(null);
  const [lastPrompt, setLastPrompt] = useState("");
  const abortRef = useRef(null);
  const scrollRef = useRef(null);
  const sourceResizeRef = useRef(null);

  useEffect(() => {
    const storedWidth = Number(window.localStorage.getItem("citemind-source-panel-width"));
    if (Number.isFinite(storedWidth) && storedWidth > 0) {
      setSourcePanelWidth(storedWidth);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem("citemind-source-panel-width", String(sourcePanelWidth));
  }, [sourcePanelWidth]);

  useEffect(() => {
    return () => {
      sourceResizeRef.current?.();
      sourceResizeRef.current = null;
    };
  }, []);

  useEffect(() => {
    refreshSessions().catch((error) => setNotice({ type: "error", text: friendlyError(error) }));
  }, []);

  useEffect(() => {
    if (activeId) {
      loadSession(activeId).catch((error) =>
        setNotice({ type: "error", text: friendlyError(error) })
      );
    }
  }, [activeId]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  const activeSession = sessions.find((session) => session.id === activeId);
  const filteredSessions = useMemo(() => {
    const query = sessionQuery.trim().toLowerCase();
    if (!query) return sessions;
    return sessions.filter((session) => session.name?.toLowerCase().includes(query));
  }, [sessions, sessionQuery]);

  async function refreshSessions(preferredId = null) {
    const data = await api("/api/sessions");
    setSessions(data.sessions);
    const candidateId = preferredId || activeId;
    const nextId = data.sessions.some((session) => session.id === candidateId)
      ? candidateId
      : data.sessions[0]?.id;
    setActiveId(nextId || null);
  }

  async function loadSession(sessionId) {
    const data = await api(`/api/sessions/${sessionId}`);
    setMessages(data.messages || []);
    setDocuments(data.documents || []);
    setSelectedTurn(null);
  }

  async function createSession() {
    const data = await api("/api/sessions", { method: "POST" });
    await refreshSessions(data.session.id);
    setMessages([]);
    setDocuments([]);
    setInput("");
    setSelectedTurn(null);
    setSidebarDrawerOpen(false);
  }

  async function deleteSession(sessionId) {
    if (!sessionId) return;
    if (loading && sessionId === activeId) {
      setNotice({ type: "error", text: "Stop the current response before deleting this chat." });
      return;
    }

    const deletingActive = sessionId === activeId;
    setNotice({ type: "loading", text: "Deleting chat..." });
    try {
      const data = await api(`/api/sessions/${sessionId}`, { method: "DELETE" });
      setSessions(data.sessions || []);

      if (deletingActive) {
        setMessages([]);
        setDocuments([]);
        setSelectedTurn(null);
        setSourcePanelOpen(false);
        if (data.next_session_id) {
          setActiveId(data.next_session_id);
        } else {
          await refreshSessions();
        }
      }

      setSidebarDrawerOpen(false);
      setNotice({ type: "success", text: "Deleted chat." });
    } catch (error) {
      setNotice({ type: "error", text: friendlyError(error) });
    }
  }

  async function deleteAllSessions() {
    if (loading) {
      setNotice({ type: "error", text: "Stop the current response before deleting all chats." });
      return;
    }

    setNotice({ type: "loading", text: "Deleting all chats..." });
    try {
      await api("/api/sessions", { method: "DELETE" });
      setSessions([]);
      setActiveId(null);
      setMessages([]);
      setDocuments([]);
      setInput("");
      setSelectedTurn(null);
      setSourcePanelOpen(false);
      setSidebarDrawerOpen(false);
      await refreshSessions();
      setNotice({ type: "success", text: "Deleted all chats." });
    } catch (error) {
      setNotice({ type: "error", text: friendlyError(error) });
    }
  }

  async function reloadDocuments() {
    if (!activeId) return;
    const data = await api(`/api/sessions/${activeId}/documents`);
    setDocuments(data.documents || []);
  }

  async function handleUpload(files) {
    if (!activeId || !files?.length) return;
    const form = new FormData();
    Array.from(files).forEach((file) => form.append("files", file));
    setNotice({ type: "loading", text: "Uploading and indexing files..." });
    try {
      const data = await api(`/api/sessions/${activeId}/documents/files`, {
        method: "POST",
        body: form,
      });
      setDocuments(data.documents || []);
      const failures = data.failed?.length ? ` ${data.failed.length} failed.` : "";
      setNotice({ type: "success", text: `Loaded ${data.loaded.length} file(s).${failures}` });
    } catch (error) {
      setNotice({ type: "error", text: friendlyError(error) });
    }
  }

  async function loadUrls(value) {
    if (!activeId || !value.trim()) return;
    setNotice({ type: "loading", text: "Loading web pages..." });
    try {
      const data = await api(`/api/sessions/${activeId}/documents/urls`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: value.split("\n") }),
      });
      setDocuments(data.documents || []);
      const failures = data.failed?.length ? ` ${data.failed.length} failed.` : "";
      setNotice({ type: "success", text: `Loaded ${data.loaded.length} URL(s).${failures}` });
    } catch (error) {
      setNotice({ type: "error", text: friendlyError(error) });
    }
  }

  async function loadArxiv(query) {
    if (!activeId || !query.trim()) return;
    setNotice({ type: "loading", text: "Loading arXiv paper..." });
    try {
      const data = await api(`/api/sessions/${activeId}/documents/arxiv`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      setDocuments(data.documents || []);
      setNotice({ type: "success", text: `Loaded ${data.loaded[0] || "paper"}.` });
    } catch (error) {
      setNotice({ type: "error", text: friendlyError(error) });
    }
  }

  async function sendPrompt(prompt) {
    const content = prompt.trim();
    if (!content || !activeId || loading) return;

    const assistantId = crypto.randomUUID();
    setInput("");
    setLoading(true);
    setLastPrompt(content);
    setNotice(null);
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", content },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        pending: true,
        graph_state: null,
        retrieved_context: [],
      },
    ]);

    const abort = new AbortController();
    abortRef.current = abort;
    try {
      const response = await fetch(`${API_BASE}/api/sessions/${activeId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content }),
        signal: abort.signal,
      });
      if (!response.ok || !response.body) throw new Error(`Chat failed with status ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamed = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          if (!block.trim()) continue;
          const { event: eventName, data } = parseSseBlock(block);
          if (eventName === "token") {
            streamed += data.content || "";
            patchAssistant(assistantId, { content: streamed });
          }
          if (eventName === "replace") {
            streamed = data.content || "";
            patchAssistant(assistantId, { content: streamed });
          }
          if (eventName === "final") {
            streamed = data.answer || streamed;
            patchAssistant(assistantId, {
              content: streamed,
              pending: false,
              graph_state: data.graph_state || {},
              retrieved_context: data.retrieved_context || [],
              failed: false,
            });
            await refreshSessions(activeId);
          }
          if (eventName === "error") throw new Error(data.message || "Chat failed");
        }
      }
    } catch (error) {
      if (error.name === "AbortError") {
        patchAssistant(assistantId, {
          pending: false,
          stopped: true,
          content: "Generation stopped.",
        });
      } else {
        patchAssistant(assistantId, {
          pending: false,
          failed: true,
          content: friendlyError(error),
          graph_state: { error: error.message },
        });
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
      reloadDocuments().catch(() => {});
    }
  }

  function patchAssistant(id, patch) {
    setMessages((current) =>
      current.map((message) => (message.id === id ? { ...message, ...patch } : message))
    );
  }

  function sendMessage(event) {
    event?.preventDefault();
    sendPrompt(input);
  }

  function stopGeneration() {
    abortRef.current?.abort();
  }

  function startSourcePanelResize(event) {
    if (event.button !== 0) return;
    event.preventDefault();
    const minWidth = 320;
    const maxWidth = Math.max(minWidth, Math.min(760, window.innerWidth - 64));

    const updateWidth = (clientX) => {
      const nextWidth = Math.round(window.innerWidth - clientX);
      setSourcePanelWidth(Math.min(maxWidth, Math.max(minWidth, nextWidth)));
    };

    const handlePointerMove = (moveEvent) => {
      updateWidth(moveEvent.clientX);
    };

    const handlePointerUp = () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
      sourceResizeRef.current = null;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    sourceResizeRef.current = handlePointerUp;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    updateWidth(event.clientX);
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("pointercancel", handlePointerUp);
  }

  function retryLastPrompt() {
    if (lastPrompt && !loading) sendPrompt(lastPrompt);
  }

  function openTurnDetails(message) {
    setSelectedTurn(message);
    setSourcePanelOpen(true);
  }

  return (
    <main
      className={cls(
        "app-shell",
        sidebarCollapsed && "sidebar-collapsed",
        sidebarDrawerOpen && "sidebar-drawer-open",
        sourcePanelOpen && "source-panel-open"
      )}
      style={{ "--source-panel-width": `${sourcePanelWidth}px` }}
    >
      <button
        className="scrim"
        onClick={() => {
          setSidebarDrawerOpen(false);
          setSourcePanelOpen(false);
        }}
        aria-label="Close open panel"
      />

      <Sidebar
        sessions={filteredSessions}
        activeId={activeId}
        sessionQuery={sessionQuery}
        onSessionQuery={setSessionQuery}
        onCreate={createSession}
        onSelect={(id) => {
          setActiveId(id);
          setSidebarDrawerOpen(false);
        }}
        onDelete={deleteSession}
        onDeleteAll={deleteAllSessions}
        deletingDisabled={loading}
        collapsed={sidebarCollapsed}
      />

      <section className="main-pane">
        <TopBar
          activeSession={activeSession}
          documents={documents}
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={() => setSidebarCollapsed((value) => !value)}
          onOpenSidebar={() => setSidebarDrawerOpen(true)}
          onOpenSources={() => setSourcePanelOpen(true)}
        />

        <ConversationViewport
          messages={messages}
          loading={loading}
          onPrompt={sendPrompt}
          onInspect={openTurnDetails}
          onRetry={retryLastPrompt}
          bottomRef={scrollRef}
        />

        <Composer
          value={input}
          onChange={setInput}
          onSubmit={sendMessage}
          onStop={stopGeneration}
          loading={loading}
          disabled={!activeId}
          onOpenSources={() => setSourcePanelOpen(true)}
        />
      </section>

      <SourcePanel
        open={sourcePanelOpen}
        documents={documents}
        notice={notice}
        setNotice={setNotice}
        selectedTurn={selectedTurn}
        onClose={() => setSourcePanelOpen(false)}
        onUpload={handleUpload}
        onLoadUrls={loadUrls}
        onLoadArxiv={loadArxiv}
        onResizeStart={startSourcePanelResize}
      />
    </main>
  );
}

function Sidebar({
  sessions,
  activeId,
  sessionQuery,
  onSessionQuery,
  onCreate,
  onSelect,
  onDelete,
  onDeleteAll,
  deletingDisabled,
  collapsed,
}) {
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);
  const [confirmDeleteAll, setConfirmDeleteAll] = useState(false);

  function requestDelete(event, session) {
    event.stopPropagation();
    setConfirmDeleteId(session.id);
  }

  function cancelDelete(event) {
    event.stopPropagation();
    setConfirmDeleteId(null);
  }

  async function confirmDelete(event, session) {
    event.stopPropagation();
    await onDelete(session.id);
    setConfirmDeleteId(null);
  }

  async function confirmDeleteAllChats() {
    await onDeleteAll();
    setConfirmDeleteAll(false);
    setConfirmDeleteId(null);
  }

  return (
    <aside className="sidebar" aria-label="Conversation history">
      <div className="sidebar-header">
        <div className="brand">
          <div className="brand-mark">
            <BookOpen size={18} />
          </div>
          <div>
            <strong>CiteMind AI</strong>
            <span>Evidence intelligence</span>
          </div>
        </div>
      </div>

      <button className="new-chat" onClick={onCreate}>
        <MessageSquarePlus size={17} />
        <span>New chat</span>
      </button>

      <label className="session-search">
        <Search size={15} />
        <input
          value={sessionQuery}
          onChange={(event) => onSessionQuery(event.target.value)}
          placeholder="Search chats"
        />
      </label>

      <div className="sidebar-label">
        <History size={14} />
        <span>Conversations</span>
      </div>

      <nav className="session-list">
        {sessions.map((session) => (
          <div
            key={session.id}
            className={cls("session-row", session.id === activeId && "active")}
            data-session-id={session.id}
          >
            <button
              className="session-item"
              onClick={() => onSelect(session.id)}
              title={session.name}
            >
              <span>{session.name || "New Session"}</span>
              <small>{formatDate(session.created_at)}</small>
            </button>
            {confirmDeleteId === session.id ? (
              <div className="session-confirm" aria-label={`Confirm delete ${session.name || "chat"}`}>
                <button
                  className="confirm-delete"
                  onClick={(event) => confirmDelete(event, session)}
                  type="button"
                >
                  Delete
                </button>
                <button className="cancel-delete" onClick={cancelDelete} type="button">
                  Cancel
                </button>
              </div>
            ) : (
              <button
                className="session-delete"
                onClick={(event) => requestDelete(event, session)}
                disabled={deletingDisabled && session.id === activeId}
                title={`Delete ${session.name || "chat"}`}
                aria-label={`Delete ${session.name || "chat"}`}
                type="button"
              >
                <Trash2 size={15} />
              </button>
            )}
          </div>
        ))}
        {!sessions.length ? <p className="sidebar-empty">No matching chats.</p> : null}
      </nav>

      <div className="sidebar-footer">
        {confirmDeleteAll ? (
          <div className="delete-all-confirm">
            <span>Delete all chats?</span>
            <button
              className="confirm-delete"
              onClick={confirmDeleteAllChats}
              type="button"
            >
              Delete all
            </button>
            <button
              className="cancel-delete"
              onClick={() => setConfirmDeleteAll(false)}
              type="button"
            >
              Cancel
            </button>
          </div>
        ) : (
          <>
            <div className="sidebar-memory">
              <Database size={15} />
              <span>{collapsed ? "" : "Session-scoped evidence memory"}</span>
            </div>
            <button
              className="delete-all-button"
              onClick={() => setConfirmDeleteAll(true)}
              disabled={deletingDisabled || !sessions.length}
              type="button"
            >
              <Trash2 size={14} />
              <span>Delete all</span>
            </button>
          </>
        )}
      </div>
    </aside>
  );
}

function TopBar({
  activeSession,
  documents,
  sidebarCollapsed,
  onToggleSidebar,
  onOpenSidebar,
  onOpenSources,
}) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <button className="icon-button mobile-only" onClick={onOpenSidebar} aria-label="Open sidebar">
          <Menu size={20} />
        </button>
        <button
          className="icon-button desktop-only"
          onClick={onToggleSidebar}
          aria-label="Toggle sidebar"
        >
          {sidebarCollapsed ? <PanelLeftOpen size={19} /> : <PanelLeftClose size={19} />}
        </button>
        <div className="topbar-title">
          <h1>{activeSession?.name || "New chat"}</h1>
          <p>{documents.length ? `${documents.length} source${documents.length === 1 ? "" : "s"} loaded` : "No sources loaded"}</p>
        </div>
      </div>
      <button className="sources-button" onClick={onOpenSources}>
        <Paperclip size={17} />
        <span>Sources</span>
      </button>
    </header>
  );
}

function ConversationViewport({ messages, loading, onPrompt, onInspect, onRetry, bottomRef }) {
  return (
    <div className="conversation-viewport">
      <div className="conversation-column">
        {messages.length === 0 ? <EmptyState onPrompt={onPrompt} /> : null}
        {messages.map((message, index) => (
          <ChatMessage
            key={message.id || `${message.role}-${index}`}
            message={message}
            onInspect={() => onInspect(message)}
            onRetry={onRetry}
          />
        ))}
        {loading ? (
          <div className="thinking">
            <Loader2 size={16} className="spin" />
            <span>Tracing evidence through the research graph</span>
          </div>
        ) : null}
        <div className="bottom-spacer" ref={bottomRef} />
      </div>
    </div>
  );
}

function EmptyState({ onPrompt }) {
  const prompts = [
    "Summarize the uploaded paper like a reviewer.",
    "Verify whether this claim is still supported.",
    "Find newer papers that challenge this result.",
  ];
  return (
    <section className="empty-state">
      <div className="empty-mark">
        <Bot size={26} />
      </div>
      <h2>Turn research papers into defensible answers.</h2>
      <p>
        CiteMind AI reads papers, checks claims against recent literature, and keeps the
        retrieved evidence visible for every answer.
      </p>
      <div className="prompt-grid">
        {prompts.map((prompt) => (
          <button key={prompt} onClick={() => onPrompt(prompt)}>
            {prompt}
          </button>
        ))}
      </div>
    </section>
  );
}

function ChatMessage({ message, onInspect, onRetry }) {
  const isUser = message.role === "user";
  const hasTurnDetails =
    !isUser && ((message.retrieved_context || []).length > 0 || message.graph_state);
  return (
    <article className={cls("message", isUser ? "user-message" : "assistant-message")}>
      <div className="avatar" aria-hidden="true">
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div className="message-body">
        <div className={cls("message-content", message.failed && "failed")}>
          {message.content ? (
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
              {message.content}
            </ReactMarkdown>
          ) : (
            <span className="muted">Waiting for response...</span>
          )}
          {message.pending ? <span className="cursor" /> : null}
        </div>

        {!isUser ? (
          <div className="message-actions">
            {hasTurnDetails ? (
              <button onClick={onInspect}>
                <Database size={14} />
                Inspect sources
              </button>
            ) : null}
            {message.failed ? (
              <button onClick={onRetry}>
                <RefreshCcw size={14} />
                Retry
              </button>
            ) : null}
            {message.stopped ? <span className="status-note">Stopped</span> : null}
          </div>
        ) : null}
      </div>
    </article>
  );
}

function Composer({ value, onChange, onSubmit, onStop, loading, disabled, onOpenSources }) {
  const textareaRef = useRef(null);
  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, 180)}px`;
  }, [value]);

  const canSend = value.trim() && !loading && !disabled;

  return (
    <form className="composer-area" onSubmit={onSubmit}>
      <div className="composer-shell">
        <button
          type="button"
          className="composer-tool"
          onClick={onOpenSources}
          aria-label="Open sources"
        >
          <Paperclip size={18} />
        </button>
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) onSubmit(event);
          }}
          placeholder="Ask about papers, verify a claim, or search recent literature"
          rows={1}
          disabled={disabled}
        />
        {loading ? (
          <button type="button" className="composer-send stop" onClick={onStop} aria-label="Stop">
            <Square size={16} />
          </button>
        ) : (
          <button type="submit" className="composer-send" disabled={!canSend} aria-label="Send">
            <Send size={17} />
          </button>
        )}
      </div>
      <p className="composer-hint">Enter to send, Shift+Enter for a new line</p>
    </form>
  );
}

function SourcePanel({
  open,
  documents,
  notice,
  setNotice,
  selectedTurn,
  onClose,
  onUpload,
  onLoadUrls,
  onLoadArxiv,
  onResizeStart,
}) {
  const [urlText, setUrlText] = useState("");
  const [arxivText, setArxivText] = useState("");
  const [tab, setTab] = useState("sources");
  const retrieved = selectedTurn?.retrieved_context || [];
  const graphState = selectedTurn?.graph_state || null;

  useEffect(() => {
    if (selectedTurn?.retrieved_context?.length) setTab("retrieved");
  }, [selectedTurn]);

  return (
    <aside
      className={cls("source-panel", open && "open")}
      aria-label="Evidence sources"
    >
      <button
        className="panel-resize-handle"
        type="button"
        aria-label="Resize research sources panel"
        onPointerDown={onResizeStart}
        title="Drag to resize"
      />
      <div className="source-header">
        <div>
          <h2>Evidence Sources</h2>
          <p>Attached to this research workspace</p>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="Close sources">
          <X size={18} />
        </button>
      </div>

      <div className="source-tabs" role="tablist">
        <button className={cls(tab === "sources" && "active")} onClick={() => setTab("sources")}>
          Sources
        </button>
        <button className={cls(tab === "retrieved" && "active")} onClick={() => setTab("retrieved")}>
          Retrieved
        </button>
        <button className={cls(tab === "graph" && "active")} onClick={() => setTab("graph")}>
          Graph
        </button>
      </div>

      <div className="source-body">
        {tab === "sources" ? (
          <>
            <label className="upload-control">
              <Upload size={18} />
              <span>Upload PDF, TXT, or Markdown</span>
              <input
                type="file"
                multiple
                accept=".pdf,.txt,.md,.markdown"
                onChange={(event) => onUpload(event.target.files)}
              />
            </label>

            <div className="source-card">
              <div className="source-card-title">
                <Globe2 size={16} />
                <span>Web pages</span>
              </div>
              <textarea
                value={urlText}
                onChange={(event) => setUrlText(event.target.value)}
                placeholder="https://example.com/research-page"
                rows={3}
              />
              <button
                onClick={() => {
                  onLoadUrls(urlText);
                  setUrlText("");
                }}
              >
                Load web pages
              </button>
            </div>

            <div className="source-card">
              <div className="source-card-title">
                <Archive size={16} />
                <span>arXiv</span>
              </div>
              <input
                value={arxivText}
                onChange={(event) => setArxivText(event.target.value)}
                placeholder="1706.03762 or paper title"
              />
              <button
                onClick={() => {
                  onLoadArxiv(arxivText);
                  setArxivText("");
                }}
              >
                Load arXiv paper
              </button>
            </div>

            <Notice notice={notice} onClear={() => setNotice(null)} />

            <section className="loaded-docs">
              <h3>Loaded documents</h3>
              {documents.length ? (
                <div className="doc-list">
                  {documents.map((title) => (
                    <div className="doc-item" key={title} title={title}>
                      <FileText size={15} />
                      <span>{title}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="empty-copy">No sources loaded in this session yet.</p>
              )}
            </section>
          </>
        ) : null}

        {tab === "retrieved" ? <RetrievedContext context={retrieved} /> : null}
        {tab === "graph" ? <GraphState state={graphState} /> : null}
      </div>
    </aside>
  );
}

function Notice({ notice, onClear }) {
  if (!notice) return null;
  return (
    <button className={cls("notice", notice.type)} onClick={onClear}>
      {notice.type === "loading" ? <Loader2 className="spin" size={15} /> : null}
      {notice.type === "success" ? <CheckCircle2 size={15} /> : null}
      {notice.type === "error" ? <AlertCircle size={15} /> : null}
      <span>{notice.text}</span>
    </button>
  );
}

function RetrievedContext({ context }) {
  if (!context?.length) {
    return (
      <div className="empty-panel">
        <Database size={20} />
        <p>Select an assistant turn with retrieved context to inspect its sources.</p>
      </div>
    );
  }
  return (
    <div className="retrieved-list">
      {context.map((item, index) => (
        <article className="retrieved-item" key={`${item.title}-${index}`}>
          <div className="retrieved-head">
            <strong>{item.title || "Retrieved source"}</strong>
            {item.url ? (
              <a href={item.url} target="_blank" rel="noreferrer">
                <Link size={13} />
                Open
              </a>
            ) : null}
          </div>
          <p>{item.content}</p>
        </article>
      ))}
    </div>
  );
}

function GraphState({ state }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const json = useMemo(() => {
    try {
      return JSON.stringify(state, null, 2);
    } catch {
      return "{\n  \"error\": \"Unable to serialize graph state\"\n}";
    }
  }, [state]);

  if (!state) {
    return (
      <div className="empty-panel">
        <Database size={20} />
        <p>Select an assistant turn to inspect graph state.</p>
      </div>
    );
  }
  const route = state.route || "unknown";
  const retrievedCount = state.retrieved_docs?.length || 0;

  async function handleCopy() {
    try {
      await copyTextToClipboard(json);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="graph-panel">
      <div className="graph-summary">
        <div>
          <span>Route</span>
          <strong>{route}</strong>
        </div>
        <div>
          <span>Retrieved</span>
          <strong>{retrievedCount}</strong>
        </div>
        <div>
          <span>Rewrites</span>
          <strong>{state.rewrite_count ?? 0}</strong>
        </div>
      </div>
      <button className="debug-toggle" onClick={() => setOpen((value) => !value)}>
        <span>Developer JSON</span>
        <ChevronDown size={16} className={open ? "open" : ""} />
      </button>
      {open ? (
        <div className="graph-json-shell">
          <div className="graph-json-toolbar">
            <span>Raw graph state</span>
            <button className="copy-json-button" onClick={handleCopy} type="button">
              <Copy size={14} />
              <span>{copied ? "Copied" : "Copy JSON"}</span>
            </button>
          </div>
          <pre tabIndex={0}>{json}</pre>
        </div>
      ) : null}
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
