import json
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from backend.btw_handler import handle_btw
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.providers import get_chat_model
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, delete_session_collection, list_papers

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
SESSIONS_FILE = ROOT_DIR / "sessions.json"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
PROCESS_STARTED_AT = time.time()

graph = build_graph()
rename_llm = get_chat_model("gpt-5-mini")

app = FastAPI(title="CiteMind AI API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class UrlLoadRequest(BaseModel):
    urls: list[str]


class ArxivLoadRequest(BaseModel):
    query: str


def load_sessions() -> dict[str, dict[str, Any]]:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict[str, dict[str, Any]]) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def create_session() -> dict[str, Any]:
    sessions = load_sessions()
    sid = str(uuid.uuid4())
    session = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    sessions[sid] = session
    save_sessions(sessions)
    return session


def ensure_session(session_id: str) -> dict[str, Any]:
    sessions = load_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


def maybe_rename_session(session_id: str, first_message: str) -> None:
    sessions = load_sessions()
    session = sessions.get(session_id)
    if not session or session.get("is_named"):
        return
    try:
        response = rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        name = response.content.strip() or "New Session"
    except Exception:
        name = "New Session"
    session["name"] = name
    session["is_named"] = True
    save_sessions(sessions)


def serialize_state(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if key == "messages":
            out[key] = [
                {
                    "type": type(message).__name__,
                    "content": (
                        message.content[:300]
                        if isinstance(message.content, str)
                        else repr(message.content)[:300]
                    ),
                }
                for message in (value or [])
            ]
        elif key == "retrieved_docs":
            out[key] = [
                {"content": doc.page_content[:300], "metadata": doc.metadata}
                for doc in (value or [])
            ]
        else:
            out[key] = value
    return out


def serialize_retrieved_context(docs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "title": doc.metadata.get("title", "Retrieved document"),
            "url": doc.metadata.get("url"),
            "content": doc.page_content[:1500],
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


def load_session_chats(session_id: str) -> list[dict[str, Any]]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        messages: list[dict[str, Any]] = []
        turn = 0
        for message in state.values.get("messages", []):
            type_name = type(message).__name__
            content = message.content if isinstance(message.content, str) else str(message.content)
            if type_name == "HumanMessage":
                messages.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                turn += 1
                graph_state = serialize_state(state.values) if turn == 1 else {}
                retrieved_context = (
                    serialize_retrieved_context(state.values.get("retrieved_docs") or [])
                    if graph_state
                    else []
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "turn": turn,
                        "graph_state": graph_state,
                        "retrieved_context": retrieved_context,
                    }
                )
        return messages
    except Exception:
        return []


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False, response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    """Expose lightweight Prometheus metrics without external middleware."""
    session_count = len(load_sessions())
    uptime_seconds = max(0.0, time.time() - PROCESS_STARTED_AT)
    payload = (
        "# HELP citemind_up Whether the CiteMind API process is running.\n"
        "# TYPE citemind_up gauge\n"
        "citemind_up 1\n"
        "# HELP citemind_sessions Current number of persisted research sessions.\n"
        "# TYPE citemind_sessions gauge\n"
        f"citemind_sessions {session_count}\n"
        "# HELP citemind_process_uptime_seconds API process uptime in seconds.\n"
        "# TYPE citemind_process_uptime_seconds gauge\n"
        f"citemind_process_uptime_seconds {uptime_seconds:.3f}\n"
    )
    return PlainTextResponse(payload, media_type="text/plain; version=0.0.4")


@app.get("/api/sessions")
def get_sessions() -> dict[str, Any]:
    sessions = sorted(
        load_sessions().values(),
        key=lambda session: session["created_at"],
        reverse=True,
    )
    if not sessions:
        sessions = [create_session()]
    return {"sessions": sessions}


@app.post("/api/sessions")
def post_session() -> dict[str, Any]:
    return {"session": create_session()}


@app.delete("/api/sessions")
def delete_all_sessions() -> dict[str, Any]:
    sessions = load_sessions()
    deleted = sorted(
        sessions.values(),
        key=lambda session: session["created_at"],
        reverse=True,
    )

    for session_id in list(sessions):
        delete_session_collection(session_id)

    save_sessions({})
    return {
        "deleted_count": len(deleted),
        "deleted": deleted,
        "sessions": [],
        "next_session_id": None,
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = ensure_session(session_id)
    return {
        "session": session,
        "messages": load_session_chats(session_id),
        "documents": list_papers(session_id),
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    sessions = load_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    deleted = sessions.pop(session_id)
    save_sessions(sessions)
    delete_session_collection(session_id)

    remaining = sorted(
        sessions.values(),
        key=lambda session: session["created_at"],
        reverse=True,
    )
    return {
        "deleted": deleted,
        "sessions": remaining,
        "next_session_id": remaining[0]["id"] if remaining else None,
    }


@app.get("/api/sessions/{session_id}/documents")
def get_documents(session_id: str) -> dict[str, Any]:
    ensure_session(session_id)
    return {"documents": list_papers(session_id)}


@app.post("/api/sessions/{session_id}/documents/files")
async def upload_files(
    session_id: str,
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    ensure_session(session_id)
    loaded: list[str] = []
    failed: list[dict[str, str]] = []
    for upload in files:
        suffix = Path(upload.filename or "").suffix
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await upload.read())
                tmp_path = tmp.name
            docs = load_document(tmp_path)
            title = Path(upload.filename or tmp_path).stem
            for doc in docs:
                doc.metadata["title"] = title
            add_paper(docs, session_id)
            loaded.append(title)
        except Exception as exc:
            failed.append({"name": upload.filename or "file", "error": str(exc)})
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
    return {"loaded": loaded, "failed": failed, "documents": list_papers(session_id)}


@app.post("/api/sessions/{session_id}/documents/urls")
def load_urls(session_id: str, request: UrlLoadRequest) -> dict[str, Any]:
    ensure_session(session_id)
    loaded: list[str] = []
    failed: list[dict[str, str]] = []
    for url in [url.strip() for url in request.urls if url.strip()]:
        try:
            docs = load_webpage(url)
            add_paper(docs, session_id)
            loaded.append(docs[0].metadata.get("title") if docs else url)
        except Exception as exc:
            failed.append({"name": url, "error": str(exc)})
    return {"loaded": loaded, "failed": failed, "documents": list_papers(session_id)}


@app.post("/api/sessions/{session_id}/documents/arxiv")
def load_arxiv_endpoint(session_id: str, request: ArxivLoadRequest) -> dict[str, Any]:
    ensure_session(session_id)
    try:
        docs = load_arxiv(request.query.strip())
        add_paper(docs, session_id)
        title = docs[0].metadata.get("title") if docs else request.query
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"loaded": [title], "failed": [], "documents": list_papers(session_id)}


@app.post("/api/sessions/{session_id}/chat")
def chat(session_id: str, request: ChatRequest) -> StreamingResponse:
    ensure_session(session_id)
    prompt = request.message.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Message is required")

    def event_stream():
        is_btw = prompt.lower().startswith("/btw")
        if is_btw:
            query = prompt[4:].strip()
            if not query:
                yield sse("error", {"message": "Please add a question after /btw."})
                return
            response_text = ""
            try:
                yield sse("start", {"mode": "btw"})
                for token in handle_btw(query):
                    response_text += token
                    yield sse("token", {"content": token})
                yield sse(
                    "final",
                    {
                        "answer": response_text,
                        "graph_state": {},
                        "retrieved_context": [],
                        "saved": False,
                    },
                )
            except Exception as exc:
                yield sse("error", {"message": str(exc)})
            return

        existing_messages = load_session_chats(session_id)
        if not existing_messages:
            maybe_rename_session(session_id, prompt)

        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": session_id,
            "query": prompt,
            "route": None,
            "retrieval_source": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": session_id}}
        response_text = ""
        try:
            yield sse("start", {"mode": "graph"})
            for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
                if (
                    metadata.get("langgraph_node") == "generate_answer"
                    and hasattr(chunk, "content")
                    and chunk.content
                ):
                    response_text += chunk.content
                    yield sse("token", {"content": chunk.content})
            final_values = graph.get_state(config).values
            answer = final_values.get("answer") or response_text or "No response generated."
            if answer != response_text:
                yield sse("replace", {"content": answer})
            yield sse(
                "final",
                {
                    "answer": answer,
                    "graph_state": serialize_state(final_values),
                    "retrieved_context": serialize_retrieved_context(
                        final_values.get("retrieved_docs") or []
                    ),
                    "session": load_sessions().get(session_id),
                    "saved": True,
                },
            )
        except Exception as exc:
            yield sse("error", {"message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{path:path}")
def frontend(path: str):
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "CiteMind AI API is running. Build frontend assets to serve the UI."}
