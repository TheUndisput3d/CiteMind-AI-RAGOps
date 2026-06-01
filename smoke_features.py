import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

from backend.btw_handler import handle_btw
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.providers import get_chat_model, get_embeddings
from backend.rag_graph import build_graph, query_rewrite_node, web_search
from backend.vector_store import (
    add_paper,
    get_collection_name,
    list_papers,
    qdrant_client,
    search,
)


load_dotenv()


def record(results: list[dict], feature: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"{status} | {feature} | {detail}")
    results.append({"feature": feature, "ok": ok, "detail": detail})


def run() -> int:
    results: list[dict] = []
    session_id = f"smoke-{uuid.uuid4()}"
    collection = get_collection_name(session_id)

    try:
        record(
            results,
            "env keys present",
            all(os.getenv(k) for k in ["TAVILY_API_KEY", "QDRANT_URL"]),
            "TAVILY_API_KEY/QDRANT_URL checked; QDRANT_API_KEY optional locally",
        )

        try:
            llm = get_chat_model()
            started = time.time()
            response = llm.invoke("Reply with OK only.")
            record(
                results,
                "ollama chat model",
                "OK" in str(response.content).upper(),
                f"response={response.content!r}, elapsed={time.time() - started:.1f}s",
            )
        except Exception as exc:
            record(results, "ollama chat model", False, repr(exc))

        try:
            embeddings = get_embeddings()
            vector = embeddings.embed_query("dimension probe")
            record(results, "ollama embedding model", bool(vector), f"dimension={len(vector)}")
        except Exception as exc:
            record(results, "ollama embedding model", False, repr(exc))

        try:
            qdrant_client.get_collections()
            record(results, "qdrant connection", True, "collections endpoint reachable")
        except Exception as exc:
            record(results, "qdrant connection", False, repr(exc))

        try:
            tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            response = tavily.search("latest arxiv diffusion models", max_results=2)
            count = len(response.get("results", []))
            record(results, "tavily web search", count > 0, f"results={count}")
        except Exception as exc:
            record(results, "tavily web search", False, repr(exc))

        try:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
                handle.write(
                    "CiteMind AI smoke test document.\n"
                    "The Openclaw method improves retrieval precision with cached embeddings.\n"
                )
                text_path = handle.name
            docs = load_document(text_path)
            Path(text_path).unlink(missing_ok=True)
            record(results, "txt document loader", bool(docs), f"chunks={len(docs)}")

            add_paper(docs, session_id)
            titles = list_papers(session_id)
            retrieved = search("What improves retrieval precision?", session_id=session_id, k=2)
            ok = bool(titles) and bool(retrieved)
            record(
                results,
                "qdrant add/list/search",
                ok,
                f"titles={len(titles)}, retrieved={len(retrieved)}",
            )
        except Exception as exc:
            record(results, "qdrant add/list/search", False, repr(exc))

        try:
            docs = load_webpage("https://example.com")
            record(results, "webpage loader", bool(docs), f"chunks={len(docs)}")
        except Exception as exc:
            record(results, "webpage loader", False, repr(exc))

        try:
            docs = load_arxiv("1706.03762")
            record(results, "arxiv loader", bool(docs), f"chunks={len(docs)}")
        except Exception as exc:
            record(results, "arxiv loader", False, repr(exc))

        try:
            chunks = []
            for index, chunk in enumerate(handle_btw("What is softmax?")):
                chunks.append(chunk)
                if index >= 10:
                    break
            record(results, "/btw direct path", bool("".join(chunks).strip()), "stream produced text")
        except Exception as exc:
            record(results, "/btw direct path", False, repr(exc))

        try:
            chunks = []
            for index, chunk in enumerate(handle_btw("What is the latest NVIDIA GPU news today?")):
                chunks.append(chunk)
                if index >= 10:
                    break
            record(results, "/btw web-search path", bool("".join(chunks).strip()), "stream produced text")
        except Exception as exc:
            record(results, "/btw web-search path", False, repr(exc))

        try:
            tool_result = web_search.invoke(
                {
                    "optimized_query": "latest AI research news",
                    "max_results": 2,
                    "current_docs": [],
                    "tool_call_id": "smoke-tool-call",
                }
            )
            record(
                results,
                "rag web_search tool return",
                tool_result is not None,
                f"type={type(tool_result).__name__}",
            )
        except Exception as exc:
            record(results, "rag web_search tool return", False, repr(exc))

        try:
            graph = build_graph(db_path="smoke_checkpoints.db")
            thread_id = f"smoke-thread-{uuid.uuid4()}"
            input_state = {
                "messages": [HumanMessage(content="What is softmax?")],
                "session_id": session_id,
                "query": "What is softmax?",
                "route": None,
                "retrieved_docs": [],
                "retrieval_attempts": 0,
                "claim_verdict": None,
                "claim_source": None,
                "superseding_papers": [],
                "answer": None,
                "is_relevant": None,
                "rewrite_count": 0,
            }
            final_state = graph.invoke(
                input_state,
                {"configurable": {"thread_id": thread_id}},
            )
            record(
                results,
                "langgraph direct-answer flow",
                bool(final_state.get("answer")),
                f"route={final_state.get('route')}",
            )
        except Exception as exc:
            record(results, "langgraph direct-answer flow", False, repr(exc))

        try:
            graph = build_graph(db_path="smoke_checkpoints.db")
            thread_id = f"smoke-retrieve-{uuid.uuid4()}"
            input_state = {
                "messages": [
                    HumanMessage(
                        content="According to the uploaded smoke test document, what improves retrieval precision?"
                    )
                ],
                "session_id": session_id,
                "query": "According to the uploaded smoke test document, what improves retrieval precision?",
                "route": None,
                "retrieved_docs": [],
                "retrieval_attempts": 0,
                "claim_verdict": None,
                "claim_source": None,
                "superseding_papers": [],
                "answer": None,
                "is_relevant": None,
                "rewrite_count": 0,
            }
            final_state = graph.invoke(
                input_state,
                {"configurable": {"thread_id": thread_id}},
            )
            answer = final_state.get("answer") or ""
            ok = final_state.get("route") == "retrieve" and bool(final_state.get("retrieved_docs")) and bool(answer)
            record(
                results,
                "langgraph retrieve flow",
                ok,
                f"route={final_state.get('route')}, docs={len(final_state.get('retrieved_docs') or [])}",
            )
        except Exception as exc:
            record(results, "langgraph retrieve flow", False, repr(exc))

        try:
            graph = build_graph(db_path="smoke_checkpoints.db")
            thread_id = f"smoke-web-{uuid.uuid4()}"
            input_state = {
                "messages": [HumanMessage(content="What are the latest developments in RAG this week?")],
                "session_id": f"empty-{session_id}",
                "query": "What are the latest developments in RAG this week?",
                "route": None,
                "retrieved_docs": [],
                "retrieval_attempts": 0,
                "claim_verdict": None,
                "claim_source": None,
                "superseding_papers": [],
                "answer": None,
                "is_relevant": None,
                "rewrite_count": 0,
            }
            final_state = graph.invoke(
                input_state,
                {"configurable": {"thread_id": thread_id}},
            )
            docs = final_state.get("retrieved_docs") or []
            ok = final_state.get("route") == "retrieve" and bool(docs) and any(
                doc.metadata.get("url") for doc in docs
            )
            record(
                results,
                "langgraph live-web retrieve flow",
                ok,
                f"route={final_state.get('route')}, docs={len(docs)}",
            )
        except Exception as exc:
            record(results, "langgraph live-web retrieve flow", False, repr(exc))

        try:
            graph = build_graph(db_path="smoke_checkpoints.db")
            thread_id = f"smoke-verify-{uuid.uuid4()}"
            claim = "Verify the claim that recurrent neural networks are the best architecture for machine translation."
            input_state = {
                "messages": [HumanMessage(content=claim)],
                "session_id": session_id,
                "query": claim,
                "route": None,
                "retrieved_docs": [],
                "retrieval_attempts": 0,
                "claim_verdict": None,
                "claim_source": None,
                "superseding_papers": [],
                "answer": None,
                "is_relevant": None,
                "rewrite_count": 0,
            }
            final_state = graph.invoke(
                input_state,
                {"configurable": {"thread_id": thread_id}},
            )
            ok = final_state.get("route") == "verify_claim" and bool(final_state.get("claim_verdict"))
            record(
                results,
                "langgraph verify-claim flow",
                ok,
                f"route={final_state.get('route')}, papers={len(final_state.get('superseding_papers') or [])}",
            )
        except Exception as exc:
            record(results, "langgraph verify-claim flow", False, repr(exc))

        try:
            rewritten = query_rewrite_node(
                {
                    "query": "unclear retrieval query",
                    "rewrite_count": 0,
                    "messages": [HumanMessage(content="unclear retrieval query")],
                    "retrieved_docs": [],
                    "retrieval_attempts": 0,
                }
            )
            ok = rewritten.get("rewrite_count") == 1 and bool(rewritten.get("query"))
            record(results, "query rewrite node", ok, f"query={rewritten.get('query')!r}")
        except Exception as exc:
            record(results, "query rewrite node", False, repr(exc))

    finally:
        try:
            if qdrant_client.collection_exists(collection):
                qdrant_client.delete_collection(collection)
        except Exception:
            pass
        Path("smoke_checkpoints.db").unlink(missing_ok=True)
        Path("smoke_checkpoints.db-shm").unlink(missing_ok=True)
        Path("smoke_checkpoints.db-wal").unlink(missing_ok=True)

    print("\nJSON_SUMMARY")
    print(json.dumps(results, indent=2))
    return 0 if all(item["ok"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(run())
