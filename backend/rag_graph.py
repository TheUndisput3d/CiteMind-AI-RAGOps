import os
import sqlite3
import warnings
from typing import Annotated

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field
from tavily import TavilyClient

from backend.models import (
    ClaimVerificationResult,
    RelevancyDecision,
    RetrievalSourceDecision,
    RouterDecision,
)
from backend.providers import get_chat_model, get_structured_output, is_ollama
from backend.reranker import RERANKER_FINAL_K, rerank_documents
from backend.vector_store import list_papers, search as vs_search

load_dotenv()

llm = get_chat_model("gpt-5.4-mini")


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(MessagesState):
    session_id: str
    query: str
    route: str | None
    retrieval_source: str | None
    retrieved_docs: list[Document]
    retrieval_attempts: int
    claim_verdict: str | None
    claim_source: str | None
    superseding_papers: list[dict] | None
    answer: str | None
    is_relevant: bool | None
    rewrite_count: int


def _format_recent_messages(messages: list, limit: int = 8) -> str:
    recent = []
    for message in messages[-limit:]:
        if isinstance(message, HumanMessage):
            role = "User"
        elif isinstance(message, AIMessage):
            role = "Assistant"
        else:
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        content = " ".join(content.split())
        if content:
            recent.append(f"{role}: {content[:800]}")
    return "\n".join(recent) if recent else "No prior conversation."


# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a routing assistant for a research paper Q&A system. "
        "Available sources in this active session:\n{loaded_sources}\n\n"
        "Recent conversation:\n{conversation_history}\n\n"
        "Classify the latest user query into exactly one of three categories:\n\n"
        "  retrieve — Use this for TWO types of questions:\n"
        "    (a) Questions about the content of uploaded research papers "
        "or any loaded source in the active session "
        "(e.g. methods, results, conclusions, authors, CV details, loaded web-page content).\n"
        "    (b) Questions that require live or current information that cannot be "
        "answered from general knowledge alone — such as current events, today's weather, "
        "live prices, recent news, or anything where the answer changes over time "
        "(e.g. 'Who is the current president?', 'What is the price of gold today?', "
        "'What is the weather in Delhi?').\n"
        "  verify_claim — The user wants to check whether a specific claim or finding "
        "from a paper is still accurate or has been superseded.\n"
        "  direct_answer — A stable general knowledge question answerable from training data "
        "with no retrieval needed (e.g. 'What is softmax?', 'Who invented the transformer?', "
        "'Explain backpropagation.').\n\n"
        "Resolve short follow-ups using the recent conversation. "
        "If active-session sources are listed and the user asks an underspecified question "
        "that could refer to those sources, route to retrieve. "
        "Use direct_answer only when the question is self-contained and does not need sources.\n\n"
        "Examples when active-session sources are listed:\n"
        "- 'Who is this person?' -> retrieve\n"
        "- 'What is this person's educational qualification?' -> retrieve\n"
        "- 'What does this document say about experience?' -> retrieve\n"
        "- 'Summarize it' -> retrieve\n"
        "- 'What is softmax?' -> direct_answer\n\n"
        "Return only the route field.",
    ),
    ("human", "{query}"),
])

router_chain = ROUTER_PROMPT | get_structured_output(llm, RouterDecision)

_SOURCE_REFERENCE_PHRASES = (
    "uploaded paper",
    "uploaded document",
    "loaded paper",
    "loaded document",
    "provided paper",
    "provided document",
    "attached paper",
    "attached document",
    "the paper",
    "the document",
    "these papers",
    "these documents",
    "both papers",
    "two papers",
    "the sources",
    "these sources",
    "source-grounded",
    "using only the",
)


def _references_loaded_sources(query: str, loaded_titles: list[str]) -> bool:
    """Recognize explicit source references before asking the probabilistic router."""
    if not loaded_titles:
        return False

    normalized_query = " ".join(query.casefold().split())
    if any(phrase in normalized_query for phrase in _SOURCE_REFERENCE_PHRASES):
        return True

    # A title mention is also an explicit request to use session evidence. Match a
    # distinctive title prefix so long filenames and punctuation remain usable.
    for title in loaded_titles:
        normalized_title = " ".join(title.casefold().replace("_", " ").replace("-", " ").split())
        if normalized_title and normalized_title in normalized_query:
            return True
        significant_words = [word for word in normalized_title.split() if len(word) > 3]
        if len(significant_words) >= 2:
            prefix = " ".join(significant_words[:3])
            if prefix in normalized_query:
                return True
    return False


def router_node(state: RAGState) -> dict:
    query = state["messages"][-1].content
    conversation_history = _format_recent_messages(state.get("messages", [])[:-1])
    loaded_titles = list_papers(state["session_id"])
    loaded_sources = (
        "\n".join(f"- {title}" for title in loaded_titles)
        if loaded_titles
        else "No loaded sources in this session."
    )
    if _references_loaded_sources(query, loaded_titles):
        return {"route": "retrieve"}

    decision: RouterDecision = router_chain.invoke({
        "query": query,
        "conversation_history": conversation_history,
        "loaded_sources": loaded_sources,
    })
    return {"route": decision.route}


# ── Tool schemas ──────────────────────────────────────────────────────────────

class RetrieverInput(BaseModel):
    query: str = Field(description="Semantic query to search research paper chunks")
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve")


class WebSearchInput(BaseModel):
    optimized_query: str = Field(description="Query rewritten and optimized for web search")
    max_results: int = Field(default=3, ge=1, le=10, description="Number of web results to return")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query: str,
    k: int,
    session_id: Annotated[str, InjectedState("session_id")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the uploaded research paper vector store for relevant passages."""
    docs = vs_search(query=query, session_id=session_id, k=k)
    if not docs:
        return [ToolMessage(content="No relevant documents found in the vector store.", tool_call_id=tool_call_id)]
    summary = f"Retrieved {len(docs)} chunk(s) from the vector store."
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + docs}),
    ]


@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query: str,
    max_results: int,
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the web for current or supplementary information using Tavily."""
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        results = client.search(optimized_query, max_results=max_results)
    except Exception as exc:
        raise RuntimeError(
            "Live web search is temporarily unavailable. Loaded papers and URLs can still be used."
        ) from exc
    if not results.get("results"):
        return [ToolMessage(content="No web results found.", tool_call_id=tool_call_id)]
    web_docs = [
        Document(
            page_content=r["content"],
            metadata={"url": r["url"], "title": r.get("title", "Web Result")},
        )
        for r in results["results"]
    ]
    summary = f"Found {len(web_docs)} web result(s) for: {optimized_query}"
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + web_docs}),
    ]


def _search_web_documents(query: str, max_results: int = 3) -> list[Document]:
    """Retrieve web results for provider modes that do not support tool calls."""
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        results = client.search(query, max_results=max_results)
    except Exception as exc:
        raise RuntimeError(
            "Web search is temporarily unavailable. If you already loaded a URL, ask "
            "about the loaded link or uploaded source instead of asking for live web search."
        ) from exc
    return [
        Document(
            page_content=result["content"],
            metadata={"url": result["url"], "title": result.get("title", "Web Result")},
        )
        for result in results.get("results", [])
    ]


# ── Retrieval agent singletons ────────────────────────────────────────────────

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
retrieval_llm = (
    llm.bind_tools(RETRIEVAL_TOOLS)
    if is_ollama()
    else llm.bind_tools(RETRIEVAL_TOOLS, parallel_tool_calls=False)
)
base_tool_node = ToolNode(RETRIEVAL_TOOLS)

RETRIEVE_SYSTEM = (
    "You are a research assistant gathering context to answer a user's question about research papers.\n\n"
    "You have two tools available and full control over how you use them:\n\n"
    "1. retrieve_from_vectorstore — searches the uploaded paper collection.\n"
    "   You decide:\n"
    "   - query: the semantic search query (phrase it to best match relevant paper chunks)\n"
    "   - k: how many chunks to retrieve (1–10; use more for broad questions, fewer for specific ones)\n\n"
    "2. web_search — searches the live web via Tavily.\n"
    "   You decide:\n"
    "   - optimized_query: rewrite the user's question as a concise, keyword-rich web search query\n"
    "   - max_results: how many results to fetch (1–10)\n\n"
    "Choose the right source based on the question:\n"
    "- Questions about the uploaded papers → use retrieve_from_vectorstore\n"
    "- Questions about current events, recent developments, or supplementary information → use web_search\n"
    "- Call only one tool per turn.\n\n"
    "Do NOT produce a final answer. Only call tools to collect context."
)

SOURCE_SELECTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Choose the best retrieval source for this research assistant.\n\n"
        "Available sources in this active session:\n{loaded_sources}\n\n"
        "Recent conversation:\n{conversation_history}\n\n"
        "Use vectorstore when the user is asking about the active session's loaded sources, "
        "uploaded papers, loaded URLs, attached files, provided links, document contents, "
        "or anything that should be answered from the sources listed above.\n\n"
        "Use web_search when the user is asking for outside information beyond the loaded "
        "sources, broad literature discovery, or current/recent information that should be "
        "looked up live.\n\n"
        "If loaded sources are available and the query refers to a provided, loaded, attached, "
        "or uploaded source, choose vectorstore.\n\n"
        "Return only the source field.",
    ),
    ("human", "{query}"),
])

source_selector_chain = SOURCE_SELECTION_PROMPT | get_structured_output(
    llm, RetrievalSourceDecision
)

# ── Relevancy check ───────────────────────────────────────────────────────────

RELEVANCY_CHECK_SYSTEM = (
    "You are evaluating whether retrieved document chunks are relevant enough "
    "to answer a user's question about research papers.\n\n"
    "Return is_relevant=true if the chunks contain information that meaningfully "
    "addresses the question — even partially. "
    "Return is_relevant=false only if the chunks are clearly off-topic or contain "
    "no useful information.\n\nBe lenient: if there is any substantive overlap, return true."
)

relevancy_llm = get_structured_output(llm, RelevancyDecision)

QUERY_REWRITE_SYSTEM = (
    "You are a query rewriting assistant for a research paper retrieval system. "
    "The previous query failed to retrieve relevant document chunks. "
    "Rewrite the query using more specific or alternative terminology, "
    "domain-specific keywords, or a narrower sub-question.\n\n"
    "Return ONLY the rewritten query as plain text. No explanation, no preamble."
)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def agent_node(state: RAGState) -> dict:
    current_attempts = state.get("retrieval_attempts", 0)
    if current_attempts > 0:
        # A tool has just returned. Let the graph evaluate/generate instead of
        # asking the agent LLM again, which can produce conversational filler.
        return {}

    loaded_titles = list_papers(state["session_id"])
    loaded_sources = (
        "\n".join(f"- {title}" for title in loaded_titles)
        if loaded_titles
        else "No loaded sources in this session."
    )
    source = state.get("retrieval_source")
    if source not in ("vectorstore", "web_search"):
        source_decision: RetrievalSourceDecision = source_selector_chain.invoke(
            {
                "query": state["query"],
                "loaded_sources": loaded_sources,
                "conversation_history": _format_recent_messages(
                    state.get("messages", [])[:-1]
                ),
            }
        )
        source = source_decision.source

    if source == "vectorstore":
        search_query = state["query"]
        if len(search_query.split()) <= 3:
            search_query = f"{_format_recent_messages(state.get('messages', []), limit=6)}\nLatest question: {state['query']}"
        candidates = vs_search(query=search_query, session_id=state["session_id"])
        docs = [
            item.document
            for item in rerank_documents(
                query=search_query,
                docs=candidates,
                top_n=RERANKER_FINAL_K,
            )
        ]
        return {
            "retrieved_docs": docs,
            "retrieval_attempts": current_attempts + 1,
            "retrieval_source": "vectorstore",
        }

    docs = _search_web_documents(state["query"], max_results=5)
    return {
        "retrieved_docs": docs,
        "retrieval_attempts": current_attempts + 1,
        "retrieval_source": "web_search",
    }


def relevancy_check_node(state: RAGState) -> dict:
    query = state["query"]
    docs = state.get("retrieved_docs") or []
    doc_snippets = "\n\n---\n\n".join(doc.page_content[:300] for doc in docs[:3])
    if not doc_snippets:
        return {"is_relevant": False}
    prompt = (
        f"Recent conversation:\n{_format_recent_messages(state.get('messages', [])[:-1])}\n\n"
        f"Question: {query}\n\nRetrieved chunks:\n{doc_snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )
    decision: RelevancyDecision = relevancy_llm.invoke([
        {"role": "system", "content": RELEVANCY_CHECK_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    return {"is_relevant": decision.is_relevant}


def query_rewrite_node(state: RAGState) -> dict:
    original_query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)
    response = llm.invoke([
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{_format_recent_messages(state.get('messages', [])[:-1])}\n\n"
                f"Original query: {original_query}\n\nWrite an improved search query."
            ),
        },
    ])
    rewritten = response.content.strip()
    return {
        "messages": [HumanMessage(content=rewritten)],
        "query": rewritten,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "rewrite_count": rewrite_count + 1,
        "is_relevant": None,
        "retrieval_source": state.get("retrieval_source"),
    }


CLAIM_ANALYSIS_PROMPT = (
    "You are a research fact-checker. Given a claim from a research paper and "
    "a set of recent web and arXiv search results, determine:\n"
    "1. Has this claim been superseded, significantly challenged, or updated by more recent work?\n"
    "2. Identify up to 3 papers from the provided results that supersede or update the claim.\n\n"
    "Rules:\n"
    "- Use ONLY titles and URLs that appear verbatim in the provided search results.\n"
    "- Prefer arXiv paper links (arxiv.org) over general web links when available.\n"
    "- For each superseding paper, write one sentence explaining how it supersedes the claim.\n"
    "- If the claim still holds, set is_superseded=false and return an empty superseding_papers list.\n"
    "- verdict_summary should be 1-2 sentences suitable for display to the user."
)

verification_llm = get_structured_output(llm, ClaimVerificationResult)


def verify_claim_node(state: RAGState) -> dict:
    claim = state["messages"][-1].content
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # General web search for recent work superseding the claim
    general_results = tavily_client.search(
        f"recent research superseding: {claim[:200]}",
        max_results=5,
    ).get("results", [])

    # arXiv-targeted search via web to get paper titles and links
    arxiv_results = tavily_client.search(
        f"site:arxiv.org {claim[:200]}",
        max_results=5,
    ).get("results", [])

    # Build context block
    lines = ["=== General Web Search Results ==="]
    for r in general_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    lines.append("=== arXiv Paper Search Results ===")
    for r in arxiv_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    context = "\n".join(lines)

    prompt = (
        f"{CLAIM_ANALYSIS_PROMPT}\n\n"
        f"Claim to verify:\n{claim}\n\n"
        f"Search Results:\n{context}"
    )
    result: ClaimVerificationResult = verification_llm.invoke([
        {"role": "user", "content": prompt}
    ])

    papers_dicts = [p.model_dump() for p in result.superseding_papers[:3]]
    return {
        "claim_verdict": result.verdict_summary,
        "claim_source": papers_dicts[0]["url"] if papers_dicts else None,
        "superseding_papers": papers_dicts,
    }


def generate_answer_node(state: RAGState) -> dict:
    route = state.get("route")
    query = state["query"]
    conversation_history = _format_recent_messages(state.get("messages", [])[:-1])

    if route == "retrieve":
        if state.get("is_relevant") is False and state.get("rewrite_count", 0) >= 1:
            if state.get("retrieval_source") == "web_search":
                answer = (
                    "I wasn't able to find relevant current web results to answer your question. "
                    "Try a narrower research query or retry in a moment."
                )
            else:
                answer = (
                    "I wasn't able to find relevant information in the uploaded papers "
                    "to answer your question. You may want to rephrase your question "
                    "or upload additional papers."
                )
        else:
            docs = state.get("retrieved_docs") or []
            if not docs:
                answer = "I don't know the answer."
            else:
                context_blocks = []
                for i, doc in enumerate(docs, start=1):
                    title = doc.metadata.get("title") or "Retrieved document"
                    url = doc.metadata.get("url")
                    source = f"Document: {title}"
                    if url:
                        source = f"{source}\nURL: {url}"
                    context_blocks.append(f"{source}\nExcerpt:\n{doc.page_content}")
                context = "\n\n---\n\n".join(context_blocks)
                # prompt = (
                #     "You are CiteMind AI, a research assistant. Answer the latest user question "
                #     "using only the retrieved context below and the recent conversation for pronoun "
                #     "or follow-up resolution.\n\n"
                #     "Style rules:\n"
                #     # "- Give the direct answer first.\n"
                #     "- Do not narrate the retrieval process.\n"
                #     "- Do not write phrases like 'Based on the provided sources', "
                #     "'According to Source 1', or 'Source 2 states'.\n"
                #     "- If useful, cite by real document title or URL, not by chunk number.\n"
                #     "- If the context does not contain enough information, say that plainly.\n\n"
                #     "The context may include uploaded documents, live web-search results, or loaded URLs. "
                #     "If web-search results are present, do not say you cannot search the web; "
                #     "treat the supplied web results as available context.\n\n"
                #     f"Recent conversation:\n{conversation_history}\n\n"
                #     f"Retrieved context:\n{context}\n\n"
                #     f"Latest question: {query}"
                # )

                prompt = (
                            "Answer the question using only the retrieved context below.\n"
                            "The context may include uploaded documents, live web-search results, or loaded URLs. "
                            "If web-search results are present, do not say you cannot search the web; "
                            "treat the supplied web results as available context. "
                            "Cite source titles or URLs when they help the user verify the answer.\n\n"
                            f"{context}\n\nQuestion: {query}"
                        )
                answer = llm.invoke([{"role": "user", "content": prompt}]).content

    elif route == "verify_claim":
        verdict = state.get("claim_verdict", "")
        papers = state.get("superseding_papers") or []
        claim_text = state["query"]
        if papers:
            papers_block = "\n\n".join(
                f"{i + 1}. **{p['title']}**\n   {p['summary']}\n   Link: {p['url']}"
                for i, p in enumerate(papers)
            )
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"**Superseding Papers:**\n\n{papers_block}\n\n"
                f"---\n"
                f"*You can load any of these papers into your knowledge base "
                f"to continue your research with the latest findings.*"
            )
        else:
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"*No papers directly superseding this claim were found in recent literature.*"
            )

    else:  # direct_answer
        prompt = (
            "You are CiteMind AI, an AI research assistant for papers, loaded web pages, "
            "arXiv imports, claim verification, and source-grounded research chat.\n\n"
            "Use the recent conversation to resolve short follow-ups such as 'yes', "
            "'your question', 'what about that', or pronouns. If the latest message is "
            "still ambiguous after reading the conversation, ask one concise clarifying question.\n\n"
            "For identity questions such as 'who are you' or 'what are you', answer as CiteMind AI, "
            "not as a generic language model.\n\n"
            "Answer naturally and briefly unless the user asks for detail.\n\n"
            f"Recent conversation:\n{conversation_history}\n\n"
            f"Latest question: {query}"
        )
        answer = llm.invoke([{"role": "user", "content": prompt}]).content

    return {"answer": answer, "messages": [AIMessage(content=answer)]}


# ── Graph ─────────────────────────────────────────────────────────────────────

MAX_RETRIEVAL_ATTEMPTS = 3


def route_query(state: RAGState) -> str:
    return state["route"]


def agent_routing(state: RAGState) -> str:
    # Always execute pending tool calls first — shortcutting here would leave
    # an AIMessage with tool_calls unmatched by ToolMessages in the checkpointer,
    # corrupting history for all future turns in the same session.
    tc = tools_condition(state)
    if tc == "tools":
        return "retrieval"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate_answer"
    return "relevancy_check"


def after_relevancy_routing(state: RAGState) -> str:
    if state.get("is_relevant", False):
        return "generate_answer"
    if state.get("rewrite_count", 0) < 1:
        return "query_rewrite"
    return "generate_answer"


def build_graph(db_path: str = "checkpoints.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(RAGState)
    graph.add_node("router", router_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("retrieval", base_tool_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("verify_claim", verify_claim_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_query,
        {
            "retrieve": "agent_node",
            "verify_claim": "verify_claim",
            "direct_answer": "generate_answer",
        },
    )

    graph.add_conditional_edges(
        "agent_node",
        agent_routing,
        {
            "retrieval": "retrieval",
            "relevancy_check": "relevancy_check",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("retrieval", "agent_node")

    graph.add_conditional_edges(
        "relevancy_check",
        after_relevancy_routing,
        {"query_rewrite": "query_rewrite", "generate_answer": "generate_answer"},
    )
    graph.add_edge("query_rewrite", "agent_node")

    graph.add_edge("verify_claim", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=checkpointer)
