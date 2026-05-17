import os
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

def _normalize_model_name(value: str) -> str:
    value = value.strip()
    if not value:
        return "BAAI/bge-reranker-base"
    if "/" not in value:
        return f"BAAI/{value}"
    return value


RERANKER_MODEL = _normalize_model_name(
    os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
)
RERANKER_MAX_DOC_CHARS = int(os.getenv("RERANKER_MAX_DOC_CHARS", "2500"))
RERANKER_FINAL_K = int(os.getenv("RERANKER_FINAL_K", "6"))

_cross_encoder: Any | None = None


@dataclass(frozen=True)
class RerankedDocument:
    document: Document
    score: float
    rerank_rank: int
    retrieval_candidate_rank: int


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


def rerank_documents(
    query: str,
    docs: list[Document],
    top_n: int = RERANKER_FINAL_K,
) -> list[RerankedDocument]:
    if not docs:
        return []

    model = _get_cross_encoder()
    pairs = [
        (query, doc.page_content[:RERANKER_MAX_DOC_CHARS])
        for doc in docs
    ]
    scores = model.predict(pairs)

    ranked: list[RerankedDocument] = []
    for candidate_rank, (doc, score) in enumerate(zip(docs, scores), start=1):
        ranked.append(
            RerankedDocument(
                document=doc,
                score=float(score),
                rerank_rank=0,
                retrieval_candidate_rank=candidate_rank,
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    selected: list[RerankedDocument] = []
    for rerank_rank, item in enumerate(ranked[:top_n], start=1):
        doc = Document(
            page_content=item.document.page_content,
            metadata={
                **item.document.metadata,
                "rerank_score": item.score,
                "rerank_rank": rerank_rank,
                "retrieval_candidate_rank": item.retrieval_candidate_rank,
            },
        )
        selected.append(
            RerankedDocument(
                document=doc,
                score=item.score,
                rerank_rank=rerank_rank,
                retrieval_candidate_rank=item.retrieval_candidate_rank,
            )
        )
    return selected
