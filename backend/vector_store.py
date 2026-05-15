import logging
import os

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, SparseVectorParams, VectorParams
from backend.providers import get_embeddings

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

# ── Singletons ────────────────────────────────────────────────────────────────

COLLECTION_PREFIX = os.getenv("QDRANT_COLLECTION_PREFIX", "citemind")
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
HYBRID_CANDIDATE_K = 24

base_embeddings = get_embeddings()
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
embedding_file_store = LocalFileStore("./embedding_cache/")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model,
    query_embedding_cache=True,
    key_encoder="blake2b",
)

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.getenv("QDRANT_API_KEY") or None,
    timeout=45,
)

_embedding_dimension: int | None = None


def get_embedding_dimension() -> int:
    """Read the configured embedding model's dimension once for Qdrant."""
    global _embedding_dimension
    if _embedding_dimension is None:
        _embedding_dimension = len(base_embeddings.embed_query("dimension probe"))
    return _embedding_dimension


def _collection_has_hybrid_schema(collection_name: str, expected_dimension: int) -> bool:
    params = qdrant_client.get_collection(collection_name).config.params
    vectors = params.vectors
    sparse_vectors = params.sparse_vectors or {}
    return (
        isinstance(vectors, dict)
        and DENSE_VECTOR_NAME in vectors
        and vectors[DENSE_VECTOR_NAME].size == expected_dimension
        and SPARSE_VECTOR_NAME in sparse_vectors
    )


# ── Collection ───────────────────────────────────────────────────────────────


def get_collection_name(session_id: str) -> str:
    return f"{COLLECTION_PREFIX}_{session_id.replace('-', '_')}"


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    collection_name = get_collection_name(session_id)
    expected_dimension = get_embedding_dimension()
    if qdrant_client.collection_exists(collection_name):
        if not _collection_has_hybrid_schema(collection_name, expected_dimension):
            logger.warning(
                "Recreating collection %s with the required hybrid dense+sparse schema.",
                collection_name,
            )
            qdrant_client.delete_collection(collection_name)

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=expected_dimension,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
            },
        )
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_embedding=sparse_embeddings,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    get_vectorstore(session_id).add_documents(docs)


def delete_session_collection(session_id: str) -> None:
    collection_name = get_collection_name(session_id)
    try:
        if qdrant_client.collection_exists(collection_name):
            qdrant_client.delete_collection(collection_name)
    except Exception as exc:
        logger.warning(
            "Unable to delete vector collection for session %s: %s",
            session_id,
            exc,
        )


def list_papers(session_id: str) -> list[str]:
    collection_name = get_collection_name(session_id)
    try:
        if not qdrant_client.collection_exists(collection_name):
            return []
    except Exception as exc:
        logger.warning(
            "Unable to list papers for session %s because Qdrant was unavailable: %s",
            session_id,
            exc,
        )
        return []
    seen: set[str] = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles


def search(query: str, session_id: str, k: int = HYBRID_CANDIDATE_K) -> list[Document]:
    try:
        return get_vectorstore(session_id).similarity_search(query, k=k)
    except Exception as exc:
        logger.warning(
            "Vector search failed for session %s and query %r: %s",
            session_id,
            query,
            exc,
        )
        return []
