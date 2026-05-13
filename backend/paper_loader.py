import re
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from urllib.error import URLError
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)


def _stamp_title(docs: list[Document], title: str) -> list[Document]:
    for doc in docs:
        doc.metadata["title"] = title
    return docs


def load_pdf(file_path: str) -> list[Document]:
    docs = PyMuPDFLoader(file_path).load()
    return _stamp_title(_splitter.split_documents(docs), Path(file_path).stem)


def load_text(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(_splitter.split_documents(docs), Path(file_path).stem)


def load_markdown(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(_md_splitter.split_documents(docs), Path(file_path).stem)


def load_webpage(url: str) -> list[Document]:
    docs = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
    title = (docs[0].metadata.get("title") or url) if docs else url
    return _stamp_title(_splitter.split_documents(docs), title)


def _extract_arxiv_id(query: str) -> str | None:
    """Return bare ArXiv ID (no version suffix) if one appears in the query."""
    m = _ARXIV_ID_RE.search(query)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    return None


def _arxiv_api_metadata(arxiv_id: str) -> tuple[str, list[str]]:
    """Fetch paper title and authors by ID from the ArXiv Atom API."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        xml = resp.read().decode()
    root = ET.fromstring(xml)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        return arxiv_id, []
    title = " ".join((entry.findtext("atom:title", default=arxiv_id, namespaces=ns) or arxiv_id).split())
    authors = [
        " ".join((author.findtext("atom:name", default="", namespaces=ns) or "").split())
        for author in entry.findall("atom:author", ns)
    ]
    return title, [author for author in authors if author]


def _arxiv_search(query: str) -> str:
    """Search ArXiv Atom API by title phrase and return the top result's bare paper ID."""
    phrase = query.strip('"')
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = f"https://export.arxiv.org/api/query?search_query={search_query}&max_results=1&sortBy=relevance"
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read().decode()
    m = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>", xml)
    if not m:
        raise ValueError(f"No ArXiv paper found for: {query}")
    return re.sub(r"v\d+$", "", m.group(1))


def _download_arxiv_pdf(arxiv_id: str) -> bytes:
    """Download an ArXiv PDF, trying the primary host and a fallback host."""
    candidates = [
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        f"https://arxiv.org/pdf/{arxiv_id}",
        f"https://export.arxiv.org/pdf/{arxiv_id}.pdf",
        f"https://export.arxiv.org/pdf/{arxiv_id}",
    ]
    last_error: Exception | None = None
    for pdf_url in candidates:
        try:
            with urllib.request.urlopen(pdf_url, timeout=60) as resp:
                return resp.read()
        except URLError as exc:
            last_error = exc
    raise RuntimeError(
        f"Could not download ArXiv PDF for {arxiv_id}. "
        "Try again later or use the direct PDF/abs URL."
    ) from last_error


def _load_arxiv_by_id(arxiv_id: str) -> list[Document]:
    """Download and chunk an ArXiv paper PDF by its bare ID."""
    api_title, authors = _arxiv_api_metadata(arxiv_id)
    pdf_bytes = _download_arxiv_pdf(arxiv_id)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        docs = PyMuPDFLoader(tmp_path).load()
        if not docs:
            raise ValueError(f"Could not load PDF for ArXiv ID: {arxiv_id}")
        title = (docs[0].metadata.get("title") or "").strip() or api_title
        chunks = _stamp_title(_splitter.split_documents(docs), title)
        for doc in chunks:
            doc.metadata["arxiv_id"] = arxiv_id
            if authors:
                doc.metadata["authors"] = authors
        if authors:
            metadata_doc = Document(
                page_content=(
                    f"Title: {title}\n"
                    f"ArXiv ID: {arxiv_id}\n"
                    f"Authors: {', '.join(authors)}"
                ),
                metadata={
                    "title": title,
                    "arxiv_id": arxiv_id,
                    "authors": authors,
                    "source": "arxiv_metadata",
                },
            )
            return [metadata_doc, *chunks]
        return chunks
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def load_arxiv(query: str) -> list[Document]:
    arxiv_id = _extract_arxiv_id(query) or _arxiv_search(query)
    return _load_arxiv_by_id(arxiv_id)


def load_document(source: str) -> list[Document]:
    """Dispatch to the appropriate loader based on URL prefix or file extension."""
    if source.startswith(("http://", "https://")):
        return load_webpage(source)
    ext = Path(source).suffix.lower()
    if ext == ".pdf":
        return load_pdf(source)
    if ext == ".txt":
        return load_text(source)
    if ext in (".md", ".markdown"):
        return load_markdown(source)
    raise ValueError(f"Unsupported file type: {ext}")
