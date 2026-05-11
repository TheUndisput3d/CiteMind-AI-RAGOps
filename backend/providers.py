import os
import json
import subprocess
from typing import Any, Type

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def _auto_ollama_num_gpu() -> int | None:
    """Return Ollama's num_gpu setting.

    None means "do not pass num_gpu" so Ollama can use GPU normally.
    0 means force CPU when the GPU is already too full.
    """
    configured = os.getenv("OLLAMA_NUM_GPU")
    if configured is None:
        return None

    configured = configured.strip().lower()
    if configured != "auto":
        return int(configured)

    min_free_mb = int(os.getenv("OLLAMA_GPU_MIN_FREE_MB", "5000"))
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        free_mb = max(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except Exception:
        return None

    return None if free_mb >= min_free_mb else 0


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "").strip().lower()


def is_ollama() -> bool:
    """Return whether the active LLM provider is local Ollama."""
    return not _use_openai() and not _use_groq()


def _use_openai() -> bool:
    provider = _provider()
    if provider == "openai":
        return True
    if provider == "ollama":
        return False
    if provider == "groq":
        return False
    return bool(os.getenv("OPENAI_API_KEY"))


def _use_groq() -> bool:
    provider = _provider()
    if provider == "groq":
        return True
    if provider in {"openai", "ollama"}:
        return False
    return bool(os.getenv("GROQ_API_KEY")) and not bool(os.getenv("OPENAI_API_KEY"))


def get_chat_model(model_name: str | None = None):
    """Return the configured chat model.

    Defaults to OpenAI when an API key is present, otherwise falls back to Ollama.
    """
    if _use_openai():
        return ChatOpenAI(model=model_name or os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini"))

    if _use_groq():
        return ChatOpenAI(
            model=os.getenv("GROQ_CHAT_MODEL") or model_name or "llama-3.1-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )

    kwargs = {
        "model": os.getenv("OLLAMA_CHAT_MODEL") or model_name or "llama3.1",
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        "client_kwargs": {
            "timeout": int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300")),
        },
    }
    num_gpu = _auto_ollama_num_gpu()
    if num_gpu is not None:
        kwargs["num_gpu"] = num_gpu
    return ChatOllama(**kwargs)


def get_structured_output(model, schema: Type[BaseModel]):
    """Use native structured output, with JSON parsing for Ollama."""
    try:
        return model.with_structured_output(schema)
    except NotImplementedError:
        json_model = model.bind(format="json")

        def parse_response(response: Any) -> BaseModel:
            content = response.content if hasattr(response, "content") else response
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            text = str(content).strip()
            if text.startswith("```"):
                text = text.strip("`").removeprefix("json").strip()
            try:
                return schema.model_validate(json.loads(text))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Model returned invalid JSON for {schema.__name__}: {text}") from exc

        return json_model | RunnableLambda(parse_response)


def get_embeddings():
    """Return the configured embedding model.

    Uses OpenAI embeddings when available; otherwise uses local Ollama embeddings.
    """
    if _use_openai():
        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    kwargs = {
        "model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "client_kwargs": {
            "timeout": int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300")),
        },
    }
    num_gpu = _auto_ollama_num_gpu()
    if num_gpu is not None:
        kwargs["num_gpu"] = num_gpu
    return OllamaEmbeddings(**kwargs)
