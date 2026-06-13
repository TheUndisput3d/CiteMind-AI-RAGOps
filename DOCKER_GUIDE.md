# CiteMind AI Docker Guide

## Build And Run

```bash
docker compose build
docker compose up
```

The app is served at:

```text
http://127.0.0.1:8020
```

This starts the frontend, backend, Qdrant, Ollama, and the model-pull init service. The React frontend is served by Nginx and proxies `/api` to the FastAPI backend.

## Environment

`docker-compose.yml` reads `.env` from the project root, but it wires container services internally. You do not need to run Qdrant or Ollama separately.

Required for web search and claim verification:

```env
TAVILY_API_KEY=tvly-...
```

Useful overrides:

```env
CITEMIND_PORT=8020
OLLAMA_CHAT_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_NUM_GPU=999
```

For OpenAI or Groq deployments, set `LLM_PROVIDER=openai` or `LLM_PROVIDER=groq` and provide the matching API key.

## Useful Commands

```bash
docker compose logs -f frontend backend
docker compose restart frontend backend
docker compose down
```

## Image Only

```bash
docker build -t citemind-ai:latest .
docker run --env-file .env -p 8000:8000 citemind-ai:latest
```
