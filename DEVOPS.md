# CiteMind-AI-RAGOps DevOps Guide

CiteMind-AI-RAGOps is packaged as a cloud-native RAG platform with Docker, Kubernetes, CI/CD validation, health checks, persistent storage, and an observability stack.

## DevOps Features

| Area | Implementation |
|---|---|
| Containerization | Multi-stage Dockerfile for React/Nginx frontend and FastAPI backend |
| Local orchestration | Docker Compose for frontend, backend, Qdrant, Ollama, Prometheus, and Grafana |
| Kubernetes | Namespace, Deployments, Services, PVCs, Ingress, ConfigMap, and HPA |
| Observability | Prometheus scrape config and Grafana dashboard provisioning |
| CI/CD | GitHub Actions workflow for frontend build, Compose validation, Docker image build, and Kubernetes dry-run |
| Operations | Makefile for repeatable build, run, logs, cleanup, and Kubernetes commands |
| Reliability | Health checks, readiness probes, liveness probes, persistent vector/model storage |

## Docker Compose Run

```bash
make build
make up
```

Application endpoints:

| Service | URL |
|---|---|
| Frontend | http://127.0.0.1:8020 |
| Backend API health | http://127.0.0.1:8020/api/health through frontend proxy, or backend container on port 8000 internally |
| Prometheus | http://127.0.0.1:9090 |
| Grafana | http://127.0.0.1:3000 |

Grafana default login:

```text
admin / admin
```

You can change the login with:

```env
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your-password
```

## Kubernetes Run

Build images locally first:

```bash
make build
```

If using Minikube, build images inside Minikube's Docker environment:

```bash
eval $(minikube docker-env)
make build
```

Deploy:

```bash
make k8s-apply
make k8s-status
make k8s-port-forward
```

Open:

```text
http://127.0.0.1:8020
```

Remove deployment:

```bash
make k8s-delete
```

## Kubernetes Architecture

```text
Ingress / Port Forward
        |
        v
citemind-frontend Service
        |
        v
React + Nginx frontend Pod
        |
        v
citemind-backend Service
        |
        v
FastAPI backend Pod
   |             |
   v             v
Qdrant PVC    Ollama PVC
Vector DB     Local LLM models
```

## CI/CD Pipeline

The GitHub Actions workflow in `.github/workflows/ci.yml` validates the project like a production repository:

1. Checks out the repository.
2. Installs Node.js and Python.
3. Builds the React frontend.
4. Validates Docker Compose configuration.
5. Builds backend and frontend Docker images.
6. Runs Kubernetes manifest dry-run validation.

## Recruiter Talking Points

- Converted a research-focused RAG application into a deployable cloud-native platform.
- Added Kubernetes manifests with persistent volumes, probes, services, ingress, and autoscaling.
- Added observability using Prometheus and Grafana provisioning.
- Added CI/CD validation with GitHub Actions.
- Added operational Makefile commands so engineers can run the stack consistently.
