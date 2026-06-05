PROJECT_NAME := citemind-ai-ragops
COMPOSE := docker compose
K8S_NAMESPACE := citemind-ragops

.PHONY: help build up down restart logs ps clean compose-check k8s-apply k8s-delete k8s-status k8s-port-forward

help:
	@echo "CiteMind-AI-RAGOps DevOps commands"
	@echo "  make build              Build Docker images"
	@echo "  make up                 Start Docker Compose stack"
	@echo "  make down               Stop Docker Compose stack"
	@echo "  make restart            Restart Docker Compose stack"
	@echo "  make logs               Follow service logs"
	@echo "  make ps                 Show running containers"
	@echo "  make compose-check      Validate docker-compose.yml"
	@echo "  make clean              Remove stopped containers and build cache"
	@echo "  make k8s-apply          Deploy Kubernetes manifests"
	@echo "  make k8s-delete         Delete Kubernetes manifests"
	@echo "  make k8s-status         Show Kubernetes pods/services"
	@echo "  make k8s-port-forward   Open frontend on http://127.0.0.1:8020"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) down
	$(COMPOSE) up

logs:
	$(COMPOSE) logs -f --tail=100

ps:
	$(COMPOSE) ps

compose-check:
	$(COMPOSE) config >/tmp/citemind-compose.yml
	@echo "Docker Compose config is valid"

clean:
	docker builder prune -f
	docker container prune -f

k8s-apply:
	kubectl apply -f k8s/

k8s-delete:
	kubectl delete -f k8s/ --ignore-not-found=true

k8s-status:
	kubectl get all,pvc,configmap -n $(K8S_NAMESPACE)

k8s-port-forward:
	kubectl -n $(K8S_NAMESPACE) port-forward service/citemind-frontend 8020:80
