.PHONY: help train serve test docker-build docker-push k8s-deploy k8s-clean

REGISTRY   ?= docker.io/thiagoar587
TAG        ?= latest

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Local ──────────────────────────────────────

train: ## Train model locally
	MLFLOW_TRACKING_URI=http://localhost:5000 \
	DATA_PATH=data/transactions_1M.csv \
	python src/train.py

serve: ## Start API locally
	MLFLOW_TRACKING_URI=http://localhost:5000 \
	uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload

test: ## Run API integration tests
	python tests/test_api.py

# ── Docker ─────────────────────────────────────

docker-build: ## Build all Docker images
	docker build -f docker/Dockerfile.mlflow -t $(REGISTRY)/mlflow-server:$(TAG) .
	docker build -f docker/Dockerfile.train  -t $(REGISTRY)/fraud-train:$(TAG) .
	docker build -f docker/Dockerfile.serve  -t $(REGISTRY)/fraud-serve:$(TAG) .

docker-push: ## Push images to registry
	docker push $(REGISTRY)/mlflow-server:$(TAG)
	docker push $(REGISTRY)/fraud-train:$(TAG)
	docker push $(REGISTRY)/fraud-serve:$(TAG)

# ── Kubernetes ─────────────────────────────────

k8s-deploy: ## Deploy full stack to Kubernetes
	kubectl apply -f k8s/00-namespace.yaml
	kubectl apply -f k8s/01-mlflow-pvc.yaml
	kubectl apply -f k8s/07-data-pvc.yaml
	kubectl apply -f k8s/02-mlflow-deployment.yaml
	kubectl apply -f k8s/03-mlflow-service.yaml
	kubectl apply -f k8s/08-upload-pod.yaml
	kubectl apply -f k8s/04-train-job.yaml
	kubectl apply -f k8s/05-serve-deployment.yaml
	kubectl apply -f k8s/06-serve-service.yaml

k8s-clean: ## Remove all resources
	kubectl delete namespace fraud-detection --ignore-not-found
