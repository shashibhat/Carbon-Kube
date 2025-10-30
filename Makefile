# Carbon-Kube Makefile
# Provides convenient commands for building, testing, and deploying Carbon-Kube

# Variables
PROJECT_NAME := carbon-kube
DOCKER_REGISTRY := ghcr.io/your-org
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "v1.0.0")
NAMESPACE := carbon-kube

# Go variables
GO_MODULE := github.com/your-org/carbon-kube
GO_PACKAGES := ./pkg/...
GO_BUILD_FLAGS := -ldflags "-X main.version=$(VERSION)"
GOOS ?= linux
GOARCH ?= amd64
CGO_ENABLED ?= 0

# Python variables
PYTHON_VERSION := 3.9
VENV_DIR := .venv

# Docker images
SCHEDULER_IMAGE := $(DOCKER_REGISTRY)/$(PROJECT_NAME)-scheduler:$(VERSION)
POLLER_IMAGE := $(DOCKER_REGISTRY)/$(PROJECT_NAME)-poller:$(VERSION)
RL_TUNER_IMAGE := $(DOCKER_REGISTRY)/$(PROJECT_NAME)-rl-tuner:$(VERSION)

# Kubernetes
KUBECONFIG ?= ~/.kube/config
KUBECTL := kubectl --kubeconfig=$(KUBECONFIG)

.PHONY: help
help: ## Display this help message
	@echo "Carbon-Kube Development Commands"
	@echo "================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development Setup
.PHONY: dev-setup
dev-setup: ## Set up development environment
	@echo "Setting up development environment..."
	@command -v go >/dev/null 2>&1 || { echo "Go is required but not installed. Aborting." >&2; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting." >&2; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
	@command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed. Aborting." >&2; exit 1; }
	@command -v helm >/dev/null 2>&1 || { echo "Helm is required but not installed. Aborting." >&2; exit 1; }
	$(MAKE) setup-python
	$(MAKE) setup-go
	@echo "Development environment setup complete!"

.PHONY: setup-python
setup-python: ## Set up Python virtual environment
	@echo "Setting up Python environment..."
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install pytest pytest-cov pytest-asyncio aiohttp kubernetes numpy stable-baselines3 gymnasium

.PHONY: setup-go
setup-go: ## Set up Go dependencies
	@echo "Setting up Go environment..."
	cd pkg/emissionplugin && go mod tidy
	cd pkg/emissionplugin && go mod download

# Building
.PHONY: build
build: build-go build-python ## Build all components

.PHONY: build-go
build-go: ## Build Go scheduler plugin
	@echo "Building Go scheduler plugin..."
	mkdir -p bin
	cd pkg/emissionplugin && CGO_ENABLED=1 GOOS=$(GOOS) GOARCH=$(GOARCH) go build -buildmode=plugin -o ../../bin/scheduler-plugin.so ./cmd/scheduler

.PHONY: build-python
build-python: setup-python ## Build Python components
	@echo "Building Python components..."
	$(VENV_DIR)/bin/python -m py_compile scripts/poller.py
	$(VENV_DIR)/bin/python -m py_compile scripts/rl_tuner.py

.PHONY: build-images
build-images: ## Build all Docker images
	$(MAKE) build-scheduler-image
	$(MAKE) build-poller-image
	$(MAKE) build-rl-tuner-image

.PHONY: build-scheduler-image
build-scheduler-image: ## Build scheduler Docker image
	@echo "Building scheduler Docker image..."
	docker build -f docker/Dockerfile.scheduler -t $(SCHEDULER_IMAGE) .

.PHONY: build-poller-image
build-poller-image: ## Build poller Docker image
	@echo "Building poller Docker image..."
	docker build -f docker/Dockerfile.poller -t $(POLLER_IMAGE) .

.PHONY: build-rl-tuner-image
build-rl-tuner-image: ## Build RL tuner Docker image
	@echo "Building RL tuner Docker image..."
	docker build -f docker/Dockerfile.rl-tuner -t $(RL_TUNER_IMAGE) .

# Testing
.PHONY: test
test: test-go test-python ## Run all tests

.PHONY: test-go
test-go: ## Run Go unit tests
	@echo "Running Go unit tests..."
	cd pkg/emissionplugin && go test -v -race -coverprofile=coverage.out $(GO_PACKAGES)
	cd pkg/emissionplugin && go tool cover -html=coverage.out -o coverage.html

.PHONY: test-python
test-python: setup-python ## Run Python unit tests
	@echo "Running Python unit tests..."
	cd scripts && ../$(VENV_DIR)/bin/python -m pytest test_poller.py test_rl_tuner.py -v --cov=. --cov-report=html

.PHONY: test-integration
test-integration: setup-python ## Run integration tests
	@echo "Running integration tests..."
	cd test/integration && ../../$(VENV_DIR)/bin/python -m pytest test_e2e.py -v

.PHONY: test-performance
test-performance: setup-python ## Run performance tests
	@echo "Running performance tests..."
	cd test/integration && ../../$(VENV_DIR)/bin/python -m pytest test_e2e.py::TestCarbonKubePerformance -v

# Helm targets
.PHONY: helm-lint
helm-lint: ## Lint Helm charts
	@echo "Linting Helm charts..."
	helm lint ./charts/carbon-kube

.PHONY: helm-template
helm-template: ## Generate Kubernetes manifests from Helm charts
	@echo "Generating Kubernetes manifests..."
	helm template $(PROJECT_NAME) ./charts/carbon-kube \
		--namespace $(NAMESPACE) \
		--output-dir ./manifests

.PHONY: helm-package
helm-package: ## Package Helm chart
	@echo "Packaging Helm chart..."
	helm package ./charts/carbon-kube --destination ./dist

helm-install:
	helm install carbon-kube charts/carbon-kube \
		--set image.tag=$(VERSION) \
		--set electricityMaps.apiKey=$(ELECTRICITY_MAPS_API_KEY)

# CDK deployment
deploy-cdk:
	@echo "Deploying with CDK..."
	cd cdk && cdk deploy --require-approval never

destroy-cdk:
	cd cdk && cdk destroy --force

# EKS testing
test-eks: deploy-cdk
	@echo "Running EKS integration tests..."
	kubectl apply -f test/workloads/
	python test/run_evaluation.py

# Clean targets
clean:
	rm -rf bin/
	rm -f coverage.out coverage.html
	rm -f carbon-kube-*.tgz

# Development targets
dev-setup:
	@echo "Setting up development environment..."
	go mod download
	pip install -r scripts/requirements.txt
	pip install -r cdk/requirements.txt

lint:
	golangci-lint run ./...
	flake8 scripts/ cdk/
	helm lint charts/carbon-kube

# Help
help:
	@echo "Available targets:"
	@echo "  build          - Build all binaries"
	@echo "  test           - Run all tests"
	@echo "  docker         - Build Docker images"
	@echo "  helm-package   - Package Helm chart"
	@echo "  deploy-cdk     - Deploy to AWS with CDK"
	@echo "  test-eks       - Run EKS integration tests"
	@echo "  clean          - Clean build artifacts"
	@echo "  dev-setup      - Setup development environment"