# Makefile for mcp-server-qdrant development

.PHONY: help build up down logs test clean dev-setup

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development setup
dev-setup: ## Set up development environment
	@echo "Setting up development environment..."
	@cp .env.example .env
	@echo "Created .env file from .env.example"
	@echo "Please edit .env file with your configuration"
	@uv sync --dev
	@echo "Development environment ready!"

# Docker operations
build: ## Build the Docker images
	docker-compose build

up: ## Start all services
	docker-compose up -d

up-logs: ## Start all services with logs
	docker-compose up

down: ## Stop all services
	docker-compose down

logs: ## Show logs from all services
	docker-compose logs -f

logs-mcp: ## Show logs from MCP server only
	docker-compose logs -f mcp-server-qdrant

logs-qdrant: ## Show logs from Qdrant only
	docker-compose logs -f qdrant

# Development workflows
dev: ## Start development environment
	docker-compose up

dev-build: ## Build and start development environment
	docker-compose up --build

# Testing
test: ## Run tests locally
	uv run pytest

test-docker: ## Run tests in Docker container
	docker-compose exec mcp-server-qdrant uv run pytest

test-integration: ## Run integration tests
	uv run pytest tests/test_*integration*.py -v

# Qdrant operations
qdrant-only: ## Start only Qdrant service
	docker-compose up -d qdrant

qdrant-ui: ## Start Qdrant with web UI
	docker-compose --profile web-ui up -d

# Cleanup
clean: ## Clean up Docker resources
	docker-compose down -v
	docker system prune -f

clean-all: ## Clean up everything including images
	docker-compose down -v --rmi all
	docker system prune -af

# Health checks
health: ## Check service health
	@echo "Checking Qdrant health..."
	@curl -f http://localhost:6333/ > /dev/null && echo "✅ Qdrant is healthy" || echo "❌ Qdrant not responding"
	@echo "Checking MCP server health..."
	@curl -f http://localhost:8000/ > /dev/null 2>&1 && echo "✅ MCP server is responding" || echo "❌ MCP server not responding"
	@echo "Checking MCP SSE endpoint..."
	@curl -I -f http://localhost:8000/sse 2>/dev/null | grep -q "text/event-stream" && echo "✅ MCP SSE endpoint is available" || echo "❌ MCP SSE endpoint not available"

# Development utilities
shell-mcp: ## Open shell in MCP server container
	docker-compose exec mcp-server-qdrant /bin/bash

shell-qdrant: ## Open shell in Qdrant container
	docker-compose exec qdrant /bin/bash

inspect-mcp: ## Start MCP inspector for debugging
	COLLECTION_NAME=dev-collection uv run fastmcp dev src/mcp_server_qdrant/server.py

# Code quality
lint: ## Run code linting
	uv run ruff check src/ tests/
	uv run mypy src/

format: ## Format code
	uv run ruff format src/ tests/
	uv run isort src/ tests/

pre-commit: ## Run pre-commit hooks
	uv run pre-commit run --all-files