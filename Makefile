# ================================================================
# XYZ MDM 3.0 — Feature Store Service Makefile
# Uses pip + setuptools (not Poetry)
# ================================================================
.PHONY: install run-local run-dev migrate test lint fmt clean help

install: ## Install all dependencies (warning: downloads ~2GB ML models on first run)
	pip install -e ".[dev]"

install-no-ml: ## Install without heavy ML deps (faster, no torch/sentence-transformers)
	pip install fastapi uvicorn pydantic pydantic-settings sqlalchemy asyncpg redis aiokafka boto3 alembic prometheus-client structlog httpx

env-setup: ## Create .env.local from .env.local.example
	@if not exist .env.local (copy .env.local.example .env.local && echo .env.local created — fill in your values.) else (echo .env.local already exists.)

## Database
migrate: ## Run Alembic migrations
	alembic upgrade head

migrate-create: ## Create new migration (usage: make migrate-create name=add_feature_table)
	alembic revision --autogenerate -m "$(name)"

## Run
run-local: ## Run locally — loads .env + .env.local via Pydantic Settings
	@if not exist .env.local (echo ERROR: .env.local not found. Run: make env-setup && exit 1)
	uvicorn src.main:app --host 0.0.0.0 --port 8115 --reload

run-dev: ## Run in dev mode
	uvicorn src.main:app --host 0.0.0.0 --port 8115 --reload --log-level debug

run: ## Run without reload
	uvicorn src.main:app --host 0.0.0.0 --port 8115

## Test
test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

## Code Quality
lint: ## Run linter
	ruff check src/

fmt: ## Format code
	ruff check src/ --fix

## Docker
docker-build: ## Build Docker image
	docker build -t xyz-mdm/feature-store-service:3.0.0 .

compose-up: ## Start with docker-compose
	docker-compose up -d

compose-down: ## Stop docker-compose
	docker-compose down

compose-logs: ## View logs
	docker-compose logs -f feature-store-service

## Clean
clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>nul || true
	find . -type f -name "*.pyc" -delete 2>nul || true
	rmdir /s /q .pytest_cache 2>nul || true
	rmdir /s /q htmlcov 2>nul || true
	rmdir /s /q feature_store_service.egg-info 2>nul || true

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
