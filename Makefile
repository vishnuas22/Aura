# =============================================================================
# AI RESEARCH ASSISTANT - MAKEFILE
# =============================================================================
# Production-ready Makefile for development and deployment tasks

.PHONY: help install dev test lint format clean build deploy docs

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON_VERSION := 3.11
NODE_VERSION := 18
PROJECT_NAME := ai-research-assistant
DOCKER_IMAGE := $(PROJECT_NAME)
DOCKER_TAG := latest

# Colors for output
BOLD := \033[1m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
BLUE := \033[34m
NC := \033[0m

# =============================================================================
# HELP TARGET
# =============================================================================
help: ## Show this help message
	@echo "$(BOLD)AI Research Assistant - Available Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BOLD)Examples:$(NC)"
	@echo "  make install      # Install all dependencies"
	@echo "  make dev          # Start development environment"
	@echo "  make test         # Run all tests"
	@echo "  make lint         # Run code quality checks"

# =============================================================================
# INSTALLATION AND SETUP
# =============================================================================
install: ## Install all dependencies and setup environment
	@echo "$(GREEN)Installing dependencies...$(NC)"
	@chmod +x scripts/setup.sh
	@./scripts/setup.sh

install-dev: ## Install development dependencies only
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	@python -m pip install --upgrade pip
	@pip install -e ".[dev,test]"
	@cd frontend && yarn install

setup-env: ## Setup environment files from templates
	@echo "$(GREEN)Setting up environment files...$(NC)"
	@[ ! -f backend/.env ] && cp .env.example backend/.env || echo "Backend .env already exists"
	@[ ! -f frontend/.env ] && echo "REACT_APP_BACKEND_URL=http://localhost:8001" > frontend/.env || echo "Frontend .env already exists"

# =============================================================================
# DEVELOPMENT
# =============================================================================
dev: ## Start development environment with Docker
	@echo "$(GREEN)Starting development environment...$(NC)"
	@chmod +x scripts/dev.sh
	@./scripts/dev.sh start

dev-native: ## Start development servers natively (without Docker)
	@echo "$(GREEN)Starting native development servers...$(NC)"
	@echo "$(YELLOW)Starting backend server...$(NC)"
	@cd backend && uvicorn server:app --host 0.0.0.0 --port 8001 --reload &
	@echo "$(YELLOW)Starting frontend server...$(NC)"
	@cd frontend && yarn start &
	@echo "$(GREEN)Servers started! Backend: http://localhost:8001, Frontend: http://localhost:3000$(NC)"

stop: ## Stop development environment
	@echo "$(GREEN)Stopping development environment...$(NC)"
	@./scripts/dev.sh stop

restart: ## Restart development environment
	@echo "$(GREEN)Restarting development environment...$(NC)"
	@./scripts/dev.sh restart

logs: ## Show logs from all services
	@./scripts/dev.sh logs

logs-backend: ## Show backend logs
	@./scripts/dev.sh logs backend

logs-frontend: ## Show frontend logs
	@./scripts/dev.sh logs frontend

# =============================================================================
# TESTING
# =============================================================================
test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(NC)"
	@./scripts/dev.sh test all

test-backend: ## Run backend tests only
	@echo "$(GREEN)Running backend tests...$(NC)"
	@./scripts/dev.sh test backend

test-frontend: ## Run frontend tests only
	@echo "$(GREEN)Running frontend tests...$(NC)"
	@./scripts/dev.sh test frontend

test-e2e: ## Run end-to-end tests
	@echo "$(GREEN)Running e2e tests...$(NC)"
	@./scripts/dev.sh test e2e

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	@docker-compose run --rm backend python -m pytest tests/ --cov=backend --cov-report=html --cov-report=term-missing

# =============================================================================
# CODE QUALITY
# =============================================================================
lint: ## Run all code quality checks
	@echo "$(GREEN)Running code quality checks...$(NC)"
	@./scripts/dev.sh lint

lint-backend: ## Run backend linting
	@echo "$(GREEN)Running backend linting...$(NC)"
	@python -m black --check backend/
	@python -m isort --check-only backend/
	@python -m flake8 backend/
	@python -m mypy backend/

lint-frontend: ## Run frontend linting
	@echo "$(GREEN)Running frontend linting...$(NC)"
	@cd frontend && yarn lint

format: ## Format all code
	@echo "$(GREEN)Formatting code...$(NC)"
	@./scripts/dev.sh format

format-backend: ## Format backend code
	@echo "$(GREEN)Formatting backend code...$(NC)"
	@python -m black backend/
	@python -m isort backend/

format-frontend: ## Format frontend code
	@echo "$(GREEN)Formatting frontend code...$(NC)"
	@cd frontend && yarn format

security-scan: ## Run security scans
	@echo "$(GREEN)Running security scans...$(NC)"
	@python -m bandit -r backend/ -f json -o security-report.json
	@python -m safety check --json --output safety-report.json || true
	@echo "$(GREEN)Security reports generated: security-report.json, safety-report.json$(NC)"

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================
db-status: ## Check database status
	@./scripts/dev.sh db status

db-shell: ## Open database shell
	@./scripts/dev.sh db shell

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(RED)WARNING: This will destroy all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		./scripts/dev.sh db reset; \
	fi

db-backup: ## Create database backup
	@./scripts/dev.sh db backup

db-migrate: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(NC)"
	@docker-compose run --rm backend python -c "from backend.database import init_db; init_db()"

# =============================================================================
# BUILD AND DEPLOYMENT
# =============================================================================
build: ## Build production Docker images
	@echo "$(GREEN)Building production images...$(NC)"
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) --target production .

build-dev: ## Build development Docker images
	@echo "$(GREEN)Building development images...$(NC)"
	@docker-compose build

push: ## Push Docker images to registry
	@echo "$(GREEN)Pushing Docker images...$(NC)"
	@docker push $(DOCKER_IMAGE):$(DOCKER_TAG)

deploy-staging: ## Deploy to staging environment
	@echo "$(GREEN)Deploying to staging...$(NC)"
	@./scripts/dev.sh deploy-staging

deploy-prod: ## Deploy to production environment
	@echo "$(RED)Production deployment requires manual approval$(NC)"
	@echo "Please use the CI/CD pipeline for production deployment"

# =============================================================================
# DOCUMENTATION
# =============================================================================
docs: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(NC)"
	@cd docs && mkdocs build

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8000$(NC)"
	@cd docs && mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(GREEN)Deploying documentation...$(NC)"
	@cd docs && mkdocs gh-deploy

# =============================================================================
# UTILITIES
# =============================================================================
clean: ## Clean up development environment
	@echo "$(GREEN)Cleaning up...$(NC)"
	@./scripts/dev.sh clean

clean-cache: ## Clean Python and Node caches
	@echo "$(GREEN)Cleaning caches...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name "node_modules" -exec rm -rf {} +

requirements: ## Generate requirements.txt from pyproject.toml
	@echo "$(GREEN)Generating requirements.txt...$(NC)"
	@pip-compile pyproject.toml

deps-upgrade: ## Upgrade all dependencies
	@echo "$(GREEN)Upgrading dependencies...$(NC)"
	@pip install --upgrade pip
	@pip-compile --upgrade pyproject.toml
	@cd frontend && yarn upgrade

check-deps: ## Check for security vulnerabilities in dependencies
	@echo "$(GREEN)Checking dependencies for vulnerabilities...$(NC)"
	@python -m safety check
	@cd frontend && yarn audit

shell-backend: ## Open backend shell
	@docker-compose exec backend bash

shell-frontend: ## Open frontend shell
	@docker-compose exec frontend bash

shell-db: ## Open database shell
	@docker-compose exec mongodb mongosh ai_research_assistant

# =============================================================================
# CI/CD HELPERS
# =============================================================================
ci-test: ## Run CI test suite
	@echo "$(GREEN)Running CI test suite...$(NC)"
	@python -m pytest tests/ --cov=backend --cov-report=xml --junitxml=test-results.xml

ci-build: ## Build for CI
	@echo "$(GREEN)Building for CI...$(NC)"
	@docker build --target testing .

ci-deploy: ## Deploy via CI
	@echo "$(GREEN)CI deployment...$(NC)"
	@echo "This should be run by CI/CD pipeline only"

# =============================================================================
# MONITORING AND HEALTH CHECKS
# =============================================================================
health: ## Check application health
	@echo "$(GREEN)Checking application health...$(NC)"
	@curl -f http://localhost:8001/api/health || echo "Backend not healthy"
	@curl -f http://localhost:3000 || echo "Frontend not healthy"

monitor: ## Start monitoring dashboard
	@echo "$(GREEN)Starting monitoring dashboard...$(NC)"
	@docker-compose --profile monitoring up -d

performance: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(NC)"
	@docker-compose run --rm performance-tests