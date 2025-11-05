.PHONY: help install install-dev lint format type-check test clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

lint: ## Run Ruff linter
	ruff check .

format: ## Format code with Ruff
	ruff format .

format-check: ## Check formatting without making changes
	ruff format --check .

type-check: ## Run Pyright type checker
	pyright shared_ollama_client.py utils.py examples/

check: lint format-check type-check ## Run all checks (lint, format, type-check)

fix: ## Auto-fix linting issues
	ruff check --fix .

test: ## Run tests with pytest
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=shared_ollama_client --cov=utils --cov-report=html --cov-report=term

clean: ## Clean up generated files
	rm -rf .ruff_cache .pyright_cache .pytest_cache .mypy_cache
	rm -rf build dist *.egg-info
	rm -rf htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

all: clean install-dev format fix type-check test ## Run full development cycle

