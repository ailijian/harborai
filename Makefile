# HarborAI Makefile
# Common development tasks for the HarborAI project

.PHONY: help install install-dev test test-cov lint format type-check clean build upload docs serve-docs

# Default target
help:
	@echo "HarborAI Development Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install      Install the package in development mode"
	@echo "  install-dev  Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run all tests"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  test-unit    Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run all linting tools"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo ""
	@echo "Database:"
	@echo "  init-db      Initialize database tables"
	@echo "  migrate      Run database migrations"
	@echo ""
	@echo "Build & Release:"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build the package"
	@echo "  upload       Upload to PyPI"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Build documentation"
	@echo "  serve-docs   Serve documentation locally"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,database,all]"

# Testing
test:
	pytest

test-cov:
	pytest --cov=harborai --cov-report=html --cov-report=term-missing

test-unit:
	pytest -m "unit"

test-integration:
	pytest -m "integration"

# Code quality
lint: flake8 mypy

flake8:
	flake8 harborai tests

format:
	black harborai tests
	isort harborai tests

type-check:
	mypy harborai

mypy: type-check

# Database
init-db:
	harborai init-db

migrate:
	alembic upgrade head

# Build and release
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	build: clean
	python -m build

upload: build
	twine upload dist/*

# Documentation
docs:
	@echo "Documentation build not implemented yet"

serve-docs:
	@echo "Documentation server not implemented yet"

# Development helpers
dev-setup: install-dev init-db
	@echo "Development environment setup complete!"

check: format lint test
	@echo "All checks passed!"

# CI/CD helpers
ci-test: install-dev test-cov lint type-check
	@echo "CI tests complete