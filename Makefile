# Makefile for Quantum Hamiltonian Simulation Framework

.PHONY: help install install-dev test test-unit test-numerical test-all test-fast test-coverage clean lint format verify

help:
	@echo "Quantum Hamiltonian Simulation Framework - Make Commands"
	@echo "=========================================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install          Install package and dependencies"
	@echo "  make install-dev      Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests with pytest"
	@echo "  make test-unit        Run only unit tests"
	@echo "  make test-numerical   Run only numerical validation tests"
	@echo "  make test-fast        Run fast tests only (skip slow tests)"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo "  make test-manual      Run tests without pytest"
	@echo ""
	@echo "Verification:"
	@echo "  make verify           Run static verification"
	@echo "  make lint             Run code linting"
	@echo "  make format           Format code with black"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove build artifacts and cache"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Installation complete"

install-dev:
	@echo "Installing with development dependencies..."
	pip install -e ".[dev]"
	@echo "✓ Development installation complete"

test:
	@echo "Running all tests with pytest..."
	pytest tests/ -v

test-unit:
	@echo "Running unit tests..."
	pytest tests/test_algorithms.py -v -m unit

test-numerical:
	@echo "Running numerical validation tests..."
	pytest tests/test_numerical_validation.py -v -m numerical

test-all:
	@echo "Running comprehensive test suite..."
	pytest tests/ -v --tb=short

test-fast:
	@echo "Running fast tests only..."
	pytest tests/ -v -m "not slow"

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/"

test-manual:
	@echo "Running manual test runner..."
	python tests/run_all_tests.py

verify:
	@echo "Running static verification..."
	python tests/static_verification.py

lint:
	@echo "Running linting checks..."
	-flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	@echo "✓ Linting complete"

format:
	@echo "Formatting code with black..."
	-black src/ tests/ --line-length=100
	@echo "✓ Formatting complete"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✓ Cleanup complete"

.DEFAULT_GOAL := help
