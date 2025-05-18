.PHONY: install format lint test test-cov clean help

# Variables
PYTHON = python3
PIP = pip3
PYTEST = python -m pytest
COVERAGE = python -m coverage

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install development dependencies"
	@echo "  format      - Format code with Black and isort"
	@echo "  lint        - Run code quality checks (flake8, mypy, black, isort)"
	@echo "  test        - Run tests with coverage"
	@echo "  test-fast   - Run tests without coverage"
	@echo "  test-cov    - Run tests with coverage report"
	@echo "  clean       - Clean up build and test artifacts"

# Install development dependencies
install:
	$(PIP) install -e .
	$(PIP) install -r requirements-dev.txt
	pre-commit install

# Format code
format:
	black src tests
	isort src tests

# Lint code
lint: lint-flake8 lint-mypy lint-black lint-isort

lint-flake8:
	flake8 src tests

lint-mypy:
	mypy src

lint-black:
	black --check src tests

lint-isort:
	isort --check-only src tests

# Run tests
test: test-cov

test-fast:
	$(PYTEST) -v tests/

test-cov:
	$(PYTEST) --cov=src --cov-report=term-missing --cov-report=xml:coverage.xml --cov-report=html:htmlcov -v tests/

# Clean up
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .coverage coverage.xml htmlcov/ .hypothesis/
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
