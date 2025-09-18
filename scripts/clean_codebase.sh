#!/usr/bin/env bash
# clean_codebase.sh
# Run Ruff and isort to auto-fix, format, and clean up the codebase.
# Usage: bash scripts/clean_codebase.sh

set -e

# Run Ruff to auto-fix lint issues
pdm run ruff check . --fix

# Format code with Ruff
pdm run ruff format .

# Sort imports with isort
pdm run isort .

echo "Codebase cleaned: ruff check --fix, ruff format, isort complete."
