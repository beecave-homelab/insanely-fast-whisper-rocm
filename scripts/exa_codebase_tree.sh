#!/bin/bash

set -euo pipefail

target_path="${1:-.}"

if command -v exa >/dev/null 2>&1; then
  exa -T "$target_path" --ignore-glob '*.pyc|__pycache__|.git|.venv|.ruff|.pytest'
elif command -v eza >/dev/null 2>&1; then
  eza -T "$target_path" --ignore-glob '*.pyc|__pycache__|.git|.venv|.ruff|.pytest'
else
  echo "Error: neither 'exa' nor 'eza' is installed." >&2
  exit 127
fi
