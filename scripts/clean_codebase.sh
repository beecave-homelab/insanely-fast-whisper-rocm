#!/usr/bin/env bash
# clean_codebase.sh
# Run Ruff and isort to auto-fix, format, and clean up the codebase.
# Usage:
#   bash scripts/clean_codebase.sh [PATHS...]
# If no PATHS are provided, defaults to the current directory (.).

set -euo pipefail

show_help() {
  cat << EOF
Usage: $(basename "$0") [PATHS...]

Description:
  Runs Ruff auto-fixes, formatting, and isort on the given target paths.
  If no paths are provided, defaults to the current directory (.).

Examples:
  scripts/clean_codebase.sh
  scripts/clean_codebase.sh tests/ tests/cli
EOF
}

TARGETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      TARGETS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(.)
fi

# Run Ruff to auto-fix lint issues
pdm run ruff check "${TARGETS[@]}" --fix

# Format code with Ruff
pdm run ruff format "${TARGETS[@]}"

echo "Codebase cleaned: 'ruff check --fix' and 'ruff format' for: ${TARGETS[*]}"
