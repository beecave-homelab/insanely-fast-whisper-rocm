#!/bin/bash
set -euo pipefail

# ==============================================================================
# Script Description: Run Ruff in grouped passes (optionally keep going on errors),
#   and apply safe auto-fixes where requested.
# Author: elvee
# Version: 1.3.0
# License: MIT
# Creation Date: 17/08/2025
# Last Modified: 17/08/2025
# Usage: clean_codebase_sorted.sh [OPTIONS]
# ==============================================================================

# ==============================================================================
# Constants
# ==============================================================================
SCRIPT_NAME="$(basename "$0")"
KEEP_GOING=false  # If true, do not stop on Ruff failures; run all passes.
USE_PREVIEW=false # If true, enable Ruff preview rules (needed for DOC/FA).

# Colors (disabled if NO_COLOR is set or not a TTY)
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RED="\033[0;31m"
  GREEN="\033[0;32m"
  YELLOW="\033[1;33m"
  CYAN="\033[0;36m"
  BOLD="\033[1m"
  DIM="\033[2m"
  NC="\033[0m"
else
  RED=""; GREEN=""; YELLOW=""; CYAN=""; BOLD=""; DIM=""; NC=""
fi

CHECK="${GREEN}✔${NC}"
CROSS="${RED}✖${NC}"
ARROW="${CYAN}▶${NC}"

# ==============================================================================
# ASCII Art
# ==============================================================================
print_ascii_art() {
  echo -e "${BOLD}
Clean Codebase with Ruff${NC}
"
}

# ==============================================================================
# Show Help
# ==============================================================================
show_help() {
  cat << EOF
Usage: $SCRIPT_NAME [OPTIONS] [PATHS...]

Options:
  -k, --keep-going    Continue running all passes even if a pass fails.
  -p, --preview       Enable Ruff preview rules (recommended for DOC, FA).
  -h, --help          Show this help message and exit.

Description:
  Runs Ruff checks in grouped passes for the configured select codes:
   - Pyflakes (F)
   - Pycodestyle (E, W)
   - Naming (N)
   - Import sorting with fixes (I)
   - Docstrings (D)
   - Pydoclint (DOC) [often requires --preview]
   - Annotations (ANN)
   - Tidy imports (TID)
   - Pyupgrade with fixes (UP)
   - Future annotations (FA) [often requires --preview]

Targets:
  Provide one or more files/directories (e.g., src/ tests/cli). If omitted,
  defaults to the current directory (.). Shell globs are supported by Bash.

Notes:
  By default, the script stops on the first failing Ruff pass (set -e).
  Use --keep-going to run *all* passes and see a complete report.
EOF
}

# ==============================================================================
# Error Handling
# ==============================================================================
error_exit() {
  echo -e "${CROSS} ${BOLD}Error:${NC} $1" >&2
  exit 1
}

trap 'echo -e "\n${CROSS} ${BOLD}A command failed.${NC} (see output above)"; exit 1' ERR

# ==============================================================================
# Preconditions
# ==============================================================================
require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    error_exit "'${cmd}' not found in PATH"
  fi
}

# ==============================================================================
# Ruff Runner
# - Prints the command in cyan, runs it, then prints a success tick.
# - Honors KEEP_GOING to avoid aborting the script on non-zero exit codes.
# ==============================================================================
RUFF_COMMON=( )

run_ruff() {
  # Build a rendered command including common args for visibility
  local rendered=""
  local arg
  for arg in "${RUFF_COMMON[@]}" "$@"; do
    # shellcheck disable=SC2089
    rendered+=" $(printf "%q" "${arg}")"
  done

  echo -e "\n${ARROW} ${CYAN}Running:${NC} ruff check${DIM}${rendered}${NC}"

  if [[ "${KEEP_GOING}" == true ]]; then
    # Continue even if this pass fails; still show the end marker.
    if ruff check "${RUFF_COMMON[@]}" "$@"; then
      echo -e "${CHECK} Completed: ruff check${DIM}${rendered}${NC}"
    else
      echo -e "${YELLOW}⚠ Completed with issues:${NC} ruff check${DIM}${rendered}${NC}"
    fi
  else
    ruff check "${RUFF_COMMON[@]}" "$@"
    echo -e "${CHECK} Completed: ruff check${DIM}${rendered}${NC}"
  fi
}

# ==============================================================================
# Main Logic
# ==============================================================================
main_logic() {
  echo -e "${YELLOW}${BOLD}Ruff passes${NC}"

  # Pyflakes
  run_ruff "${TARGETS[@]}" --select F

  # Pycodestyle
  run_ruff "${TARGETS[@]}" --select E,W

  # Naming
  run_ruff "${TARGETS[@]}" --select N

  # Import sorting (auto-fix)
  run_ruff "${TARGETS[@]}" --select I --fix

  # Docstrings
  run_ruff "${TARGETS[@]}" --select D

  # Pydoclint (DOC)
  run_ruff "${TARGETS[@]}" --select DOC

  # Annotations
  run_ruff "${TARGETS[@]}" --select ANN

  # Tidy imports
  run_ruff "${TARGETS[@]}" --select TID

  # Pyupgrade (auto-fix)
  run_ruff "${TARGETS[@]}" --select UP --fix

  # Future annotations (FA)
  run_ruff "${TARGETS[@]}" --select FA

  echo -e "\n${CHECK} ${BOLD}All passes completed.${NC}"
}

# ==============================================================================
# Main
# ==============================================================================
main() {
  require_cmd "ruff"

  TARGETS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -k|--keep-going)
        KEEP_GOING=true
        shift
        ;;
      -p|--preview)
        USE_PREVIEW=true
        shift
        ;;
      -h|--help)
        show_help
        exit 0
        ;;
      *)
        # Treat as a path/target (supports multiple)
        TARGETS+=("$1")
        shift
        ;;
    esac
  done

  # Configure common Ruff args
  if [[ "${USE_PREVIEW}" == true ]]; then
    RUFF_COMMON+=(--preview)
  fi

  # Default to current directory if no targets provided
  if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(.)
  fi

  main_logic
}

# ==============================================================================
# Execution
# ==============================================================================
print_ascii_art
main "$@"
