#!/bin/bash
set -euo pipefail

# Script Description: Run lightweight reviewer CI without heavy ML dependencies.
# Author: elvee
# Version: 0.1.0
# License: MIT
# Creation Date: 21/06/2026
# Last Modified: 21/06/2026
# Usage: local-ci-reviewer.sh [OPTIONS]

# Constants
DEFAULT_OUTPUT_FILE="${PWD}/ci-reviewer-output.log"

readonly DEFAULT_OUTPUT_FILE

# ASCII Art (Calvin font)
print_ascii_art() {
  echo "
    ██╗      ██████╗  ██████╗ █████╗ ██╗       ██████╗██╗
    ██║     ██╔═══██╗██╔════╝██╔══██╗██║      ██╔════╝██║
    ██║     ██║   ██║██║     ███████║██║█████╗██║     ██║
    ██║     ██║   ██║██║     ██╔══██║██║╚════╝██║     ██║
    ███████╗╚██████╔╝╚██████╗██║  ██║███████╗ ╚██████╗██║
    ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝  ╚═════╝╚═╝

██████╗ ███████╗██╗   ██╗██╗███████╗██╗    ██╗███████╗██████╗
██╔══██╗██╔════╝██║   ██║██║██╔════╝██║    ██║██╔════╝██╔══██╗
██████╔╝█████╗  ██║   ██║██║█████╗  ██║ █╗ ██║█████╗  ██████╔╝
██╔══██╗██╔══╝  ╚██╗ ██╔╝██║██╔══╝  ██║███╗██║██╔══╝  ██╔══██╗
██║  ██║███████╗ ╚████╔╝ ██║███████╗╚███╔███╔╝███████╗██║  ██║
╚═╝  ╚═╝╚══════╝  ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝
"
}

# Function to display help.
show_help() {
  echo "
Usage: $0 [OPTIONS]

Options:
  -o, --output_file FILE     Write CI logs to file.
                             Default: ${DEFAULT_OUTPUT_FILE}
  -h, --help                 Show help.

This reviewer CI performs:
  - Heavy dependency guard.
  - pdm run ruff check .
  - pdm run ruff format --check .
  - Reviewer-safe pytest paths.

It intentionally does not run the full test suite. Current package imports
require torch during pytest collection, so full local-ci.sh needs the normal
development/ROCm dependency set.
"
}

# Function for error handling.
error_exit() {
  echo "Error: $1" >&2
  exit 1
}

# Run a command and keep logging output through tee.
run_step() {
  local label="$1"
  shift

  echo ""
  echo "[+] ${label}"
  "$@"
}

# Fail if the reviewer environment contains known heavy packages.
check_no_heavy_packages() {
  local heavy_pattern
  local heavy_hits
  heavy_pattern="^(torch|torchvision|torchaudio|torch-audiomentations|"
  heavy_pattern+="torch-pitch-shift|torchcodec|torchmetrics|onnx|"
  heavy_pattern+="onnxruntime|onnxruntime-gpu|onnxruntime-rocm|"
  heavy_pattern+="pytorch-triton-rocm|triton|demucs|stable-ts|"
  heavy_pattern+="openai-whisper|pyannote-audio|lightning|"
  heavy_pattern+="pytorch-lightning|accelerate|optimum|tensorflow|"
  heavy_pattern+="jax|jaxlib|flax|ctranslate2)=="

  heavy_hits="$(mktemp)"
  if pdm list --freeze | grep -E "${heavy_pattern}" >"${heavy_hits}"; then
    echo "Heavy packages found in reviewer environment:" >&2
    cat "${heavy_hits}" >&2
    rm -f "${heavy_hits}"
    return 1
  fi
  rm -f "${heavy_hits}"
}

# Run checks that are expected to work with the reviewer dependency set.
main_logic() {
  local output_file="$1"
  local pytest_paths=(
    "tests/test_setup_config.py"
  )

  {
    echo "[+] Running reviewer CI checks."
    echo "[+] Output file: ${output_file}"
    echo ""
    echo "Note: full pytest/local-ci.sh is expected to fail without torch."

    run_step "Checking for forbidden heavy packages..." \
      check_no_heavy_packages

    run_step "Running Ruff lint check..." \
      pdm run ruff check .

    run_step "Running Ruff format check..." \
      pdm run ruff format --check .

    run_step "Running reviewer-safe pytest paths..." \
      pdm run pytest -q "${pytest_paths[@]}"

    echo ""
    echo "[+] Reviewer CI check successful."
  } | tee "${output_file}"
}

# Main function to encapsulate script logic.
main() {
  local output_file="${DEFAULT_OUTPUT_FILE}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -o|--output_file)
        if [[ $# -lt 2 ]]; then
          error_exit "Missing value for $1."
        fi
        output_file="$2"
        shift 2
        ;;
      -h|--help)
        show_help
        exit 0
        ;;
      *)
        error_exit "Invalid option: $1"
        ;;
    esac
  done

  main_logic "${output_file}"
}

# Header ASCII art.
print_ascii_art

# Execute the main function.
main "$@"
