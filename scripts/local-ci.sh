#!/bin/bash
set -euo pipefail

# Script Description: Run local CI pipeline (fix, format, test, coverage)
# Author: elvee
# Version: 0.1.0
# License: MIT
# Creation Date: 03/12/2025
# Last Modified: 03/12/2025
# Usage: local-ci.sh

# Constants
DEFAULT_OUTPUT_FILE="${PWD}/ci-output.log"

# ASCII Art (Calvin font)
print_ascii_art() {
  echo "
╦    ╔═╗  ╔═╗  ╔═╗  ╦         ╔═╗  ╦
║    ║ ║  ║    ╠═╣  ║    ───  ║    ║
╩═╝  ╚═╝  ╚═╝  ╩ ╩  ╩═╝       ╚═╝  ╩
"
}

# Help
show_help() {
  echo "
Usage: $0 [OPTIONS]

Options:
  -o, --output_file FILE     Write CI logs to file (default: $DEFAULT_OUTPUT_FILE)
  -h, --help                 Show help

This script performs:
  • pdm run fix
  • pdm run format
  • pdm run test
  • pdm run test-cov
"
}

# Error handling
error_exit() {
  echo "Error: $1" >&2
  exit 1
}

# Main logic
main_logic() {
  echo "[+] The following tasks will be executed:"
  echo "    • pdm run fix"
  echo "    • pdm run format"
  echo "    • pdm run test"
  echo "    • pdm run test-cov"
  echo ""

  local output_file="$1"

  {
    echo "[+] Running fix..."
    echo ""
    pdm run fix
    echo ""
    echo "[+] Running format..."
    pdm run format
    echo ""
    echo "[+] Running tests..."
    pdm run test
    echo ""
    echo "[+] Running test coverage..."
    pdm run test-cov
    echo ""
    echo "[+] Local CI check successful. You can commit these changes."
  } | tee "${output_file}"
}

# Main
main() {
  local output_file="$DEFAULT_OUTPUT_FILE"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -o|--output_file)
        if [[ $# -lt 2 ]]; then
          echo "Error: Option '$1' requires a file path argument" >&2
          show_help
          exit 1
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

  main_logic "$output_file"
}

# Header ASCII art
print_ascii_art

# Execute
main "$@"
