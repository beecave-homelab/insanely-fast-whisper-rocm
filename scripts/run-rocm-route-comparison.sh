#!/bin/bash
set -euo pipefail

# Script Description: Compare ROCm Docker route builds and diagnostics.
# Author: elvee
# Version: 0.2.0
# License: MIT
# Creation Date: 31/05/2026
# Last Modified: 31/05/2026
# Usage: run-rocm-route-comparison.sh [OPTIONS]

# Constants
DEFAULT_COMPOSE_FILE="docker-compose.test.rocm-routes.yaml"
DEFAULT_RESULT_ROOT="test-results/rocm-route-comparison"
DEFAULT_DROP_TO_SHELL="0"
SERVICE_PREFIX="insanely-fast-whisper-rocm-test-"
DIAGNOSTICS_FILE="rocm-container-diagnostics.txt"
STALE_TOP_LEVEL_RESULTS=(
  "api-chunk-result.json"
  "api-word-result.json"
  "webui-chunk-result.json"
  "webui-word-result.json"
)

DEFAULT_SERVICES=(
  "insanely-fast-whisper-rocm-test-rocm-pytorch"
  "insanely-fast-whisper-rocm-test-rocm-pytorch-jit"
  "insanely-fast-whisper-rocm-test-slim-rocm-apt"
  "insanely-fast-whisper-rocm-test-slim-rocm-apt-jit"
)

# ASCII Art (Calvin font)
print_ascii_art() {
  echo "
       ██████╗  ██████╗  ██████╗███╗   ███╗       
       ██╔══██╗██╔═══██╗██╔════╝████╗ ████║       
       ██████╔╝██║   ██║██║     ██╔████╔██║       
       ██╔══██╗██║   ██║██║     ██║╚██╔╝██║       
       ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║       
       ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝       
                                                   
 ██████╗  ██████╗  ██████╗██╗  ██╗███████╗██████╗   
 ██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝██╔════╝██╔══██╗  
 ██║  ██║██║   ██║██║     █████╔╝ █████╗  ██████╔╝  
 ██║  ██║██║   ██║██║     ██╔═██╗ ██╔══╝  ██╔══██╗  
 ██████╔╝╚██████╔╝╚██████╗██║  ██╗███████╗██║  ██║  
 ╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝  
                                                   
██████╗  ██████╗ ██╗   ██╗████████╗███████╗███████╗
██╔══██╗██╔═══██╗██║   ██║╚══██╔══╝██╔════╝██╔════╝
██████╔╝██║   ██║██║   ██║   ██║   █████╗  ███████╗
██╔══██╗██║   ██║██║   ██║   ██║   ██╔══╝  ╚════██║
██║  ██║╚██████╔╝╚██████╔╝   ██║   ███████╗███████║
╚═╝  ╚═╝ ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚══════╝                                     
"
}

# Help
show_help() {
  echo "
Usage: $0 [OPTIONS]

Build and run ROCm Docker route services, then collect comparison artifacts.

Options:
  -f, --compose-file FILE     Docker Compose file
                             (default: ${DEFAULT_COMPOSE_FILE})
  -o, --output-dir DIR        Directory for comparison artifacts
                             (default: ${DEFAULT_RESULT_ROOT})
  -s, --service SERVICE       Service to run; repeat for multiple services
                             (default: all known ROCm route services)
  --drop-to-shell             Set TEST_DROP_TO_SHELL=1 for service runs
  --dry-run                   Print planned actions without running Docker
  --verbose                   Stream build and run output while saving logs
  --no-header                 Do not print the ASCII art header
  --force-build               Rebuild images even if they already exist
  -h, --help                  Show this help message

Environment:
  COMPOSE_FILE                Overrides the default compose file.
  RESULT_ROOT                 Overrides the default output directory.
  TEST_DROP_TO_SHELL          Overrides the default run shell behavior.

Examples:
  $0
  $0 --output-dir test-results/routes
  $0 --dry-run --service insanely-fast-whisper-rocm-test-rocm-pytorch
  $0 --service insanely-fast-whisper-rocm-test-rocm-pytorch
"
}

# Error handling
error_exit() {
  echo "Error: $1" >&2
  exit 1
}

log_info() {
  echo "[+] $1"
}

log_warn() {
  echo "[!] $1" >&2
}

require_command() {
  local command_name="$1"

  command -v "${command_name}" >/dev/null 2>&1 \
    || error_exit "Required command not found: ${command_name}"
}

validate_inputs() {
  local compose_file="$1"
  local result_root="$2"

  [[ -n "${compose_file}" ]] || error_exit "Compose file cannot be empty."
  [[ -f "${compose_file}" ]] || error_exit "Compose file not found: ${compose_file}"
  [[ -n "${result_root}" ]] || error_exit "Output directory cannot be empty."

  require_command "docker"
  [[ -x "/usr/bin/time" ]] || error_exit "Required command not found: /usr/bin/time"
}

write_summary_header() {
  local summary_file="$1"
  local service="$2"
  local compose_file="$3"

  {
    echo "service=${service}"
    echo "compose_file=${compose_file}"
    echo "started_at=$(date --iso-8601=seconds)"
  } >"${summary_file}"
}

capture_image_info() {
  local compose_file="$1"
  local service="$2"
  local image_log="$3"

  docker compose -f "${compose_file}" images "${service}" \
    >"${image_log}" 2>&1 || true
}

image_exists() {
  local compose_file="$1"
  local service="$2"

  docker compose -f "${compose_file}" images "${service}" \
    --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -q .
}

clean_stale_top_level_results() {
  local result_root="$1"
  local result_parent=""
  local result_file=""

  result_parent="$(dirname "${result_root}")"

  if [[ "${result_parent}" != "test-results" ]]; then
    return
  fi

  for result_file in "${STALE_TOP_LEVEL_RESULTS[@]}"; do
    if [[ -f "${result_parent}/${result_file}" ]]; then
      rm -f "${result_parent}/${result_file}"
      log_info "Removed stale top-level result: ${result_parent}/${result_file}"
    fi
  done
}

copy_diagnostics() {
  local result_root="$1"
  local service="$2"
  local service_dir="$3"
  local route_name="${service#${SERVICE_PREFIX}}"
  local source_file="test-results/${route_name}/${DIAGNOSTICS_FILE}"
  local target_file="${service_dir}/${DIAGNOSTICS_FILE}"

  if [[ -f "${source_file}" ]]; then
    cp "${source_file}" "${target_file}"
    echo "diagnostics=${target_file}"
    return
  fi

  log_warn "Diagnostics not found for ${service}: ${source_file}"
  echo "diagnostics_missing=${source_file}" \
    >>"${result_root}/missing-diagnostics.txt"
}

run_timed_command() {
  local verbose="$1"
  local log_file="$2"
  shift 2

  if [[ "${verbose}" == "true" ]]; then
    /usr/bin/time -f "elapsed=%E max_rss_kb=%M" "$@" 2>&1 | tee "${log_file}"
    return "${PIPESTATUS[0]}"
  fi

  /usr/bin/time -f "elapsed=%E max_rss_kb=%M" "$@" >"${log_file}" 2>&1
}

print_dry_run_plan() {
  local compose_file="$1"
  local result_root="$2"
  local drop_to_shell="$3"
  local verbose="$4"
  local force_build="$5"
  shift 5
  local services=("$@")
  local service=""
  local route_name=""
  local service_dir=""

  log_info "Dry run only; no directories, builds, containers, or logs created."
  log_info "Compose file: ${compose_file}"
  log_info "Output directory: ${result_root}"
  log_info "TEST_DROP_TO_SHELL=${drop_to_shell}"
  log_info "Verbose output: ${verbose}"
  log_info "Force build: ${force_build}"
  log_info "Service count: ${#services[@]}"
  echo ""

  for service in "${services[@]}"; do
    route_name="${service#${SERVICE_PREFIX}}"
    service_dir="${result_root}/${service}"
    echo "Service: ${route_name}"
    echo "  full_name: ${service}"
    echo "  commands:"
    if [[ "${force_build}" != "true" ]] && image_exists "${compose_file}" "${service}"; then
      echo "    docker compose -f <compose-file> images <service>  (build skipped — image exists)"
    else
      echo "    docker compose -f <compose-file> build <service>"
      echo "    docker compose -f <compose-file> images <service>"
    fi
    echo "    TEST_DROP_TO_SHELL=${drop_to_shell} \\"
    echo "      docker compose -f <compose-file> run --rm --no-deps <service>"
    echo "  artifacts:"
    echo "    directory: ${service_dir}"
    echo "    files: build.log, run.log, image.txt, summary.txt"
    echo "    optional: ${DIAGNOSTICS_FILE}"
    echo ""
  done

  echo "Legend:"
  echo "  <compose-file>: ${compose_file}"
  echo "  <service>: full_name from each service block"
}

run_and_capture() {
  local compose_file="$1"
  local result_root="$2"
  local drop_to_shell="$3"
  local verbose="$4"
  local force_build="$5"
  local service="$6"
  local service_dir="${result_root}/${service}"
  local build_log="${service_dir}/build.log"
  local run_log="${service_dir}/run.log"
  local image_log="${service_dir}/image.txt"
  local summary_file="${service_dir}/summary.txt"
  local build_exit=0
  local run_exit=0

  mkdir -p "${service_dir}"
  write_summary_header "${summary_file}" "${service}" "${compose_file}"

  if [[ "${force_build}" == "true" ]] || ! image_exists "${compose_file}" "${service}"; then
    log_info "${service}: build"
    set +e
    run_timed_command \
      "${verbose}" \
      "${build_log}" \
      docker compose -f "${compose_file}" build "${service}"
    build_exit=$?
  else
    log_info "${service}: image already built (use --force-build to rebuild)"
    echo "build_exit=skipped" >>"${summary_file}"
  fi

  echo "build_exit=${build_exit}" >>"${summary_file}"
  capture_image_info "${compose_file}" "${service}" "${image_log}"

  if (( build_exit != 0 )); then
    log_warn "Build failed for ${service}; see ${build_log}"
    return "${build_exit}"
  fi

  log_info "${service}: run"
  set +e
  TEST_DROP_TO_SHELL="${drop_to_shell}" run_timed_command \
    "${verbose}" \
    "${run_log}" \
    docker compose -f "${compose_file}" run --rm --no-deps "${service}"
  run_exit=$?

  {
    echo "run_exit=${run_exit}"
    copy_diagnostics "${result_root}" "${service}" "${service_dir}"
    echo "build_log=${build_log}"
    echo "run_log=${run_log}"
    echo "image_log=${image_log}"
    echo "finished_at=$(date --iso-8601=seconds)"
  } >>"${summary_file}"

  if (( run_exit != 0 )); then
    log_warn "Run failed for ${service}; see ${run_log}"
    return "${run_exit}"
  fi
}

main_logic() {
  local compose_file="$1"
  local result_root="$2"
  local drop_to_shell="$3"
  local verbose="$4"
  local force_build="$5"
  shift 5
  local services=("$@")
  local service=""
  local overall_exit=0

  mkdir -p "${result_root}"
  clean_stale_top_level_results "${result_root}"
  : >"${result_root}/missing-diagnostics.txt"

  log_info "Compose file: ${compose_file}"
  log_info "Output directory: ${result_root}"
  log_info "Verbose output: ${verbose}"
  log_info "Services: ${#services[@]}"
  echo ""

  for service in "${services[@]}"; do
    set +e
    run_and_capture \
      "${compose_file}" \
      "${result_root}" \
      "${drop_to_shell}" \
      "${verbose}" \
      "${force_build}" \
      "${service}"
    local service_exit=$?
    set -e

    if [[ "${service_exit}" -eq 130 ]]; then
      log_warn "Interrupted; stopping remaining services."
      return 130
    fi

    if [[ "${service_exit}" -ne 0 ]]; then
      overall_exit=1
    fi
  done

  log_info "Comparison artifacts written to ${result_root}"
  return "${overall_exit}"
}

# Main
main() {
  local compose_file="${COMPOSE_FILE:-${DEFAULT_COMPOSE_FILE}}"
  local result_root="${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"
  local drop_to_shell="${TEST_DROP_TO_SHELL:-${DEFAULT_DROP_TO_SHELL}}"
  local dry_run="false"
  local print_header="true"
  local verbose="false"
  local force_build="false"
  local services=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -f|--compose-file)
        [[ $# -ge 2 ]] || error_exit "Missing value for $1."
        compose_file="$2"
        shift 2
        ;;
      -o|--output-dir)
        [[ $# -ge 2 ]] || error_exit "Missing value for $1."
        result_root="$2"
        shift 2
        ;;
      -s|--service)
        [[ $# -ge 2 ]] || error_exit "Missing value for $1."
        services+=("$2")
        shift 2
        ;;
      --drop-to-shell)
        drop_to_shell="1"
        shift
        ;;
      --dry-run)
        dry_run="true"
        shift
        ;;
      --verbose)
        verbose="true"
        shift
        ;;
      --no-header)
        print_header="false"
        shift
        ;;
      --force-build)
        force_build="true"
        shift
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

  if [[ "${#services[@]}" -eq 0 ]]; then
    services=("${DEFAULT_SERVICES[@]}")
  fi

  validate_inputs "${compose_file}" "${result_root}"

  if [[ "${print_header}" == "true" ]]; then
    print_ascii_art
  fi

  if [[ "${dry_run}" == "true" ]]; then
    print_dry_run_plan \
      "${compose_file}" \
      "${result_root}" \
      "${drop_to_shell}" \
      "${verbose}" \
      "${force_build}" \
      "${services[@]}"
    exit 0
  fi

  main_logic "${compose_file}" "${result_root}" "${drop_to_shell}" "${verbose}" \
    "${force_build}" "${services[@]}"
}

# Execute
main "$@"
