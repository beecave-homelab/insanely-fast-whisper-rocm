#!/bin/bash
set -euo pipefail

# Script Description: Move all ".windsurf/workflows/*.md" (from current working directory)
#                    into "~/.codeium/windsurf/global_workflows/". Per-file confirmation,
#                    --dry-run support (never prompts), conflict handling, and colored UX.
# Author: elvee
# Version: 0.3.0
# License: MIT
# Creation Date: 09/10/2025
# Last Modified: 09/10/2025
# Usage: mww [OPTIONS]

# Defaults
DEFAULT_SOURCE_DIR="${PWD}/.windsurf/workflows"
DEFAULT_DEST_DIR="${HOME}/.codeium/windsurf/global_workflows"
SCRIPT_NAME="mww"
DEFAULT_EXT="md"

# ──────────────────────────────────────────────────────────────────────────────
# Color setup (auto-detect; --no-color or NO_COLOR disables)
# ──────────────────────────────────────────────────────────────────────────────
USE_COLOR=1
if [[ -n "${NO_COLOR:-}" ]]; then USE_COLOR=0; fi
if [[ ! -t 1 ]]; then USE_COLOR=0; fi
if command -v tput >/dev/null 2>&1; then
  if ! tput colors >/dev/null 2>&1; then USE_COLOR=0; fi
else
  USE_COLOR=0
fi

if [[ "${USE_COLOR}" -eq 1 ]]; then
  BOLD="$(tput bold)"; DIM="$(tput dim)"; RESET="$(tput sgr0)"
  RED="$(tput setaf 1)"; GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"
  BLUE="$(tput setaf 4)"; MAGENTA="$(tput setaf 5)"; CYAN="$(tput setaf 6)"
  GRAY="$(tput setaf 7)"
else
  BOLD=""; DIM=""; RESET=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; MAGENTA=""; CYAN=""; GRAY=""
fi

# Icons (ASCII-safe)
ICON_INFO="i"
ICON_OK="✓"
ICON_WARN="!"
ICON_ERR="✗"
ICON_MOVE="→"
ICON_SKIP="⤼"
ICON_BACKUP="⟲"
ICON_DRY="◻"

# Calvin ASCII art for "MWW"
print_ascii_art() {
  printf "%s" "${MAGENTA}"
  cat <<'EOF'
┌┬┐  ┬ ┬  ┬ ┬
│││  │││  │││
┴ ┴  └┴┘  └┴┘
EOF
  printf "%s" "${RESET}"
}

show_help() {
  cat <<EOF

${BOLD}Usage:${RESET} ${SCRIPT_NAME} [OPTIONS]

Move all Markdown workflows from ".windsurf/workflows/*.md" (in the current
directory) into "~/.codeium/windsurf/global_workflows/".

${BOLD}Options:${RESET}
  -n, --dry-run               Show actions without changing anything (forces --verbose, disables --quiet)
  -y, --yes                   Skip per-file move confirmation (but NOT overwrite prompts)
  -s, --source DIR            Source directory (default: ${DEFAULT_SOURCE_DIR})
  -d, --dest DIR              Destination directory (default: ${DEFAULT_DEST_DIR})
  -e, --ext EXT               File extension to match (default: ${DEFAULT_EXT})
      --overwrite             Overwrite destination files without asking
      --backup                If destination exists, back it up as "<name>.bak.<timestamp>"
      --no-clobber            Never overwrite existing files (skip on conflict; default)
      --no-color              Disable colored output
  -v, --verbose               Verbose logging
  -q, --quiet                 Minimal output (errors only)
  -h, --help                  Show this help and exit
  -V, --version               Show version and exit

${BOLD}Dry-run:${RESET}
  • Never prompts.
  • Prints planned actions with ${CYAN}[${ICON_DRY} dry-run]${RESET} markers.
  • Forces verbose output and disables quiet.

EOF
}

error_exit() { printf "%b\n" "${RED}[${ICON_ERR}] Error:${RESET} $1" >&2; exit 1; }

prompt_yes_no() {
  local prompt="$1" reply
  while true; do
    read -r -p "$(printf "%b" "${YELLOW}[?]${RESET} ${prompt} [y/N]: ")" reply || reply="n"
    case "${reply,,}" in
      y|yes) return 0 ;;
      n|no|"") return 1 ;;
      *) printf "%b\n" "${YELLOW}[${ICON_WARN}]${RESET} Please answer 'y' or 'n'." ;;
    esac
  done
}

log()  { [[ "${QUIET}" -eq 0 ]] && printf "%b\n" "$*"; }
vlog() { [[ "${VERBOSE}" -eq 1 && "${QUIET}" -eq 0 ]] && printf "%b\n" "$*"; }

now_ts() { date +"%Y%m%d-%H%M%S"; }

# ──────────────────────────────────────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────────────────────────────────────
move_markdown_workflows() {
  local src="$1" dst="$2" dry="$3" assume_yes="$4" overwrite="$5" no_clobber="$6" do_backup="$7" ext="$8"

  if [[ "${dry}" -eq 0 ]]; then
    mkdir -p "${dst}" || error_exit "Unable to create dest dir: ${dst}"
  else
    vlog "${CYAN}[${ICON_DRY} dry-run]${RESET} Would ensure destination exists: ${BOLD}${dst}${RESET}"
  fi

  shopt -s nullglob
  local files=( "${src}"/*.${ext} )
  shopt -u nullglob

  if (( ${#files[@]} == 0 )); then
    log "${YELLOW}[${ICON_WARN}]${RESET} No '*.${ext}' files found in: ${BOLD}${src}${RESET}"
    exit 2
  fi

  local moved=0 skipped=0
  for f in "${files[@]}"; do
    local base dest_path
    base="$(basename -- "$f")"
    dest_path="${dst}/${base}"

    if [[ "${dry}" -eq 0 && "${assume_yes}" -eq 0 ]]; then
      if ! prompt_yes_no "Move '${f}' ${ICON_MOVE} '${dest_path}'?"; then
        vlog "${GRAY}[${ICON_SKIP}] Skipped by user:${RESET} ${f}"
        ((skipped+=1))
        continue
      fi
    fi

    # Destination conflict handling
    if [[ -e "${dest_path}" ]]; then
      if [[ "${overwrite}" -eq 1 ]]; then
        : # overwrite silently
      elif [[ "${do_backup}" -eq 1 ]]; then
        local bak="${dest_path}.bak.$(now_ts)"
        if [[ "${dry}" -eq 1 ]]; then
          printf "%b\n" "${CYAN}[${ICON_DRY} dry-run]${RESET} ${BLUE}${ICON_BACKUP}${RESET} cp -- '${dest_path}' '${bak}'  ${DIM}# backup existing${RESET}"
        else
          vlog "${BLUE}${ICON_BACKUP}${RESET} Backing up existing: ${dest_path} ${ICON_MOVE} ${bak}"
          cp -- "${dest_path}" "${bak}"
        fi
      elif [[ "${no_clobber}" -eq 1 ]]; then
        if [[ "${dry}" -eq 1 ]]; then
          printf "%b\n" "${CYAN}[${ICON_DRY} dry-run]${RESET} ${GRAY}[${ICON_SKIP}]${RESET} '${dest_path}' exists (no-clobber)"
        else
          vlog "${GRAY}[${ICON_SKIP}]${RESET} Destination exists, no-clobber; skipping: ${dest_path}"
        fi
        ((skipped+=1)); continue
      else
        if [[ "${dry}" -eq 1 ]]; then
          printf "%b\n" "${CYAN}[${ICON_DRY} dry-run]${RESET} Destination exists: '${dest_path}'. Would prompt to overwrite."
          printf "%b\n" "${CYAN}[${ICON_DRY} dry-run]${RESET} Tip: use --overwrite, --no-clobber, or --backup."
          ((skipped+=1)); continue
        else
          if ! prompt_yes_no "Destination exists: '${dest_path}'. Overwrite it?"; then
            vlog "${GRAY}[${ICON_SKIP}]${RESET} User chose not to overwrite: ${dest_path}"
            ((skipped+=1)); continue
          fi
        fi
      fi
    fi

    # Execute (or show) move
    if [[ "${dry}" -eq 1 ]]; then
      printf "%b\n" "${CYAN}[${ICON_DRY} dry-run]${RESET} mv -- '${f}' '${dest_path}'"
    else
      vlog "${GREEN}[${ICON_MOVE}]${RESET} Moving: ${f} ${ICON_MOVE} ${dest_path}"
      mv -- "${f}" "${dest_path}"
    fi
    ((moved+=1))
  done

  log "${GREEN}[${ICON_OK}]${RESET} Done. ${BOLD}Moved:${RESET} ${moved}, ${BOLD}Skipped:${RESET} ${skipped}"
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
main() {
  local source_dir="${DEFAULT_SOURCE_DIR}"
  local dest_dir="${DEFAULT_DEST_DIR}"
  local ext="${DEFAULT_EXT}"
  local DRY_RUN=0 ASSUME_YES=0 OVERWRITE=0 NO_CLOBBER=1 DO_BACKUP=0
  VERBOSE=0 QUIET=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -n|--dry-run|--no-act) DRY_RUN=1; shift ;;
      -y|--yes)              ASSUME_YES=1; shift ;;
      -s|--source)           [[ $# -lt 2 ]] && error_exit "Missing DIR for $1"; source_dir="$2"; shift 2 ;;
      -d|--dest|--destination) [[ $# -lt 2 ]] && error_exit "Missing DIR for $1"; dest_dir="$2"; shift 2 ;;
      -e|--ext)              [[ $# -lt 2 ]] && error_exit "Missing EXT for $1"; ext="${2#.}"; shift 2 ;;
      --overwrite)           OVERWRITE=1; NO_CLOBBER=0; DO_BACKUP=0; shift ;;
      --backup)              DO_BACKUP=1; OVERWRITE=0; NO_CLOBBER=0; shift ;;
      --no-clobber)          OVERWRITE=0; DO_BACKUP=0; NO_CLOBBER=1; shift ;;
      --no-color)            USE_COLOR=0; BOLD=""; DIM=""; RESET=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; MAGENTA=""; CYAN=""; GRAY=""; shift ;;
      -v|--verbose)          VERBOSE=1; shift ;;
      -q|--quiet)            QUIET=1; VERBOSE=0; shift ;;
      -h|--help)             show_help; exit 0 ;;
      -V|--version)          echo "${SCRIPT_NAME} version 0.3.0"; exit 0 ;;
      *) error_exit "Invalid option: $1 (try --help)" ;;
    esac
  done

  # Force visibility for dry-run
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    VERBOSE=1
    QUIET=0
  fi

  # Banner
  print_ascii_art

  # Validation (never block in dry-run)
  if [[ ! -d "${source_dir}" ]]; then
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf "%b\n" "${CYAN}[${ICON_DRY} dry-run]${RESET} Source directory does not exist: ${BOLD}${source_dir}${RESET}"
      printf "%b\n" "${CYAN}[${ICON_DRY} dry-run]${RESET} Would search for '*.${ext}' but nothing to do."
      exit 2
    else
      error_exit "Source directory not found: ${source_dir}"
    fi
  fi

  if [[ "${source_dir}" != "${PWD}/.windsurf/workflows" ]]; then
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf "%b\n" "${CYAN}[${ICON_DRY} dry-run]${RESET} Warning: Source is not '\${PWD}/.windsurf/workflows': ${BOLD}${source_dir}${RESET}"
    elif [[ "${ASSUME_YES}" -eq 0 ]]; then
      printf "%b" "${YELLOW}[${ICON_WARN}]${RESET} Source dir is not '\${PWD}/.windsurf/workflows': ${BOLD}${source_dir}${RESET}\n"
      if ! prompt_yes_no "Continue with this custom source?"; then exit 1; fi
    fi
  fi

  move_markdown_workflows "${source_dir}" "${dest_dir}" \
    "${DRY_RUN}" "${ASSUME_YES}" "${OVERWRITE}" "${NO_CLOBBER}" "${DO_BACKUP}" "${ext}"
}

main "$@"
