#!/bin/bash
set -euo pipefail

# Script Description: Monitors a specified directory for file changes, processes each new or modified file using the insanely-fast-whisper command, and saves the output to a transcript directory.
# Author: elvee
# Version: 0.5.0
# License: MIT
# Creation Date: 16-08-2024
# Last Modified: 16-08-2024

# Default Values
DEFAULT_UPLOADS="uploads"             # Directory to watch for file changes
DEFAULT_TRANSCRIPTS="transcripts"     # Directory to save the transcript files
DEFAULT_LOGS="logs"                   # Directory to save log files
DEFAULT_BATCH_SIZE=10                 # Batch size for processing
DEFAULT_VERBOSE=false                 # Default value for verbose logging

# Function to display help
show_help() {
  echo "
Usage: $0 [OPTIONS]

Options:
  -u, --uploads DIR                   Directory to watch for file changes (default: $DEFAULT_UPLOADS)
  -t, --transcripts DIR               Directory to save transcript files (default: $DEFAULT_TRANSCRIPTS)
  -l, --logs DIR                      Directory to save log files (default: $DEFAULT_LOGS)
  -b, --batch-size SIZE               Batch size for processing (default: $DEFAULT_BATCH_SIZE)
  -v, --verbose                       Enable verbose logging (default: $DEFAULT_VERBOSE)
  -h, --help                          Show this help message

Examples:
  $0 -u uploads -t transcripts -l logs -b 4 -v
"
}

# Function for error handling
error_exit() {
  echo "Error: $1" >&2
  exit 1
}

# Load .env file if it exists
if [[ -f .env ]]; then
  set -o allexport
  source .env
  set -o allexport
else
  echo "No .env file found. Using default values."
fi

# Assign values from .env or use defaults
UPLOADS="${UPLOADS:-$DEFAULT_UPLOADS}"
TRANSCRIPTS="${TRANSCRIPTS:-$DEFAULT_TRANSCRIPTS}"
LOGS="${LOGS:-$DEFAULT_LOGS}"
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
VERBOSE="${VERBOSE:-$DEFAULT_VERBOSE}"

# Log if default values are used
[[ "$UPLOADS" == "$DEFAULT_UPLOADS" ]] && echo "UPLOADS not set in .env, using default: $UPLOADS"
[[ "$TRANSCRIPTS" == "$DEFAULT_TRANSCRIPTS" ]] && echo "TRANSCRIPTS not set in .env, using default: $TRANSCRIPTS"
[[ "$LOGS" == "$DEFAULT_LOGS" ]] && echo "LOGS not set in .env, using default: $LOGS"
[[ "$BATCH_SIZE" == "$DEFAULT_BATCH_SIZE" ]] && echo "BATCH_SIZE not set in .env, using default: $BATCH_SIZE"
[[ "$VERBOSE" == "$DEFAULT_VERBOSE" ]] && echo "VERBOSE not set in .env, using default: $VERBOSE"

# Function for verbose logging
log_verbose() {
  if [[ "$VERBOSE" = true ]]; then
    echo "$1" >> "$2"
  fi
}

# Main function to encapsulate script logic
main() {
  # Parse command-line options
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -u|--uploads)
        UPLOADS="$2"
        shift 2
        ;;
      -t|--transcripts)
        TRANSCRIPTS="$2"
        shift 2
        ;;
      -l|--logs)
        LOGS="$2"
        shift 2
        ;;
      -b|--batch-size)
        BATCH_SIZE="$2"
        shift 2
        ;;
      -v|--verbose)
        VERBOSE=true
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

  # Validate directories
  for dir in "$UPLOADS" "$TRANSCRIPTS" "$LOGS"; do
    if [[ ! -d "$dir" ]]; then
      error_exit "Directory not found: $dir"
    fi
  done

  # Continuous monitoring loop
  while true; do
    files_found=false

    for filepath in "$UPLOADS"/*; do
      if [[ -f "$filepath" ]]; then
        files_found=true
        filename=$(basename "$filepath")
        transcript_output="${TRANSCRIPTS}/${filename%.*}.txt"
        log_file="${LOGS}/${filename%.*}.log"

        # Check if the transcript already exists
        if [[ -f "$transcript_output" ]]; then
          echo "Skipping ${filename}, transcript already exists."
          log_verbose "Skipped file: ${filename}. Transcript already exists." "$log_file"
          continue
        fi

        # Define the command with the constructed paths and batch size
        command="insanely-fast-whisper --file-name ${filepath} --transcript-path ${transcript_output} --batch-size ${BATCH_SIZE}"

        # Log verbose information
        log_verbose "Processing file: ${filename}" "$log_file"
        log_verbose "Command: ${command}" "$log_file"

        # Execute the command and log the output
        echo "Processing ${filename}..."
        ${command} &>> "$log_file"

        echo "${filename} processed successfully. Log saved to ${log_file}."
        log_verbose "File processed: ${filename}" "$log_file"
      fi
    done

    # If no files were found, wait for 1 minute before checking again
    if [[ "$files_found" = false ]]; then
      echo "No new files found. Waiting for 1 minute..."
      sleep 60
    fi
  done
}

# Execute the main function
main "$@"
