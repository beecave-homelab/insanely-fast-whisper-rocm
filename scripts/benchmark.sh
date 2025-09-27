#!/bin/bash

set -e

# Advanced, adjustable benchmarking wrapper for `pdm run cli transcribe`.
#
# Features:
# - Select model, device, dtype, language
# - Run one or more audio files
# - Choose timestamp types: chunk, word (one or both)
# - Configure batch sizes per timestamp type
# - Control export format, progress, quiet, repeats
# - Optional stable-ts post-processing: stabilize, demucs, vad, vad-threshold
# - Pass benchmark extras (KEY=VALUE)
# - Dry-run mode to preview commands
#
# Examples:
#   scripts/benchmark.sh \
#     --audio uploads/foo.mp3 --audio uploads/bar.wav \
#     --model openai/whisper-medium \
#     --timestamp-types chunk,word \
#     --batch-sizes 12,4 \
#     --export-format srt --benchmark --quiet
#
#   scripts/benchmark.sh \
#     -a uploads/foo.mp3 -m distil-whisper/distil-large-v3 \
#     --device cuda:0 --dtype float16 --language en \
#     --stabilize --vad --vad-threshold 0.35 \
#     --demucs --repeats 2 --progress --no-quiet

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# Defaults (can be overridden by flags)
MODEL="openai/whisper-medium"
DEVICE=""              # Let CLI defaults apply when empty
DTYPE=""               # float16|float32
LANGUAGE=""            # e.g. en; empty=auto
EXPORT_FORMAT="srt"     # all|json|srt|txt
TIMESTAMP_TYPES=(chunk word)
BATCH_CHUNK="12"        # default batch for chunk timestamps
BATCH_WORD="4"          # default batch for word timestamps
PROGRESS_GROUP_SIZE=""  # let CLI default when empty
CHUNK_LENGTH=""         # let CLI default when empty
NO_TIMESTAMPS=false
STABILIZE=false
DEMUCS=false
VAD=false
VAD_THRESHOLD=""        # e.g. 0.35
QUIET=true
PROGRESS=true
BENCHMARK=true
REPEATS=1
DRY_RUN=false
OUTPUT_PATH=""          # optional single output path when export-format != all

declare -a AUDIO_FILES
declare -a BENCHMARK_EXTRAS

print_usage() {
  cat <<USAGE
Usage: scripts/benchmark.sh [options]

Options:
  -a, --audio PATH              Audio file to process (can be repeated)
  -m, --model NAME              Model name (default: $MODEL)
      --device NAME             Device for inference (e.g., cuda:0, cpu)
      --dtype {float16,float32} Data type for inference
  -l, --language CODE           Language code (empty=auto)
      --export-format FMT       Export format: all|json|srt|txt (default: $EXPORT_FORMAT)
      --timestamp-types LIST    Comma-separated: chunk,word (default: chunk,word)
      --batch-sizes LIST        Comma-separated: CHUNK,WORD (default: ${BATCH_CHUNK},${BATCH_WORD})
      --batch-chunk N           Batch size for chunk timestamps (overrides first item)
      --batch-word N            Batch size for word timestamps (overrides second item)
      --progress-group-size N   Chunks per progress update
      --chunk-length N          Audio chunk length in seconds
      --no-timestamps           Disable timestamps entirely

  Stable-ts options:
      --stabilize | --no-stabilize    Enable/disable stable-ts post-processing (default: ${STABILIZE})
      --demucs | --no-demucs          Enable/disable Demucs in stable-ts (default: ${DEMUCS})
      --vad | --no-vad                Enable/disable VAD in stable-ts (default: ${VAD})
      --vad-threshold F               VAD threshold (e.g., 0.35)

  Output & UX:
      --output PATH             Explicit output file path (single-format only)
      --benchmark | --no-benchmark    Toggle benchmark collection (default: ${BENCHMARK})
      --benchmark-extra K=V     Extra benchmark field (repeatable)
      --quiet | --no-quiet      Minimize console output (default: ${QUIET})
      --progress | --no-progress Show progress UI (default: ${PROGRESS})
      --repeats N               Run each configuration N times (default: ${REPEATS})
      --dry-run                 Print commands without running them

  -h, --help                    Show this help and exit

Examples:
  scripts/benchmark.sh \
    -a uploads/sample.mp3 -m openai/whisper-medium --timestamp-types word \
    --batch-word 4 --export-format srt --benchmark --quiet

  scripts/benchmark.sh \
    -a uploads/a.mp3 -a uploads/b.mp3 --batch-sizes 8,2 --repeats 2 \
    --stabilize --vad --vad-threshold 0.35
USAGE
}

# Parse arguments (supports both --opt=value and --opt value)
parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        print_usage; exit 0 ;;
      -a|--audio)
        AUDIO_FILES+=("$2"); shift 2 ;;
      --audio=*)
        AUDIO_FILES+=("${1#*=}"); shift ;;
      -m|--model)
        MODEL="$2"; shift 2 ;;
      --model=*)
        MODEL="${1#*=}"; shift ;;
      --device)
        DEVICE="$2"; shift 2 ;;
      --device=*)
        DEVICE="${1#*=}"; shift ;;
      --dtype)
        DTYPE="$2"; shift 2 ;;
      --dtype=*)
        DTYPE="${1#*=}"; shift ;;
      -l|--language)
        LANGUAGE="$2"; shift 2 ;;
      --language=*)
        LANGUAGE="${1#*=}"; shift ;;
      --export-format)
        EXPORT_FORMAT="$2"; shift 2 ;;
      --export-format=*)
        EXPORT_FORMAT="${1#*=}"; shift ;;
      --timestamp-types)
        IFS=',' read -r -a TIMESTAMP_TYPES <<< "$2"; shift 2 ;;
      --timestamp-types=*)
        IFS=',' read -r -a TIMESTAMP_TYPES <<< "${1#*=}"; shift ;;
      --batch-sizes)
        IFS=',' read -r BATCH_CHUNK BATCH_WORD <<< "$2"; shift 2 ;;
      --batch-sizes=*)
        IFS=',' read -r BATCH_CHUNK BATCH_WORD <<< "${1#*=}"; shift ;;
      --batch-chunk)
        BATCH_CHUNK="$2"; shift 2 ;;
      --batch-chunk=*)
        BATCH_CHUNK="${1#*=}"; shift ;;
      --batch-word)
        BATCH_WORD="$2"; shift 2 ;;
      --batch-word=*)
        BATCH_WORD="${1#*=}"; shift ;;
      --progress-group-size)
        PROGRESS_GROUP_SIZE="$2"; shift 2 ;;
      --progress-group-size=*)
        PROGRESS_GROUP_SIZE="${1#*=}"; shift ;;
      --chunk-length)
        CHUNK_LENGTH="$2"; shift 2 ;;
      --chunk-length=*)
        CHUNK_LENGTH="${1#*=}"; shift ;;
      --no-timestamps)
        NO_TIMESTAMPS=true; shift ;;
      --stabilize)
        STABILIZE=true; shift ;;
      --no-stabilize)
        STABILIZE=false; shift ;;
      --demucs)
        DEMUCS=true; shift ;;
      --no-demucs)
        DEMUCS=false; shift ;;
      --vad)
        VAD=true; shift ;;
      --no-vad)
        VAD=false; shift ;;
      --vad-threshold)
        VAD_THRESHOLD="$2"; shift 2 ;;
      --vad-threshold=*)
        VAD_THRESHOLD="${1#*=}"; shift ;;
      --output)
        OUTPUT_PATH="$2"; shift 2 ;;
      --output=*)
        OUTPUT_PATH="${1#*=}"; shift ;;
      --benchmark)
        BENCHMARK=true; shift ;;
      --no-benchmark)
        BENCHMARK=false; shift ;;
      --benchmark-extra)
        BENCHMARK_EXTRAS+=("$2"); shift 2 ;;
      --benchmark-extra=*)
        BENCHMARK_EXTRAS+=("${1#*=}"); shift ;;
      --quiet)
        QUIET=true; shift ;;
      --no-quiet)
        QUIET=false; shift ;;
      --progress)
        PROGRESS=true; shift ;;
      --no-progress)
        PROGRESS=false; shift ;;
      --repeats)
        REPEATS="$2"; shift 2 ;;
      --repeats=*)
        REPEATS="${1#*=}"; shift ;;
      --dry-run)
        DRY_RUN=true; shift ;;
      --)
        shift; break ;;
      *)
        echo "Unknown option: $1" >&2
        echo "Use --help for usage." >&2
        exit 2 ;;
    esac
  done
}

ensure_defaults() {
  # If no audio specified, try a sensible default used in this repo
  if [[ ${#AUDIO_FILES[@]} -eq 0 ]]; then
    DEFAULT_AUDIO="uploads/audio-overview-hoe-je-iets-vraagt-bepaalt-wat-je-krijgt-5.mp3"
    if [[ -f "$ROOT_DIR/$DEFAULT_AUDIO" ]]; then
      AUDIO_FILES+=("$DEFAULT_AUDIO")
    else
      echo "No --audio provided and default audio not found." >&2
      echo "Please provide at least one -a/--audio PATH." >&2
      exit 2
    fi
  fi

  # Normalize timestamp types to known values
  local normalized=()
  for t in "${TIMESTAMP_TYPES[@]}"; do
    case "$t" in
      chunk|word) normalized+=("$t") ;;
      *) echo "Ignoring unknown timestamp type: $t" >&2 ;;
    esac
  done
  if [[ ${#normalized[@]} -eq 0 && "$NO_TIMESTAMPS" != true ]]; then
    normalized=(chunk)
  fi
  TIMESTAMP_TYPES=("${normalized[@]}")
}

build_cmd() {
  local audio="$1"
  local ts_type="$2"
  local batch_size="$3"

  local cmd=(pdm run cli transcribe)
  cmd+=("--timestamp-type" "$ts_type")
  cmd+=("--export-format" "$EXPORT_FORMAT")
  cmd+=("--batch-size" "$batch_size")
  cmd+=("--model" "$MODEL")

  # Optional flags only when set
  if [[ -n "$DEVICE" ]]; then cmd+=("--device" "$DEVICE"); fi
  if [[ -n "$DTYPE" ]]; then cmd+=("--dtype" "$DTYPE"); fi
  if [[ -n "$LANGUAGE" ]]; then cmd+=("--language" "$LANGUAGE"); fi
  if [[ -n "$CHUNK_LENGTH" ]]; then cmd+=("--chunk-length" "$CHUNK_LENGTH"); fi
  if [[ -n "$PROGRESS_GROUP_SIZE" ]]; then cmd+=("--progress-group-size" "$PROGRESS_GROUP_SIZE"); fi
  if [[ "$NO_TIMESTAMPS" == true ]]; then cmd+=("--no-timestamps"); fi

  # stable-ts toggles
  if [[ "$STABILIZE" == true ]]; then cmd+=("--stabilize"); else cmd+=("--no-stabilize"); fi
  if [[ "$DEMUCS" == true ]]; then cmd+=("--demucs"); else cmd+=("--no-demucs"); fi
  if [[ "$VAD" == true ]]; then cmd+=("--vad"); else cmd+=("--no-vad"); fi
  if [[ -n "$VAD_THRESHOLD" ]]; then cmd+=("--vad-threshold" "$VAD_THRESHOLD"); fi

  # UX toggles
  if [[ "$BENCHMARK" == true ]]; then cmd+=("--benchmark"); fi
  if [[ "$PROGRESS" == true ]]; then cmd+=("--progress"); else cmd+=("--no-progress"); fi
  if [[ "$QUIET" == true ]]; then cmd+=("--quiet"); fi

  # output path only valid when single format selected
  if [[ -n "$OUTPUT_PATH" && "$EXPORT_FORMAT" != "all" ]]; then
    cmd+=("--output" "$OUTPUT_PATH")
  fi

  # benchmark extras
  for kv in "${BENCHMARK_EXTRAS[@]}"; do
    cmd+=("--benchmark-extra" "$kv")
  done

  cmd+=("$audio")

  printf '%q ' "${cmd[@]}"
}

main() {
  parse_args "$@"
  ensure_defaults

  echo "== Benchmark configuration =="
  echo "Model:        $MODEL"
  echo "Device:       ${DEVICE:-<default>}"
  echo "DType:        ${DTYPE:-<default>}"
  echo "Language:     ${LANGUAGE:-auto}"
  echo "Export:       $EXPORT_FORMAT"
  echo "Timestamp(s): ${TIMESTAMP_TYPES[*]}"
  echo "Batch(chunk): $BATCH_CHUNK"
  echo "Batch(word):  $BATCH_WORD"
  echo "Stabilize:    $STABILIZE  Demucs: $DEMUCS  VAD: $VAD  VAD_TH: ${VAD_THRESHOLD:-<n/a>}"
  echo "Progress:     $PROGRESS   Quiet: $QUIET  Benchmark: $BENCHMARK  Repeats: $REPEATS"
  echo "Audio files:  ${AUDIO_FILES[*]}"
  echo

  for audio in "${AUDIO_FILES[@]}"; do
    for ts_type in "${TIMESTAMP_TYPES[@]}"; do
      # Resolve batch by timestamp type
      if [[ "$ts_type" == "chunk" ]]; then
        batch="$BATCH_CHUNK"
      else
        batch="$BATCH_WORD"
      fi

      echo "-- Running: $audio | ts=$ts_type | batch=$batch"
      for ((i=1; i<=REPEATS; i++)); do
        echo "   Attempt $i/$REPEATS"
        cmd_str=$(build_cmd "$audio" "$ts_type" "$batch")
        echo "   Command: $cmd_str"
        if [[ "$DRY_RUN" == false ]]; then
          # shellcheck disable=SC2086
          eval $cmd_str
        fi
      done
    done
  done

  echo "Benchmarks run successfully"
}

main "$@"
