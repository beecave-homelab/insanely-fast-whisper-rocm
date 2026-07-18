#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# entrypoint-test.sh — Pre-PR integration test for CLI, API, and WebUI
#
# Runs transcription with all feature flags (--vad --demucs --stabilize --diarize)
# using both chunk and word timestamp types, then drops to an interactive bash
# shell for manual inspection.
#
# Usage:
#   docker compose -f docker-compose.test.yaml up
#   docker compose -f docker-compose.test.yaml run --rm insanely-fast-whisper-rocm-test
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AUDIO_FILE="temp_uploads/1-minute-test-audio.wav"
# AUDIO_FILE="temp_uploads/Jana.m4a"
MODEL="openai/whisper-large-v3-turbo"
API_PORT="${API_PORT:-8891}"
WEBUI_PORT="${WEBUI_PORT:-7864}"
API_BASE="http://localhost:${API_PORT}"
WEBUI_BASE="http://localhost:${WEBUI_PORT}"
CLI_CHUNK_RESULT="/tmp/test-results/cli-chunk-result.json"
CLI_WORD_RESULT="/tmp/test-results/cli-word-result.json"

# Test result tracking
declare -A RESULTS

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log_header() {
    echo -e "\n${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

record_result() {
    local name="$1"
    local exit_code="$2"
    if [ "$exit_code" -eq 0 ]; then
        RESULTS["$name"]="PASS"
        log_pass "$name"
    else
        RESULTS["$name"]="FAIL"
        log_fail "$name"
    fi
}

wait_for_url() {
    local url="$1"
    local max_wait="${2:-30}"
    local elapsed=0
    log_info "Waiting for $url to be reachable (max ${max_wait}s)..."
    while [ "$elapsed" -lt "$max_wait" ]; do
        if curl -sf -o /dev/null "$url" 2>/dev/null; then
            log_info "$url is reachable after ${elapsed}s"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log_fail "$url not reachable after ${max_wait}s"
    return 1
}

validate_json_text() {
    local json_file="$1"
    local label="$2"
    # Check the file exists and contains a non-empty "text" field
    if [ ! -f "$json_file" ]; then
        log_fail "$label: output file $json_file not found"
        return 1
    fi
    # Use Python to check for non-empty text (jq may not be installed)
    local text_len
    text_len=$(python3 -c "
import json, sys
try:
    data = json.load(open(sys.argv[1]))
    text = data.get('text', '') or ''
    print(len(text.strip()))
except Exception:
    print(0)
" "$json_file")
    if [ "$text_len" -gt 0 ]; then
        log_info "$label: transcription text length = ${text_len} chars"
        return 0
    else
        log_fail "$label: empty or missing 'text' field in $json_file"
        return 1
    fi
}

validate_json_diarized() {
    local json_file="$1"
    local label="$2"
    if [ ! -f "$json_file" ]; then
        log_fail "$label: output file $json_file not found"
        return 1
    fi

    python3 -c "
import json
import sys

path, label = sys.argv[1], sys.argv[2]
with open(path, encoding='utf-8') as handle:
    payload = json.load(handle)

data = payload.get('data') if isinstance(payload, dict) else None
if not isinstance(data, dict):
    data = payload

text = (data.get('text') or '').strip()
segments = data.get('segments') or data.get('chunks') or []
speaker_count = sum(
    1
    for segment in segments
    if isinstance(segment, dict) and segment.get('speaker')
)

if not text:
    print(f'{label}: empty transcription text', file=sys.stderr)
    sys.exit(1)
if data.get('diarized') is not True:
    print(f'{label}: missing diarized=true', file=sys.stderr)
    sys.exit(1)
if speaker_count <= 0:
    print(f'{label}: no speaker labels found', file=sys.stderr)
    sys.exit(1)

print(
    f'{label}: text_len={len(text)}, speaker_labels={speaker_count}, '
    f'segments={len(segments)}'
)
" "$json_file" "$label"
}

# ---------------------------------------------------------------------------
# Phase 0 — Prerequisites
# ---------------------------------------------------------------------------
log_header "Phase 0: Prerequisites"

if [ ! -f "$AUDIO_FILE" ]; then
    log_fail "Test audio file not found: $AUDIO_FILE"
    log_warn "Mount temp_uploads/ volume with 1-minute-test-audio.wav"
    log_warn "Dropping to bash for manual inspection..."
    exec /bin/bash
fi
log_info "Test audio file found: $AUDIO_FILE ($(stat -c%s "$AUDIO_FILE") bytes)"

# Ensure output directories exist
mkdir -p /tmp/test-results transcripts transcripts-txt transcripts-srt

# ---------------------------------------------------------------------------
# Phase 1 — CLI Tests
# ---------------------------------------------------------------------------
log_header "Phase 1: CLI Transcription Tests"

# --- Chunk timestamps ---
log_info "Running CLI transcription with chunk timestamps..."
set +e
insanely-fast-whisper-cli transcribe \
    "$AUDIO_FILE" \
    --model "$MODEL" \
    --vad --demucs --stabilize --diarize \
    --timestamp-type chunk \
    --export-format json \
    --output "$CLI_CHUNK_RESULT" \
    --debug
CLI_CHUNK_EXIT=$?
set -e
if [ "$CLI_CHUNK_EXIT" -eq 0 ]; then
    validate_json_diarized "$CLI_CHUNK_RESULT" "CLI chunk"
    record_result "CLI chunk" $?
else
    record_result "CLI chunk" "$CLI_CHUNK_EXIT"
fi

# --- Word timestamps ---
log_info "Running CLI transcription with word timestamps..."
set +e
insanely-fast-whisper-cli transcribe \
    "$AUDIO_FILE" \
    --model "$MODEL" \
    --vad --demucs --stabilize --diarize \
    --timestamp-type word \
    --export-format json \
    --output "$CLI_WORD_RESULT" \
    --debug
CLI_WORD_EXIT=$?
set -e
if [ "$CLI_WORD_EXIT" -eq 0 ]; then
    validate_json_diarized "$CLI_WORD_RESULT" "CLI word"
    record_result "CLI word" $?
else
    record_result "CLI word" "$CLI_WORD_EXIT"
fi

# ---------------------------------------------------------------------------
# Phase 2 — API Tests
# ---------------------------------------------------------------------------
log_header "Phase 2: API Transcription Tests"

log_info "Starting API server on port ${API_PORT}..."
python -m insanely_fast_whisper_rocm --port "$API_PORT" --debug &
API_PID=$!

# Wait for API to become ready (model download may take time on first run)
if wait_for_url "${API_BASE}/docs" 120; then
    log_info "API server is ready"

    # --- Chunk timestamps ---
    log_info "Running API transcription with chunk timestamps..."
    set +e
    curl -sf -X POST "${API_BASE}/v1/audio/transcriptions" \
        -F "file=@${AUDIO_FILE}" \
        -F "model=${MODEL}" \
        -F "timestamp_type=chunk" \
        -F "stabilize=true" \
        -F "demucs=true" \
        -F "vad=true" \
        -F "diarize=true" \
        -F "diarization_device=cuda" \
        -F "num_speakers=2" \
        -F "response_format=verbose_json" \
        -o /tmp/test-results/api-chunk-result.json
    API_CHUNK_CURL=$?
    set -e

    if [ "$API_CHUNK_CURL" -eq 0 ]; then
        validate_json_diarized /tmp/test-results/api-chunk-result.json "API chunk"
        record_result "API chunk" $?
    else
        record_result "API chunk" "$API_CHUNK_CURL"
    fi

    # --- Word timestamps ---
    log_info "Running API transcription with word timestamps..."
    set +e
    curl -sf -X POST "${API_BASE}/v1/audio/transcriptions" \
        -F "file=@${AUDIO_FILE}" \
        -F "model=${MODEL}" \
        -F "timestamp_type=word" \
        -F "stabilize=true" \
        -F "demucs=true" \
        -F "vad=true" \
        -F "diarize=true" \
        -F "diarization_device=cuda" \
        -F "num_speakers=2" \
        -F "response_format=verbose_json" \
        -o /tmp/test-results/api-word-result.json
    API_WORD_CURL=$?
    set -e

    if [ "$API_WORD_CURL" -eq 0 ]; then
        validate_json_diarized /tmp/test-results/api-word-result.json "API word"
        record_result "API word" $?
    else
        record_result "API word" "$API_WORD_CURL"
    fi
else
    log_fail "API server did not become ready — skipping API tests"
    record_result "API chunk" 1
    record_result "API word" 1
fi

log_info "Stopping API server (PID ${API_PID})..."
kill "$API_PID" 2>/dev/null || true
wait "$API_PID" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Phase 3 — WebUI Tests
# ---------------------------------------------------------------------------
log_header "Phase 3: WebUI Transcription Tests"

log_info "Starting WebUI server on port ${WEBUI_PORT}..."
python -m insanely_fast_whisper_rocm.webui \
    --port "$WEBUI_PORT" \
    --model "$MODEL" \
    --stabilize --demucs --vad --diarize \
    --debug &
WEBUI_PID=$!

# Wait for Gradio to become ready
if wait_for_url "${WEBUI_BASE}/" 120; then
    log_info "WebUI server is ready"

    # Verify the Gradio API info endpoint
    log_info "Checking Gradio API info endpoint..."
    set +e
    GRADIO_INFO=$(curl -sf "${WEBUI_BASE}/gradio_api/info" 2>/dev/null || echo "")
    set -e
    if [ -n "$GRADIO_INFO" ]; then
        log_info "Gradio /info endpoint responded"
    else
        log_warn "Gradio /info endpoint did not respond (continuing anyway)"
    fi

    # --- Chunk timestamps ---
    log_info "Running WebUI transcription with chunk timestamps..."
    set +e
    python scripts/webui_integration_test.py \
        --base-url "$WEBUI_BASE" \
        --audio-file "$AUDIO_FILE" \
        --model "$MODEL" \
        --timestamp-type chunk \
        --output-file /tmp/test-results/webui-chunk-result.json
    WEBUI_CHUNK_EXIT=$?
    set -e
    if [ "$WEBUI_CHUNK_EXIT" -eq 0 ]; then
        validate_json_diarized /tmp/test-results/webui-chunk-result.json "WebUI chunk"
        record_result "WebUI chunk" $?
    else
        record_result "WebUI chunk" "$WEBUI_CHUNK_EXIT"
    fi

    # --- Word timestamps ---
    log_info "Running WebUI transcription with word timestamps..."
    set +e
    python scripts/webui_integration_test.py \
        --base-url "$WEBUI_BASE" \
        --audio-file "$AUDIO_FILE" \
        --model "$MODEL" \
        --timestamp-type word \
        --output-file /tmp/test-results/webui-word-result.json
    WEBUI_WORD_EXIT=$?
    set -e
    if [ "$WEBUI_WORD_EXIT" -eq 0 ]; then
        validate_json_diarized /tmp/test-results/webui-word-result.json "WebUI word"
        record_result "WebUI word" $?
    else
        record_result "WebUI word" "$WEBUI_WORD_EXIT"
    fi
else
    log_fail "WebUI server did not become ready — skipping WebUI tests"
    record_result "WebUI chunk" 1
    record_result "WebUI word" 1
fi

log_info "Stopping WebUI server (PID ${WEBUI_PID})..."
kill "$WEBUI_PID" 2>/dev/null || true
wait "$WEBUI_PID" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Phase 4 — Summary
# ---------------------------------------------------------------------------
log_header "Test Summary"

echo -e "${BOLD}┌──────────────┬──────────┐${NC}"
echo -e "${BOLD}│ Test         │ Result   │${NC}"
echo -e "${BOLD}├──────────────┼──────────┤${NC}"
for name in "CLI chunk" "CLI word" "API chunk" "API word" "WebUI chunk" "WebUI word"; do
    result="${RESULTS[$name]:-SKIP}"
    if [ "$result" = "PASS" ]; then
        color="$GREEN"
    elif [ "$result" = "FAIL" ]; then
        color="$RED"
    else
        color="$YELLOW"
    fi
    printf "${BOLD}│ %-12s │ ${color}%-8s${NC}${BOLD} │${NC}\n" "$name" "$result"
done
echo -e "${BOLD}└──────────────┴──────────┘${NC}"

# Count results
pass_count=0
fail_count=0
for name in "CLI chunk" "CLI word" "API chunk" "API word" "WebUI chunk" "WebUI word"; do
    case "${RESULTS[$name]:-SKIP}" in
        PASS) pass_count=$((pass_count + 1)) ;;
        FAIL) fail_count=$((fail_count + 1)) ;;
    esac
done

echo ""
log_info "Passed: ${pass_count} / $((pass_count + fail_count))"

if [ "$fail_count" -gt 0 ]; then
    log_warn "Some tests failed — check logs above for details"
    log_warn "Test result files are in /tmp/test-results/"
else
    log_pass "All tests passed!"
fi

if [ "${TEST_DROP_TO_SHELL:-1}" = "0" ]; then
    if [ "$fail_count" -gt 0 ]; then
        exit 1
    fi
    exit 0
fi

echo ""
log_info "Dropping to interactive bash for manual inspection..."
log_info "Test result files: /tmp/test-results/"
log_info "Audio file: ${AUDIO_FILE}"
log_info "Re-run a test manually, e.g.:"
log_info "  insanely-fast-whisper-cli transcribe ${AUDIO_FILE} --model ${MODEL} --vad --demucs --stabilize --diarize --timestamp-type chunk --debug"
log_info "  python -m insanely_fast_whisper_rocm --port ${API_PORT} --debug    # start API"
log_info "  python -m insanely_fast_whisper_rocm.webui --port ${WEBUI_PORT} --model ${MODEL} --stabilize --demucs --vad --diarize --debug  # start WebUI"
echo ""

exec /bin/bash
