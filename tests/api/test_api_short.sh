#!/bin/bash

# Set the API base URL
API_URL="http://localhost:8888"

# Set the audio file to use
AUDIO_FILE="tests/data/conversion-test-file.mp3"

# Set the model to use
MODEL="distil-whisper/distil-large-v3"

# Set the language to use
LANGUAGE="nl"

# Set the timestamp type to use
TIMESTAMP_TYPE="chunk"

# Set the response format to use
RESPONSE_FORMAT="json"

echo "Testing Transcription API Endpoints"
echo "=================================="

echo -e "\n1. Quick transcription with default parameters (short file):"
echo "Request:"
echo "  POST ${API_URL}/v1/audio/transcriptions"
echo "  Headers:"
echo "    accept: application/json"
echo "    Content-Type: multipart/form-data"
echo "  Form Data:"
echo "    file=@${AUDIO_FILE}"
echo "    model=${MODEL}"
echo "    response_format=${RESPONSE_FORMAT}"
echo "    timestamp_type=${TIMESTAMP_TYPE}"
echo "    language=${LANGUAGE}"
echo ""

curl -v -X POST "${API_URL}/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@${AUDIO_FILE}" \
  -F "model=${MODEL}" \
  -F "response_format=${RESPONSE_FORMAT}" \
  -F "timestamp_type=${TIMESTAMP_TYPE}" \
  -F "language=${LANGUAGE}"

EXIT_CODE=$?
echo ""
echo "Curl exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
  echo "Error: curl command failed"
fi