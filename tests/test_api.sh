#!/bin/bash

# Set the API base URL
API_URL="http://localhost:8888"

echo "Testing Transcription API Endpoints"
echo "=================================="

echo -e "\n1. Quick transcription with default parameters (short file):"
curl -X POST "${API_URL}/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/conversion-test-file.mp3" \
  -F "model=openai/whisper-tiny"

echo -e "\n2. Transcription with custom parameters (long file):"
curl -X POST "${API_URL}/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/test-long.mp3" \
  -F "model=openai/whisper-tiny" \
  -F "response_format=text" \
  -F "timestamp_type=word" \
  -F "language=en"

echo -e "\n3. Advanced transcription with diarization (long file):"
curl -X POST "${API_URL}/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/test-long.mp3" \
  -F "model=openai/whisper-tiny" \
  -F "timestamp_type=chunk" \

echo -e "\nTesting Translation API Endpoints"
echo "================================"

echo -e "\n4. Quick translation with default parameters (short file):"
curl -X POST "${API_URL}/v1/audio/translations" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/conversion-test-file.mp3"

echo -e "\n5. Advanced translation with custom parameters (long file):"
curl -X POST "${API_URL}/v1/audio/translations" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/test-long.mp3" \
  -F "model=openai/whisper-large-v3" \
  -F "response_format=text"

# Error cases (using short file for quicker testing)
echo -e "\nTesting Error Cases"
echo "==================="

echo -e "\n6. Testing with unsupported file format:"
curl -X POST "${API_URL}/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/test.txt"

echo -e "\n7. Testing with invalid model name (short file):"
curl -X POST "${API_URL}/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/conversion-test-file.mp3" \
  -F "model=invalid-model"

# Performance testing with different batch sizes (long file)
echo -e "\n8. Performance test with large batch size:"
curl -v -X POST "${API_URL}/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/test-long.mp3" \
  -F "model=openai/whisper-large-v3" \
  -F "batch_size=6" 