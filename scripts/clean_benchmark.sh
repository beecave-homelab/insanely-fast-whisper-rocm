#!/usr/bin/env bash
# clean_benchmark.sh
# Remove .json files from benchmarks/ and transcripts/
# Remove .srt files from transcripts-srt/
# Remove .txt files from transcripts-txt/

set -euo pipefail

echo "Cleaning benchmark and transcript files..."

echo "Removing .json files from benchmarks/ and transcripts/..."
find benchmarks/ transcripts/ -name "*.json" -type f -delete
echo "Removed .json files."

echo "Removing .srt files from transcripts-srt/..."
find transcripts-srt/ -name "*.srt" -type f -delete
echo "Removed .srt files."

echo "Removing .txt files from transcripts-txt/..."
find transcripts-txt/ -name "*.txt" -type f -delete
echo "Removed .txt files."

echo "Done."
