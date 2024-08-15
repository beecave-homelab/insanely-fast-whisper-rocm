#!/bin/bash

FILE_INPUT=uploads/preamble.wav
TRANSCRIPT_OUTPUT=transcripts/output.txt
BATCH_SIZE=10

insanely-fast-whisper --file-name ${FILE_INPUT} --transcript-path ${TRANSCRIPT_OUTPUT} --batch-size ${BATCH_SIZE}