#!/bin/bash

FILE_INPUT=uploads/MoA_Groq-The_Ultimate_LLM_Architecture.mp3
TRANSCRIPT_OUTPUT=transcripts/MoA_Groq-The_Ultimate_LLM_Architecture.txt
BATCH_SIZE=4

insanely-fast-whisper --file-name ${FILE_INPUT} --transcript-path ${TRANSCRIPT_OUTPUT} --batch-size ${BATCH_SIZE}
