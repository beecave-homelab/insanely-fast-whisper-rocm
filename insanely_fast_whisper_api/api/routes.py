"""API route definitions for the Insanely Fast Whisper API.

This module contains clean, focused route definitions that use dependency
injection for ASR pipeline instances and file handling.
"""

import logging
from typing import Literal, Optional, Union

from fastapi import APIRouter, Depends, File, Form, UploadFile

from insanely_fast_whisper_api.api.dependencies import (
    get_asr_pipeline,
    get_file_handler,
)
from insanely_fast_whisper_api.api.models import TranscriptionResponse
from insanely_fast_whisper_api.api.responses import ResponseFormatter
from insanely_fast_whisper_api.core.pipeline import WhisperPipeline
from insanely_fast_whisper_api.utils import (
    DEFAULT_TIMESTAMP_TYPE,
    RESPONSE_FORMAT_JSON,
    RESPONSE_FORMAT_TEXT,
    FileHandler,
)

logger = logging.getLogger(__name__)

router = APIRouter()
file_handler = FileHandler()


@router.post(
    "/v1/audio/transcriptions",
    response_model=TranscriptionResponse,
    tags=["Transcription"],
    summary="Transcribe Audio",
    description="Convert speech in an audio file to text using the Whisper model",
    responses={
        200: {
            "description": "Successful transcription",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/TranscriptionResponse"}
                },
                "text/plain": {"schema": {"type": "string"}},
            },
        },
        400: {"description": "Invalid request parameters"},
        422: {"description": "Validation error (e.g., unsupported file format)"},
        500: {"description": "Internal server error"},
        503: {"description": "Model not loaded or unavailable"},
    },
)
async def create_transcription(
    file: UploadFile = File(..., description="The audio file to transcribe"),
    timestamp_type: str = Form(
        DEFAULT_TIMESTAMP_TYPE,
        description="Type of timestamp to generate ('chunk' or 'word')",
    ),
    language: Optional[str] = Form(
        None, description="Source language code (auto-detect if None)"
    ),
    task: Literal["transcribe"] = Form("transcribe", description="ASR task type"),
    asr_pipeline: WhisperPipeline = Depends(get_asr_pipeline),
    file_handler: FileHandler = Depends(get_file_handler),
) -> Union[str, dict]:
    """Transcribe speech in an audio file to text.

    This endpoint processes an audio file and returns its transcription using the
    specified Whisper model. It supports various configuration options including
    timestamp generation.

    Args:
        file: The audio file to transcribe (supported formats: mp3, wav, etc.)
        timestamp_type: Type of timestamp to generate ("chunk" or "word")
        language: Optional source language code (auto-detect if None)
        task: ASR task type (must be "transcribe")
        asr_pipeline: Injected ASR pipeline instance
        file_handler: Injected file handler instance

    Returns:
        Union[str, dict]: Transcription result as plain text or JSON with metadata

    Raises:
        HTTPException: If file validation fails or processing errors occur
    """
    logger.info("-" * 50)
    logger.info("Received transcription request:")
    logger.info("  File: %s", file.filename)
    logger.debug("  Timestamp type: %s", timestamp_type)
    logger.debug("  Language: %s", language)
    logger.debug("  Task: %s", task)

    # Validate and save file
    file_handler.validate_audio_file(file)
    temp_filepath = file_handler.save_upload(file)

    try:
        logger.info("Starting transcription process...")
        result = asr_pipeline.process(
            audio_file_path=temp_filepath,
            language=language,
            task=task,
            timestamp_type=timestamp_type,
            original_filename=file.filename,
        )
        logger.info("Transcription completed successfully")
        logger.debug("Transcription result: %s", result)

        # Format response - for transcription, use timestamp_type to determine format
        response_format = (
            RESPONSE_FORMAT_TEXT
            if timestamp_type == RESPONSE_FORMAT_TEXT
            else RESPONSE_FORMAT_JSON
        )
        return ResponseFormatter.format_transcription(result, response_format)

    finally:
        file_handler.cleanup(temp_filepath)


@router.post(
    "/v1/audio/translations",
    response_model=TranscriptionResponse,
    tags=["Translation"],
    summary="Translate Audio",
    description="Translate speech in an audio file to English using the Whisper model",
    responses={
        200: {
            "description": "Successful translation",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/TranscriptionResponse"}
                },
                "text/plain": {"schema": {"type": "string"}},
            },
        },
        400: {"description": "Invalid request parameters"},
        422: {"description": "Validation error (e.g., unsupported file format)"},
        500: {"description": "Internal server error"},
        503: {"description": "Model not loaded or unavailable"},
    },
)
async def create_translation(
    file: UploadFile = File(..., description="The audio file to translate"),
    response_format: str = Form(
        RESPONSE_FORMAT_JSON, description="Response format (json or text)"
    ),
    timestamp_type: str = Form(
        DEFAULT_TIMESTAMP_TYPE,
        description="Type of timestamp to generate ('chunk' or 'word')",
    ),
    language: Optional[str] = Form(
        None, description="Source language code (auto-detect if None)"
    ),
    asr_pipeline: WhisperPipeline = Depends(get_asr_pipeline),
    file_handler: FileHandler = Depends(get_file_handler),
) -> Union[str, dict]:
    """Translate speech in an audio file to English.

    This endpoint processes an audio file in any supported language and translates
    the speech to English using the specified Whisper model. It supports various
    configuration options.

    Args:
        file: The audio file to translate (supported formats: mp3, wav, etc.)
        response_format: Desired response format ("json" or "text")
        timestamp_type: Type of timestamp to generate ("chunk" or "word")
        language: Optional source language code (auto-detect if None)
        asr_pipeline: Injected ASR pipeline instance
        file_handler: Injected file handler instance

    Returns:
        Union[str, dict]: Translation result as plain text or JSON with metadata

    Raises:
        HTTPException: If file validation fails or processing errors occur
    """
    logger.info("-" * 50)
    logger.info("Received translation request:")
    logger.info("  File: %s", file.filename)
    logger.debug("  Timestamp type: %s", timestamp_type)
    logger.debug("  Language: %s", language)
    logger.debug("  Response format: %s", response_format)

    # Validate and save file
    file_handler.validate_audio_file(file)
    temp_filepath = file_handler.save_upload(file)

    try:
        logger.info("Starting translation process...")
        result = asr_pipeline.process(
            audio_file_path=temp_filepath,
            language=language,
            task="translate",
            timestamp_type=timestamp_type,
            original_filename=file.filename,
        )
        logger.info("Translation completed successfully")
        logger.debug("Translation result: %s", result)

        # Format response
        return ResponseFormatter.format_translation(result, response_format)

    finally:
        file_handler.cleanup(temp_filepath)
