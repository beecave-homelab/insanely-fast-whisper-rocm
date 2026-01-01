"""API route definitions for the Insanely Fast Whisper API.

This module contains clean, focused route definitions that use dependency
injection for ASR pipeline instances and file handling.
"""

import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from insanely_fast_whisper_rocm.api.dependencies import (
    get_asr_pipeline,
    get_file_handler,
)
from insanely_fast_whisper_rocm.api.responses import ResponseFormatter
from insanely_fast_whisper_rocm.core.errors import OutOfMemoryError
from insanely_fast_whisper_rocm.core.integrations.stable_ts import stabilize_timestamps
from insanely_fast_whisper_rocm.core.orchestrator import create_orchestrator
from insanely_fast_whisper_rocm.core.pipeline import WhisperPipeline
from insanely_fast_whisper_rocm.utils import (
    DEFAULT_DEMUCS,
    DEFAULT_STABILIZE,
    DEFAULT_TIMESTAMP_TYPE,
    DEFAULT_VAD,
    DEFAULT_VAD_THRESHOLD,
    RESPONSE_FORMAT_JSON,
    SUPPORTED_RESPONSE_FORMATS,
    FileHandler,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v1/audio/transcriptions",
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
    file: UploadFile = File(..., description="The audio file to transcribe"),  # noqa: B008
    response_format: str = Form(
        RESPONSE_FORMAT_JSON,
        description="Response format (json, verbose_json, text, srt, vtt)",
    ),
    timestamp_type: str = Form(
        DEFAULT_TIMESTAMP_TYPE,
        description="Type of timestamp to generate ('chunk' or 'word')",
    ),
    language: str | None = Form(
        None, description="Source language code (auto-detect if None)"
    ),
    task: Literal["transcribe"] = Form("transcribe", description="ASR task type"),
    stabilize: bool = Form(
        DEFAULT_STABILIZE, description="Enable timestamp stabilization"
    ),
    demucs: bool = Form(DEFAULT_DEMUCS, description="Enable Demucs noise reduction"),
    vad: bool = Form(DEFAULT_VAD, description="Enable Voice Activity Detection"),
    vad_threshold: float = Form(
        DEFAULT_VAD_THRESHOLD, description="VAD threshold for speech detection"
    ),
    asr_pipeline: WhisperPipeline = Depends(get_asr_pipeline),  # noqa: B008
    file_handler: FileHandler = Depends(get_file_handler),  # noqa: B008
) -> str | dict:
    """
    Transcribe an uploaded audio file using the configured Whisper ASR pipeline.
    
    Processes the uploaded file, runs transcription (orchestrator-managed), optionally post-processes timestamps, and returns the result in the requested format.
    
    Parameters:
        response_format (str): Output format: "json", "verbose_json", "text", "srt", or "vtt".
        timestamp_type (str): Timestamp generation mode: "chunk" or "word".
        language (str | None): Source language code; auto-detect if None.
        task (Literal["transcribe"]): ASR task; must be "transcribe".
        stabilize (bool): If True, apply timestamp stabilization to the transcription.
        demucs (bool): If True, enable Demucs noise reduction during stabilization.
        vad (bool): If True, enable Voice Activity Detection during stabilization.
        vad_threshold (float): VAD sensitivity threshold (0.0 - 1.0).
    
    Returns:
        str | dict: Transcription result as plain text or a structured dictionary depending on `response_format`.
    
    Raises:
        HTTPException: If file validation fails, an unsupported response_format is requested,
            or processing errors occur (including OutOfMemoryError mapped to HTTP 507).
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

        # Use orchestrator for transcription with OOM recovery
        orchestrator = create_orchestrator()

        # We need to construct a backend config.
        # Since we use dependency injection for asr_pipeline,
        # we can get the config from it.
        # However, the orchestrator handles pipeline acquisition
        # via borrow_pipeline.
        # We'll use the config from the injected pipeline as
        # the starting point.
        base_config = asr_pipeline.asr_backend.config

        try:
            result = orchestrator.run_transcription(
                audio_path=temp_filepath,
                backend_config=base_config,
                language=language,
                task=task,
                timestamp_type=timestamp_type,
            )
        except OutOfMemoryError as oom:
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient GPU memory for transcription: {str(oom)}",
            ) from oom
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=str(e)) from e

        # Optional stabilization (post-process) applied here for API
        if stabilize:
            try:
                result = stabilize_timestamps(
                    result, demucs=demucs, vad=vad, vad_threshold=vad_threshold
                )
            except Exception as stab_exc:  # noqa: BLE001
                logger.error("Stabilization failed: %s", stab_exc, exc_info=True)
        logger.info("Transcription completed successfully")

        # Validate response_format
        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise HTTPException(status_code=400, detail="Unsupported response_format")
        logger.debug("Transcription result: %s", result)

        # Format response according to requested response_format
        return ResponseFormatter.format_transcription(result, response_format)

    finally:
        file_handler.cleanup(temp_filepath)


@router.post(
    "/v1/audio/translations",
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
    file: UploadFile = File(..., description="The audio file to translate"),  # noqa: B008
    response_format: str = Form(
        RESPONSE_FORMAT_JSON,
        description="Response format (json, verbose_json, text, srt, vtt)",
    ),
    timestamp_type: str = Form(
        DEFAULT_TIMESTAMP_TYPE,
        description="Type of timestamp to generate ('chunk' or 'word')",
    ),
    language: str | None = Form(
        None, description="Source language code (auto-detect if None)"
    ),
    stabilize: bool = Form(
        DEFAULT_STABILIZE, description="Enable timestamp stabilization"
    ),
    demucs: bool = Form(DEFAULT_DEMUCS, description="Enable Demucs noise reduction"),
    vad: bool = Form(DEFAULT_VAD, description="Enable Voice Activity Detection"),
    vad_threshold: float = Form(
        DEFAULT_VAD_THRESHOLD, description="VAD threshold for speech detection"
    ),
    asr_pipeline: WhisperPipeline = Depends(get_asr_pipeline),  # noqa: B008
    file_handler: FileHandler = Depends(get_file_handler),  # noqa: B008
) -> str | dict:
    """
    Translate speech in an uploaded audio file to English.
    
    Processes the uploaded audio, performs translation via the ASR orchestrator, optionally stabilizes timestamps, and returns the result in the requested format.
    
    Parameters:
        file (UploadFile): The audio file to translate.
        response_format (str): Response format to return; one of "json", "verbose_json", "text", "srt", or "vtt".
        timestamp_type (str): Type of timestamp to generate; either "chunk" or "word".
        language (str | None): Source language code; if None, the language will be auto-detected.
        stabilize (bool): If True, apply timestamp stabilization to the transcription.
        demucs (bool): If True, apply Demucs noise reduction during stabilization.
        vad (bool): If True, apply Voice Activity Detection during stabilization.
        vad_threshold (float): VAD sensitivity threshold between 0.0 and 1.0.
    
    Returns:
        str or dict: The translated output formatted according to `response_format` (plain text or a structured JSON-like object).
    
    Raises:
        HTTPException: If file validation fails, `response_format` is unsupported, or processing errors occur (including insufficient GPU memory).
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

        # Use orchestrator for translation with OOM recovery
        orchestrator = create_orchestrator()
        base_config = asr_pipeline.asr_backend.config

        try:
            result = orchestrator.run_transcription(
                audio_path=temp_filepath,
                backend_config=base_config,
                language=language,
                task="translate",
                timestamp_type=timestamp_type,
            )
        except OutOfMemoryError as oom:
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient GPU memory for translation: {str(oom)}",
            ) from oom
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=str(e)) from e

        # Optional stabilization (post-process) applied here for API
        if stabilize:
            try:
                result = stabilize_timestamps(
                    result, demucs=demucs, vad=vad, vad_threshold=vad_threshold
                )
            except Exception as stab_exc:  # noqa: BLE001
                logger.error("Stabilization failed: %s", stab_exc, exc_info=True)
        logger.info("Translation completed successfully")
        logger.debug("Translation result: %s", result)

        # Validate response_format
        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise HTTPException(status_code=400, detail="Unsupported response_format")

        # Format response
        return ResponseFormatter.format_translation(result, response_format)

    finally:
        file_handler.cleanup(temp_filepath)