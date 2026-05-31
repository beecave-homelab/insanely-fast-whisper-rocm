"""API route definitions for the Insanely Fast Whisper API.

This module contains clean, focused route definitions that use dependency
injection for ASR pipeline instances and file handling.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from insanely_fast_whisper_rocm.api.dependencies import (
    get_asr_pipeline,
    get_file_handler,
)
from insanely_fast_whisper_rocm.api.responses import ResponseFormatter
from insanely_fast_whisper_rocm.core.errors import (
    DiarizationError,
    OutOfMemoryError,
)
from insanely_fast_whisper_rocm.core.integrations.stable_ts import (
    stabilize_timestamps,
)
from insanely_fast_whisper_rocm.core.orchestrator import create_orchestrator
from insanely_fast_whisper_rocm.core.pipeline import WhisperPipeline
from insanely_fast_whisper_rocm.utils import (
    DEFAULT_DEMUCS,
    DEFAULT_DIARIZATION_DEVICE,
    DEFAULT_DIARIZE,
    DEFAULT_STABILIZE,
    DEFAULT_TIMESTAMP_TYPE,
    DEFAULT_VAD,
    DEFAULT_VAD_THRESHOLD,
    HF_TOKEN,
    RESPONSE_FORMAT_JSON,
    SUPPORTED_RESPONSE_FORMATS,
    FileHandler,
)

logger = logging.getLogger(__name__)

router = APIRouter()
VALID_DIARIZATION_DEVICES = {"cpu", "cuda", "gpu"}


def _json_default(value: object) -> object:
    """Convert non-standard JSON scalar values for persisted API results.

    Args:
        value: Value passed by ``json.dumps`` when default encoding fails.

    Returns:
        JSON-compatible scalar or container value.

    Raises:
        TypeError: If the value cannot be converted.
    """
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _persist_post_processed_result(result: dict[str, Any]) -> None:
    """Persist the final API result when the pipeline already created a JSON file.

    The pipeline saves before API-level stabilization and diarization run. Rewrite
    that file after post-processing so disk artifacts match the response.

    Args:
        result: Final post-processed transcription or translation result.
    """
    output_file_path = result.get("output_file_path")
    if not isinstance(output_file_path, str) or not output_file_path:
        return

    output_path = Path(output_file_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        logger.info("Post-processed API result saved to %s", output_path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(
            "Failed to update post-processed API result at %s: %s",
            output_path,
            exc,
            exc_info=True,
        )


def _apply_post_processing(
    result: dict[str, Any],
    *,
    stabilize: bool,
    demucs: bool,
    vad: bool,
    vad_threshold: float,
    diarize: bool,
    diarization_device: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    audio_path: str,
) -> dict[str, Any]:
    """Apply optional stabilization and diarization post-processing.

    Args:
        result: Raw Whisper transcription result.
        stabilize: Whether to run timestamp stabilization.
        demucs: Enable Demucs noise reduction in stabilization.
        vad: Enable VAD in stabilization.
        vad_threshold: VAD sensitivity threshold.
        diarize: Whether to run speaker diarization.
        diarization_device: Device for diarization pipeline.
        num_speakers: Exact speaker count (None = auto).
        min_speakers: Minimum speaker count.
        max_speakers: Maximum speaker count.
        audio_path: Path to the audio file on disk.

    Returns:
        The result dict, potentially enriched with stabilization and/or
        speaker labels.  When ``stabilize=True``, a ``"stabilized"`` key
        is added (``True`` on success, ``False`` if stabilization failed).

    Raises:
        HTTPException: If diarization_device is invalid or diarization fails.
    """
    if stabilize:
        try:
            result = stabilize_timestamps(
                result, demucs=demucs, vad=vad, vad_threshold=vad_threshold
            )
            result["stabilized"] = True
        except Exception as stab_exc:  # noqa: BLE001
            logger.error("Stabilization failed: %s", stab_exc, exc_info=True)
            result["stabilized"] = False

    if diarize:
        if diarization_device not in VALID_DIARIZATION_DEVICES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid diarization_device "
                    f"'{diarization_device}'. Must be one of: "
                    f"{sorted(VALID_DIARIZATION_DEVICES)}"
                ),
            )

        normalized_device = (
            "cuda" if diarization_device == "gpu" else diarization_device
        )

        try:
            from insanely_fast_whisper_rocm.core.integrations.diarization import (
                diarize as diarize_result,
            )

            result = diarize_result(
                result,
                audio_path=audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                device=normalized_device,
                hf_token=HF_TOKEN,
            )
        except DiarizationError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    return result


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
    diarize: bool = Form(DEFAULT_DIARIZE, description="Enable speaker diarization"),
    num_speakers: int | None = Form(None, description="Exact number of speakers"),
    min_speakers: int | None = Form(None, description="Minimum number of speakers"),
    max_speakers: int | None = Form(None, description="Maximum number of speakers"),
    diarization_device: str = Form(
        DEFAULT_DIARIZATION_DEVICE,
        description="Device for diarization (cpu, cuda, or gpu)",
    ),
    asr_pipeline: WhisperPipeline = Depends(get_asr_pipeline),  # noqa: B008
    file_handler: FileHandler = Depends(get_file_handler),  # noqa: B008
) -> str | dict:
    """Transcribe speech in an audio file to text.

    This endpoint processes an audio file and returns its transcription using the
    specified Whisper model. It supports various configuration options including
    timestamp generation.

    Args:
        file: The audio file to transcribe (supported formats: mp3, wav, etc.)
        response_format: Desired response format ("json", "verbose_json",
            "text", "srt", or "vtt").
        timestamp_type: Type of timestamp to generate ("chunk" or "word")
        language: Optional source language code (auto-detect if None)
        task: ASR task type (must be "transcribe")
        stabilize: Enable timestamp stabilization if True.
        demucs: Enable Demucs noise reduction if True.
        vad: Enable Voice Activity Detection if True.
        vad_threshold: VAD sensitivity threshold (0.0 - 1.0).
        diarize: Enable speaker diarization if True.
        num_speakers: Exact number of speakers (auto-detect if None).
        min_speakers: Minimum number of speakers for diarization.
        max_speakers: Maximum number of speakers for diarization.
        diarization_device: Device for diarization ("cpu", "cuda", or
            "gpu"; "gpu" is an alias for "cuda").
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

        # Optional post-processing (stabilization + diarization)
        result = _apply_post_processing(
            result,
            stabilize=stabilize,
            demucs=demucs,
            vad=vad,
            vad_threshold=vad_threshold,
            diarize=diarize,
            diarization_device=diarization_device,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            audio_path=temp_filepath,
        )
        _persist_post_processed_result(result)

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
    diarize: bool = Form(DEFAULT_DIARIZE, description="Enable speaker diarization"),
    num_speakers: int | None = Form(None, description="Exact number of speakers"),
    min_speakers: int | None = Form(None, description="Minimum number of speakers"),
    max_speakers: int | None = Form(None, description="Maximum number of speakers"),
    diarization_device: str = Form(
        DEFAULT_DIARIZATION_DEVICE,
        description="Device for diarization (cpu, cuda, or gpu)",
    ),
    asr_pipeline: WhisperPipeline = Depends(get_asr_pipeline),  # noqa: B008
    file_handler: FileHandler = Depends(get_file_handler),  # noqa: B008
) -> str | dict:
    """Translate speech in an audio file to English.

    This endpoint processes an audio file in any supported language and translates
    the speech to English using the specified Whisper model. It supports various
    configuration options.

    Args:
        file: The audio file to translate (supported formats: mp3, wav, etc.)
        response_format: Desired response format ("json" or "text")
        timestamp_type: Type of timestamp to generate ("chunk" or "word")
        language: Optional source language code (auto-detect if None)
        stabilize: Enable timestamp stabilization if True.
        demucs: Enable Demucs noise reduction if True.
        vad: Enable Voice Activity Detection if True.
        vad_threshold: VAD sensitivity threshold (0.0 - 1.0).
        diarize: Enable speaker diarization if True.
        num_speakers: Exact number of speakers (auto-detect if None).
        min_speakers: Minimum number of speakers for diarization.
        max_speakers: Maximum number of speakers for diarization.
        diarization_device: Device for diarization ("cpu", "cuda", or
            "gpu"; "gpu" is an alias for "cuda").
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

        # Optional post-processing (stabilization + diarization)
        result = _apply_post_processing(
            result,
            stabilize=stabilize,
            demucs=demucs,
            vad=vad,
            vad_threshold=vad_threshold,
            diarize=diarize,
            diarization_device=diarization_device,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            audio_path=temp_filepath,
        )
        _persist_post_processed_result(result)

        logger.info("Translation completed successfully")
        logger.debug("Translation result: %s", result)

        # Validate response_format
        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise HTTPException(status_code=400, detail="Unsupported response_format")

        # Format response
        return ResponseFormatter.format_translation(result, response_format)

    finally:
        file_handler.cleanup(temp_filepath)
