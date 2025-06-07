"""Dependency injection providers for FastAPI routes.

This module implements dependency injection for ASR pipeline instances
and other shared resources used by the API endpoints.
"""

from fastapi import Form

from insanely_fast_whisper_api.core.pipeline import WhisperPipeline
from insanely_fast_whisper_api.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)
from insanely_fast_whisper_api.utils import (
    FileHandler,
    DEFAULT_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BETTER_TRANSFORMER,
    DEFAULT_CHUNK_LENGTH,
)


def get_asr_pipeline(
    model: str = Form(DEFAULT_MODEL, description="The Whisper model to use"),
    device: str = Form(DEFAULT_DEVICE, description="Device ID for processing"),
    batch_size: int = Form(
        DEFAULT_BATCH_SIZE, description="Number of parallel audio segments to process"
    ),
    dtype: str = Form(
        "float16", description="Data type for model inference ('float16' or 'float32')"
    ),
    better_transformer: bool = Form(
        DEFAULT_BETTER_TRANSFORMER,
        description="Whether to apply BetterTransformer optimization",
    ),
    model_chunk_length: int = Form(
        DEFAULT_CHUNK_LENGTH,
        description="Internal chunk length for the Whisper model (seconds)",
    ),
) -> WhisperPipeline:
    """Dependency to provide configured ASR pipeline.

    This function implements dependency injection for ASR pipeline instances,
    creating a properly configured pipeline based on request parameters.

    Args:
        model: Name of the Whisper model to use
        device: Device ID for processing (e.g., "0" for first GPU)
        batch_size: Number of parallel audio segments to process
        dtype: Data type for model inference ('float16' or 'float32')
        better_transformer: Whether to apply BetterTransformer optimization
        model_chunk_length: Internal chunk length for the Whisper model (seconds)

    Returns:
        WhisperPipeline: Configured ASR pipeline instance
    """
    backend_config = HuggingFaceBackendConfig(
        model_name=model,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        better_transformer=better_transformer,
        chunk_length=model_chunk_length,
    )
    backend = HuggingFaceBackend(config=backend_config)
    return WhisperPipeline(asr_backend=backend)


def get_file_handler() -> FileHandler:
    """Dependency to provide file handler instance.

    Returns:
        FileHandler: File handler instance for managing uploads and cleanup
    """
    return FileHandler()
