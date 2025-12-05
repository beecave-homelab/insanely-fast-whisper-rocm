"""Dependency injection providers for FastAPI routes.

This module implements dependency injection for ASR pipeline instances
and other shared resources used by the API endpoints.
"""

from __future__ import annotations

from typing import NoReturn

from insanely_fast_whisper_api.core.asr_backend import HuggingFaceBackendConfig
from insanely_fast_whisper_api.core.backend_cache import borrow_pipeline
from insanely_fast_whisper_api.core.pipeline import WhisperPipeline
from insanely_fast_whisper_api.utils import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_LENGTH,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    FileHandler,
    constants,
)


def get_asr_pipeline(
    model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dtype: str = "float16",
    model_chunk_length: int = DEFAULT_CHUNK_LENGTH,
) -> WhisperPipeline:  # type: ignore[override]
    """Dependency to provide configured ASR pipeline.

    This generator function implements dependency injection for ASR pipeline
    instances, creating a properly configured pipeline based on request
    parameters and yielding it to the route. After the request finishes,
    it releases the cached pipeline reference.

    Args:
        model: Name of the Whisper model to use
        device: Device ID for processing (e.g., "0" for first GPU)
        batch_size: Number of parallel audio segments to process
        dtype: Data type for model inference ('float16' or 'float32')
        model_chunk_length: Internal chunk length for the Whisper model (seconds)

    Yields:
        WhisperPipeline: Configured ASR pipeline instance for the request
    """

    # FastAPI's dependency-injection may sometimes pass param Placeholders (e.g. Form)
    # if this function is used incorrectly as a dependency with `Form` params.  Make
    # the function robust by extracting the `.default` attribute when a parameter is
    # a FastAPI param instance.
    def _normalize(value: object, default: object | None = None) -> object:
        # Detect fastapi.params.Param types without importing fastapi here.
        if hasattr(value, "__class__") and value.__class__.__module__.startswith(
            "fastapi."
        ):
            return getattr(value, "default", default)
        return value

    backend_config = HuggingFaceBackendConfig(
        model_name=_normalize(model, DEFAULT_MODEL),
        device=_normalize(device, DEFAULT_DEVICE),
        dtype=_normalize(dtype, "float16"),
        batch_size=int(_normalize(batch_size, DEFAULT_BATCH_SIZE)),
        chunk_length=int(_normalize(model_chunk_length, DEFAULT_CHUNK_LENGTH)),
        progress_group_size=constants.DEFAULT_PROGRESS_GROUP_SIZE,
    )
    # Acquire cached pipeline and ensure release after request via FastAPI
    with borrow_pipeline(
        backend_config,
        save_transcriptions=True,
    ) as pipeline:
        # Generator dependency: yield the pipeline, then release in teardown
        yield pipeline


# Expose ``__wrapped__`` to allow pytest monkeypatching of dependency overrides.
# FastAPI wraps callables passed to Depends internally, but when tests import the
# original function directly they may expect this attribute for easy stubbing.
# Setting it explicitly keeps the public behaviour unchanged while improving
# testability.
def _get_asr_pipeline_unwrapped() -> NoReturn:
    """Placeholder for tests to monkeypatch. Returns WhisperPipeline when patched.

    Raises:
        RuntimeError: Always raised unless this function is monkeypatched in tests.
    """
    raise RuntimeError("This placeholder should be monkeypatched in tests.")


# Assign to avoid FastAPI/inspect wrapper loop issues while providing the attribute.
get_asr_pipeline.__wrapped__ = _get_asr_pipeline_unwrapped  # type: ignore[attr-defined]


def get_file_handler() -> FileHandler:
    """Dependency to provide file handler instance.

    Returns:
        FileHandler: File handler instance for managing uploads and cleanup
    """
    return FileHandler()


# Expose ``__wrapped__`` to allow pytest monkeypatching of dependency overrides.
# FastAPI wraps callables passed to Depends internally, but when tests import the
# original function directly they may expect this attribute for easy stubbing.
# Setting it explicitly keeps the public behaviour unchanged while improving
# testability.
def _get_file_handler_unwrapped() -> NoReturn:
    """Placeholder for tests to monkeypatch. Returns FileHandler when patched.

    Raises:
        RuntimeError: Always raised unless this function is monkeypatched in tests.
    """
    raise RuntimeError("This placeholder should be monkeypatched in tests.")


# Assign to avoid FastAPI/inspect wrapper loop issues while providing the attribute.
get_file_handler.__wrapped__ = _get_file_handler_unwrapped  # type: ignore[attr-defined]
