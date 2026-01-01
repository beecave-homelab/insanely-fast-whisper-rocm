"""Centralized orchestrator for transcription with automatic OOM recovery.

This module provides the TranscriptionOrchestrator class which manages the
transcription process, handles retries, and implements fallback strategies
when GPU memory is exhausted.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from insanely_fast_whisper_rocm.core.asr_backend import HuggingFaceBackendConfig
from insanely_fast_whisper_rocm.core.backend_cache import (
    borrow_pipeline,
    invalidate_gpu_cache,
)
from insanely_fast_whisper_rocm.core.errors import (
    InferenceOOMError,
    ModelLoadingOOMError,
    TranscriptionError,
)
from insanely_fast_whisper_rocm.core.progress import ProgressCallback
from insanely_fast_whisper_rocm.utils.constants import MIN_BATCH_SIZE

logger = logging.getLogger(__name__)


def _format_backend_config(config: HuggingFaceBackendConfig) -> str:
    """
    Produce a one-line human-readable summary of a backend configuration.
    
    Parameters:
        config (HuggingFaceBackendConfig): Backend configuration to summarize.
    
    Returns:
        str: Single-line string containing `model_name`, `device`, `dtype`, `batch_size`, and `chunk_length`.
    """
    return (
        f"model={config.model_name} device={config.device} dtype={config.dtype} "
        f"batch_size={config.batch_size} chunk_length={config.chunk_length}"
    )


def _backend_config_to_dict(config: HuggingFaceBackendConfig) -> dict[str, Any]:
    """
    Convert a HuggingFace backend configuration into a JSON-serializable dictionary suitable for UI or debugging.
    
    Returns:
        A dictionary with keys `model_name`, `device`, `dtype`, `batch_size`, `chunk_length`, and `progress_group_size` representing the backend configuration.
    """
    return {
        "model_name": config.model_name,
        "device": config.device,
        "dtype": config.dtype,
        "batch_size": config.batch_size,
        "chunk_length": config.chunk_length,
        "progress_group_size": config.progress_group_size,
    }


class TranscriptionOrchestrator:
    """Orchestrates transcription tasks with automatic OOM recovery and retries."""

    def __init__(self) -> None:
        """Initialize the orchestrator."""
        self.max_retries = 2

    def _get_reduced_config(
        self, config: HuggingFaceBackendConfig
    ) -> HuggingFaceBackendConfig:
        """
        Create a new HuggingFaceBackendConfig with batch_size halved for OOM recovery.
        
        The returned config preserves model_name, device, dtype, chunk_length, and progress_group_size from the input and sets batch_size to max(MIN_BATCH_SIZE, config.batch_size // 2).
        
        Parameters:
            config (HuggingFaceBackendConfig): Current backend configuration.
        
        Returns:
            HuggingFaceBackendConfig: New backend configuration with a reduced batch_size.
        """
        new_batch_size = max(MIN_BATCH_SIZE, config.batch_size // 2)
        logger.info(
            "Reducing batch size from %d to %d for OOM recovery",
            config.batch_size,
            new_batch_size,
        )
        return HuggingFaceBackendConfig(
            model_name=config.model_name,
            device=config.device,
            dtype=config.dtype,
            batch_size=new_batch_size,
            chunk_length=config.chunk_length,
            progress_group_size=config.progress_group_size,
        )

    def _get_cpu_fallback_config(
        self, config: HuggingFaceBackendConfig
    ) -> HuggingFaceBackendConfig:
        """
        Produce a backend configuration adjusted for CPU fallback.
        
        Creates a new HuggingFaceBackendConfig with device set to "cpu", dtype set to "float32", batch_size limited to at most 2, chunk_length limited to at most 15, and progress_group_size preserved.
        
        Parameters:
            config (HuggingFaceBackendConfig): The current backend configuration to adapt for CPU execution.
        
        Returns:
            HuggingFaceBackendConfig: A new configuration suitable for running on CPU.
        """
        logger.info("Creating CPU fallback configuration")
        return HuggingFaceBackendConfig(
            model_name=config.model_name,
            device="cpu",
            dtype="float32",  # CPU usually prefers float32
            batch_size=min(config.batch_size, 2),
            chunk_length=min(config.chunk_length, 15),
            progress_group_size=config.progress_group_size,
        )

    def run_transcription(
        self,
        audio_path: str,
        backend_config: HuggingFaceBackendConfig,
        task: str = "transcribe",
        language: str | None = None,
        timestamp_type: bool | str = True,
        progress_callback: ProgressCallback | None = None,
        warning_callback: Callable[[str], None] | None = None,
        save_transcriptions: bool = True,
        output_dir: str = "transcripts",
    ) -> dict[str, Any]:
        """
        Orchestrates transcription with automatic retries and out-of-memory (OOM) recovery.
        
        Runs transcription attempts using the provided backend configuration, automatically retrying with reduced GPU batch size or switching to a CPU fallback when GPU OOMs occur. Attaches a per-attempt history under the `orchestrator_attempts` key in the returned result.
        
        Parameters:
            audio_path: Path to the audio file to transcribe.
            backend_config: Initial backend configuration used to acquire a pipeline.
            task: Either "transcribe" or "translate".
            language: Optional language code to force recognition/translation.
            timestamp_type: Controls timestamps returned by the pipeline; may be a boolean or one of "chunk" or "word".
            progress_callback: Optional callback invoked with progress updates from the pipeline.
            warning_callback: Optional callback invoked with user-facing warning messages (e.g., when recovery actions occur).
            save_transcriptions: If true, persist transcription outputs to disk via the pipeline.
            output_dir: Directory to write persisted transcription files when saving is enabled.
        
        Returns:
            A dictionary containing the transcription result. The dictionary includes an `orchestrator_attempts` entry describing each attempt's config, status, and any recovery actions or errors.
        
        Raises:
            ModelLoadingOOMError: When model loading fails with OOM and no recovery is possible.
            InferenceOOMError: When inference repeatedly fails with OOM and no recovery is possible.
            TranscriptionError: For other transcription failures or when maximum retry attempts are exhausted.
        """
        current_config = backend_config
        attempt_index = 0
        max_attempts = self.max_retries + 1
        attempt_history: list[dict[str, Any]] = []

        while attempt_index < max_attempts:
            try:
                attempt_no = attempt_index + 1
                attempt_msg = (
                    f"Attempt {attempt_no}/{max_attempts}: "
                    f"{_format_backend_config(current_config)}"
                )
                logger.info(attempt_msg)
                if warning_callback:
                    warning_callback(attempt_msg)

                attempt_history.append({
                    "attempt": attempt_no,
                    "config": _backend_config_to_dict(current_config),
                    "status": "started",
                })

                logger.info(
                    "Transcription attempt %d/%d using device: %s, batch_size: %d",
                    attempt_no,
                    max_attempts,
                    current_config.device,
                    current_config.batch_size,
                )

                with borrow_pipeline(
                    current_config,
                    save_transcriptions=save_transcriptions,
                    output_dir=output_dir,
                ) as pipeline:
                    result = pipeline.process(
                        audio_file_path=audio_path,
                        language=language,
                        task=task,
                        timestamp_type=timestamp_type,
                        progress_callback=progress_callback,
                    )

                # Attach attempt history for callers (WebUI/API) to display.
                attempt_history[-1]["status"] = "succeeded"
                result["orchestrator_attempts"] = attempt_history
                return result

            except ModelLoadingOOMError as e:
                logger.warning(
                    "Model loading OOM on %s: %s", current_config.device, str(e)
                )
                if attempt_history:
                    attempt_history[-1]["status"] = "failed"
                    attempt_history[-1]["error_type"] = type(e).__name__
                    attempt_history[-1]["error"] = str(e)
                if current_config.device == "cpu":
                    logger.error("OOM on CPU during model loading. Cannot recover.")
                    raise

                # Skip GPU retry for model loading OOM, go directly to CPU fallback
                cpu_config = self._get_cpu_fallback_config(current_config)
                msg = (
                    "Model load OOM. Switching configuration: "
                    f"{_format_backend_config(current_config)} -> "
                    f"{_format_backend_config(cpu_config)}"
                )
                if attempt_history:
                    attempt_history[-1]["recovery_action"] = msg
                logger.info(msg)
                if warning_callback:
                    warning_callback(msg)

                # Invalidate GPU cache to free memory before switching to CPU
                invalidate_gpu_cache()

                current_config = cpu_config
                attempt_index += 1
                continue

            except InferenceOOMError as e:
                logger.warning("Inference OOM on %s: %s", current_config.device, str(e))
                if attempt_history:
                    attempt_history[-1]["status"] = "failed"
                    attempt_history[-1]["error_type"] = type(e).__name__
                    attempt_history[-1]["error"] = str(e)
                if current_config.device == "cpu":
                    logger.error("OOM on CPU during inference. Cannot recover.")
                    raise

                attempt_index += 1
                if attempt_index == 1:
                    # First retry: reduce batch size on GPU
                    new_config = self._get_reduced_config(current_config)
                    msg = (
                        "Inference OOM. Switching configuration: "
                        f"{_format_backend_config(current_config)} -> "
                        f"{_format_backend_config(new_config)}"
                    )
                    if attempt_history:
                        attempt_history[-1]["recovery_action"] = msg
                    logger.info(msg)
                    if warning_callback:
                        warning_callback(msg)
                    current_config = new_config
                else:
                    # Second retry or subsequent: fallback to CPU
                    cpu_config = self._get_cpu_fallback_config(current_config)
                    msg = (
                        "Inference OOM. Switching configuration: "
                        f"{_format_backend_config(current_config)} -> "
                        f"{_format_backend_config(cpu_config)}"
                    )
                    if attempt_history:
                        attempt_history[-1]["recovery_action"] = msg
                    logger.info(msg)
                    if warning_callback:
                        warning_callback(msg)

                    # Invalidate GPU cache to free memory before switching to CPU
                    invalidate_gpu_cache()

                    current_config = cpu_config

                continue

            except TranscriptionError:
                if attempt_history:
                    attempt_history[-1]["status"] = "failed"
                # Re-raise non-OOM transcription errors
                raise

            except Exception as e:
                if attempt_history:
                    attempt_history[-1]["status"] = "failed"
                    attempt_history[-1]["error_type"] = type(e).__name__
                    attempt_history[-1]["error"] = str(e)
                logger.error(
                    "Unexpected error during transcription: %s", str(e), exc_info=True
                )
                raise TranscriptionError(f"Unexpected error: {str(e)}") from e

        # This should theoretically not be reached as we either return or raise
        raise TranscriptionError("Maximum retry attempts reached without success")


def create_orchestrator() -> TranscriptionOrchestrator:
    """
    Create a TranscriptionOrchestrator.
    
    Returns:
        A new TranscriptionOrchestrator instance.
    """
    return TranscriptionOrchestrator()


Orchestrator = TranscriptionOrchestrator