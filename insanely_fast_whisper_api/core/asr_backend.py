"""ASR backend integrations and configuration classes.

This module provides backend implementations for different ASR engines,
focusing on the Hugging Face Transformers integration.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import time

import torch
from transformers import pipeline

from insanely_fast_whisper_api.core.errors import (
    DeviceNotFoundError,
    TranscriptionError,
)
from insanely_fast_whisper_api.core.utils import convert_device_string

# Placeholder for logger, will be configured properly later
logger = logging.getLogger(__name__)


@dataclass
class HuggingFaceBackendConfig:
    """Configuration for HuggingFaceBackend."""

    model_name: str
    device: str
    dtype: str
    batch_size: int
    better_transformer: bool
    chunk_length: int


class ASRBackend(ABC):  # pylint: disable=too-few-public-methods
    """Abstract base class for ASR backends."""

    @abstractmethod
    def transcribe(
        self,
        audio_file_path: str,
        language: Optional[str],
        task: str,
        return_timestamps_value: Union[bool, str],
        # Potentially other common config options can go here
    ) -> Dict[str, Any]:
        """Transcribes the audio file and returns the result."""


class HuggingFaceBackend(ASRBackend):  # pylint: disable=too-few-public-methods
    """ASR Backend using Hugging Face Transformers pipeline."""

    def __init__(self, config: HuggingFaceBackendConfig):
        self.config = config
        self.effective_device = convert_device_string(self.config.device)
        self.asr_pipe = None  # Lazy initialization

        self._validate_device()

    def _validate_device(self):
        if "cuda" in self.effective_device and not torch.cuda.is_available():
            raise DeviceNotFoundError(
                f"CUDA device {self.effective_device} requested but CUDA is not "
                f"available. Try using 'cpu' instead."
            )
        if self.effective_device == "mps" and not torch.backends.mps.is_available():
            raise DeviceNotFoundError(
                "MPS device requested but MPS (Apple Silicon) is not available. "
                "Try using 'cpu' instead."
            )

    def _initialize_pipeline(self):
        if self.asr_pipe is None:
            logger.info(
                "ASR using device: %s, model: %s, dtype: %s",
                self.effective_device,
                self.config.model_name,
                self.config.dtype,
            )
            model_kwargs = {
                "model": self.config.model_name,
                "device": self.effective_device,
                "torch_dtype": (
                    torch.float16 if self.config.dtype == "float16" else torch.float32
                ),
            }
            try:
                logger.info("Loading ASR model '%s'...", self.config.model_name)
                self.asr_pipe = pipeline("automatic-speech-recognition", **model_kwargs)

                if self.config.better_transformer:
                    logger.info("Applying BetterTransformer optimization.")
                    try:
                        self.asr_pipe.model = self.asr_pipe.model.to_bettertransformer()
                    except (RuntimeError, AttributeError, ImportError) as e_bt:
                        logger.warning(
                            "Could not apply BetterTransformer: %s. Continuing without it.",
                            e_bt,
                        )
            except (OSError, ValueError, RuntimeError, ImportError) as e:
                logger.error(
                    "Failed to load ASR model '%s': %s",
                    self.config.model_name,
                    str(e),
                    exc_info=True,
                )
                raise TranscriptionError(f"Failed to load ASR model: {str(e)}") from e

    def transcribe(
        self,
        audio_file_path: str,
        language: Optional[str],
        task: str,
        return_timestamps_value: Union[bool, str],
    ) -> Dict[str, Any]:
        self._initialize_pipeline()
        if self.asr_pipe is None:  # Should not happen if _initialize_pipeline works
            logger.error("ASR pipeline not initialized before transcription attempt.")
            raise TranscriptionError("ASR pipeline not initialized.")

        logger.info("Starting transcription for: %s", audio_file_path)

        start_time = time.perf_counter()

        pipeline_kwargs = {
            "chunk_length_s": self.config.chunk_length,
            "batch_size": self.config.batch_size,
            "return_timestamps": return_timestamps_value,
            "generate_kwargs": {
                "task": task,
                "no_repeat_ngram_size": 3,  # from original script
                "temperature": 0,  # from original script
            },
        }

        if language and language.lower() != "none":
            pipeline_kwargs["generate_kwargs"]["language"] = language

        try:
            outputs = self.asr_pipe(str(audio_file_path), **pipeline_kwargs)
        except RuntimeError as e:
            # Check if this is the specific tensor size mismatch error in timestamp extraction
            if "expanded size of the tensor" in str(
                e
            ) and "must match the existing size" in str(e):
                logger.warning(
                    "Word-level timestamp extraction failed due to tensor size mismatch. "
                    "Falling back to chunk-level timestamps for %s: %s",
                    audio_file_path,
                    str(e),
                )

                # Retry with chunk-level timestamps instead of word-level
                fallback_kwargs = pipeline_kwargs.copy()
                fallback_kwargs["return_timestamps"] = True  # chunk-level timestamps

                try:
                    outputs = self.asr_pipe(str(audio_file_path), **fallback_kwargs)
                    logger.info(
                        "Successfully completed transcription with chunk-level timestamps fallback for %s",
                        audio_file_path,
                    )
                except (RuntimeError, OSError, ValueError, MemoryError) as fallback_e:
                    logger.error(
                        "Fallback transcription also failed for %s: %s",
                        audio_file_path,
                        str(fallback_e),
                        exc_info=True,
                    )
                    raise TranscriptionError(
                        f"Failed to transcribe audio even with fallback: {str(fallback_e)}"
                    ) from fallback_e
            else:
                # Re-raise other RuntimeErrors
                logger.error(
                    "Transcription failed for %s: %s",
                    audio_file_path,
                    str(e),
                    exc_info=True,
                )
                raise TranscriptionError(f"Failed to transcribe audio: {str(e)}") from e
        except (OSError, ValueError, MemoryError, TypeError) as e:
            logger.error(
                "Transcription failed for %s: %s",
                audio_file_path,
                str(e),
                exc_info=True,
            )
            raise TranscriptionError(f"Failed to transcribe audio: {str(e)}") from e

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        result = {
            "text": outputs["text"].strip(),
            "chunks": outputs.get(
                "chunks"
            ),  # .get() because it might not be present if return_timestamps=False
            "runtime_seconds": round(elapsed_time, 2),
            # Add config used for this specific transcription for clarity
            "config_used": {
                "model": self.config.model_name,
                "device": self.effective_device,
                "batch_size": self.config.batch_size,
                "language": language or "auto",
                "better_transformer": self.config.better_transformer,
                "dtype": self.config.dtype,
                "chunk_length_s": self.config.chunk_length,
                "task": task,
                "return_timestamps": return_timestamps_value,
            },
        }
        logger.info(
            "Transcription completed in %.2fs for %s", elapsed_time, audio_file_path
        )
        return result
