"""ASR backend integrations and configuration classes.

This module provides backend implementations for different ASR engines,
focusing on the Hugging Face Transformers integration.
"""

import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    pipeline,
)
from transformers.utils import logging as hf_logging

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
    chunk_length: int


class ASRBackend(ABC):  # pylint: disable=too-few-public-methods
    """Abstract base class for ASR backends."""

    @abstractmethod
    def process_audio(
        self,
        audio_file_path: str,
        language: Optional[str],
        task: str,
        return_timestamps_value: Union[bool, str],
        # Potentially other common config options can go here
    ) -> Dict[str, Any]:
        """Processes the audio file and returns the result."""


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

            model_load_kwargs = {
                "torch_dtype": (
                    torch.float16 if self.config.dtype == "float16" else torch.float32
                ),
                "use_safetensors": True,
            }

            # For newer transformers versions, SDPA is the native way to get
            # BetterTransformer-like optimizations.
            if self.effective_device != "cpu":
                model_load_kwargs["attn_implementation"] = "sdpa"

            try:
                logger.info(
                    "Loading ASR model '%s' with kwargs: %s",
                    self.config.model_name,
                    model_load_kwargs,
                )
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.config.model_name, **model_load_kwargs
                )
                tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                feature_extractor = AutoFeatureExtractor.from_pretrained(
                    self.config.model_name
                )

                self.asr_pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=tokenizer,
                    feature_extractor=feature_extractor,
                    device=self.effective_device,
                )

            except (OSError, ValueError, RuntimeError, ImportError) as e:
                logger.error(
                    "Failed to load ASR model '%s': %s",
                    self.config.model_name,
                    str(e),
                    exc_info=True,
                )
                raise TranscriptionError(f"Failed to load ASR model: {str(e)}") from e

    def process_audio(
        self,
        audio_file_path: str,
        language: Optional[str],
        task: str,
        return_timestamps_value: Union[bool, str],
    ) -> Dict[str, Any]:
        """Processes an audio file and returns the transcription result."""
        if self.asr_pipe is None:
            self._initialize_pipeline()

        start_time = time.perf_counter()

        # ------------------------------------------------------------------
        # Timestamp capability detection for distil-whisper variants
        # ------------------------------------------------------------------
        _return_timestamps_value = return_timestamps_value
        if _return_timestamps_value and "distil-whisper" in self.config.model_name:
            # distil-whisper versions ≥ v2 have timestamp support; earlier ones do not.
            # Determine this heuristically from the model name.
            # Examples that support timestamps:
            #   distil-whisper/distil-large-v2
            #   distil-whisper/distil-medium.en-v2
            # Anything ending in "-v2" or "-v3" (or greater) is considered supported.
            supports_timestamps = False
            try:
                # Extract the suffix after the last "-v"
                last_part = self.config.model_name.split("-v")[-1]
                version_num = int(last_part.split("/")[0].split(".")[0])
                supports_timestamps = version_num >= 2
            except (ValueError, IndexError):
                # Fallback: if model name explicitly contains "large-v2" etc.
                supports_timestamps = any(
                    token in self.config.model_name for token in ("-v2", "-v3", "-v4")
                )

            if not supports_timestamps:
                logger.warning(
                    "Timestamp generation not supported for model %s; disabling.",
                    self.config.model_name,
                )
                _return_timestamps_value = False

        if _return_timestamps_value:
            gen_cfg = getattr(self.asr_pipe.model, "generation_config", None)
            no_ts_token_id = getattr(gen_cfg, "no_timestamps_token_id", None)
            if no_ts_token_id is None:
                logger.warning(
                    "Timestamp generation not properly configured for model %s; disabling.",
                    self.config.model_name,
                )
                _return_timestamps_value = False

        # These are the arguments that will be passed to the pipeline
        # Suppress noisy warnings from HF transformers related to experimental chunk_length and deprecations.
        warnings.filterwarnings(
            "ignore",
            message="Using `chunk_length_s` is very experimental*",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
        hf_logging.set_verbosity_error()

        pipeline_kwargs = {
            "chunk_length_s": self.config.chunk_length,
            "batch_size": self.config.batch_size,
            "return_timestamps": _return_timestamps_value,
            "ignore_warning": True,
            "generate_kwargs": {
                "no_repeat_ngram_size": 3,  # from original script
                "temperature": 0,  # from original script
            },
        }

        # Determine if the model is multilingual.
        # Newer Transformers versions expose language/task maps via `task_to_id`,
        # older checkpoints used `lang_to_id`. We treat presence of either as a
        # sign the model supports multilingual/translation tasks.
        lang_to_id = getattr(self.asr_pipe.model.config, "lang_to_id", None)
        task_to_id = getattr(self.asr_pipe.model.config, "task_to_id", None)

        is_multilingual = False
        if task_to_id is not None:
            is_multilingual = True
        elif isinstance(lang_to_id, dict) and len(lang_to_id) > 1:
            is_multilingual = True

        # Previously we blocked translate for English-only models. With the updated
        # check, only warn (do not raise) so the underlying pipeline can handle it.
        if not is_multilingual and task == "translate":
            logger.warning(
                "Translate requested but multilingual markers not found for model %s; proceeding anyway.",
                self.config.model_name,
            )

        # Always add task and language parameters so Whisper knows we want translation.
        pipeline_kwargs["generate_kwargs"]["task"] = task
        # If translate task and no explicit language provided, default to English
        if language and language.lower() != "none":
            pipeline_kwargs["generate_kwargs"]["language"] = language
        elif task == "translate":
            pipeline_kwargs["generate_kwargs"]["language"] = "en"

        # Convert to WAV if extension not among standard Whisper-friendly set
        from insanely_fast_whisper_api.audio.conversion import (
            ensure_wav,  # local import to avoid heavy deps at import time
        )

        converted_path = ensure_wav(audio_file_path)

        try:
            outputs = self.asr_pipe(str(converted_path), **pipeline_kwargs)
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
                        "Successfully completed transcription with chunk-level "
                        "timestamps fallback for %s",
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
        except (OSError, ValueError, MemoryError, TypeError, IndexError) as e:
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
