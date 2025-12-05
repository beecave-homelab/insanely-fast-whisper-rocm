"""ASR backend integrations and configuration classes.

This module provides backend implementations for different ASR engines,
focusing on the Hugging Face Transformers integration.
"""

from __future__ import annotations

import gc
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
)
from transformers.utils import logging as hf_logging

from insanely_fast_whisper_api.core.cancellation import CancellationToken
from insanely_fast_whisper_api.core.errors import (
    DeviceNotFoundError,
    TranscriptionError,
)
from insanely_fast_whisper_api.core.progress import NoOpProgress, ProgressCallback
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
    progress_group_size: int


class ASRBackend(ABC):  # pylint: disable=too-few-public-methods
    """Abstract base class for ASR backends."""

    @abstractmethod
    def process_audio(
        self,
        audio_file_path: str,
        language: str | None,
        task: str,
        return_timestamps_value: bool | str,
        # Potentially other common config options can go here
        progress_cb: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Processes the audio file and returns the result."""


class HuggingFaceBackend(ASRBackend):  # pylint: disable=too-few-public-methods
    """ASR Backend using Hugging Face Transformers pipeline."""

    def __init__(self, config: HuggingFaceBackendConfig) -> None:
        """Initialize the backend with the given configuration.

        Args:
            config: Backend configuration including model, device, dtype,
                batch size, and chunk length.
        """
        self.config = config
        self.effective_device = convert_device_string(self.config.device)
        self.asr_pipe = None  # Lazy initialization

        self._validate_device()

    def _validate_device(self) -> None:
        """Validate that the requested device is available, else raise.

        Raises:
            DeviceNotFoundError: If the requested CUDA or MPS device is not
                available on the system.
        """
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

    def _initialize_pipeline(self, progress_cb: ProgressCallback | None = None) -> None:
        """Lazily construct the Transformers pipeline if not already created.

        Emits model load progress callbacks if provided.

        Args:
            progress_cb: Optional progress callback.

        Raises:
            TranscriptionError: If the ASR model or associated components fail
                to load.
            ImportError: Propagated if a required backend or package is missing
                and cannot be recovered.
            OSError: Propagated if underlying IO or model loading fails in a way
                that cannot be wrapped.
            RuntimeError: Propagated for low-level framework errors that may
                occur prior to wrapping.
            ValueError: Propagated for invalid configuration detected during
                model initialization when not recoverable.
        """
        if self.asr_pipe is None:
            cb = progress_cb or NoOpProgress()
            cb.on_model_load_started()
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

            # For newer transformers, SDPA is fast. On ROCm it sometimes fails at
            # runtime on certain stacks; per user preference, try SDPA first and
            # fallback to 'eager' only if needed. Keep SDPA on CUDA.
            if self.effective_device != "cpu":
                is_rocm = getattr(torch.version, "hip", None) is not None
                if is_rocm:
                    model_load_kwargs["attn_implementation"] = "sdpa"
                    logger.info(
                        "ROCm detected; trying attn_implementation='sdpa' first"
                    )
                else:
                    model_load_kwargs["attn_implementation"] = "sdpa"

            try:
                logger.info(
                    "Loading ASR model '%s' with kwargs: %s",
                    self.config.model_name,
                    model_load_kwargs,
                )
                try:
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        self.config.model_name, **model_load_kwargs
                    )
                except (OSError, ValueError, RuntimeError, ImportError) as e_first:
                    # On ROCm, fallback to 'eager' if SDPA attempt fails at load.
                    if (
                        getattr(torch.version, "hip", None) is not None
                        and model_load_kwargs.get("attn_implementation") == "sdpa"
                    ):
                        logger.warning(
                            "Model load with SDPA failed on ROCm, retrying with "
                            "attn_implementation='eager': %s",
                            str(e_first),
                        )
                        model_load_kwargs["attn_implementation"] = "eager"
                        logger.info(
                            "Reloading ASR model '%s' with kwargs: %s",
                            self.config.model_name,
                            model_load_kwargs,
                        )
                        model = AutoModelForSpeechSeq2Seq.from_pretrained(
                            self.config.model_name, **model_load_kwargs
                        )
                    else:
                        raise
                tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                feature_extractor = AutoFeatureExtractor.from_pretrained(
                    self.config.model_name
                )

                # Backfill missing Whisper generation config for fine-tuned
                # checkpoints that did not upload generation_config.json.
                # This enables timestamp extraction and multilingual task mapping.
                try:
                    gen_cfg = getattr(model, "generation_config", None)
                    missing_ts_cfg = (
                        gen_cfg is None
                        or getattr(gen_cfg, "no_timestamps_token_id", None) is None
                    )

                    name = self.config.model_name.lower()
                    if missing_ts_cfg and ("whisper" in name):
                        is_en = (".en" in name) or ("-en" in name)
                        base_id = None

                        if "large" in name:
                            if ("large-v3" in name) or ("v3" in name):
                                base_id = "openai/whisper-large-v3"
                            elif ("large-v2" in name) or ("v2" in name):
                                base_id = "openai/whisper-large-v2"
                            else:
                                base_id = "openai/whisper-large-v2"
                        elif "medium" in name:
                            base_id = "openai/whisper-medium" + (".en" if is_en else "")
                        elif "small" in name:
                            base_id = "openai/whisper-small" + (".en" if is_en else "")
                        elif "base" in name:
                            base_id = "openai/whisper-base" + (".en" if is_en else "")
                        elif "tiny" in name:
                            base_id = "openai/whisper-tiny" + (".en" if is_en else "")

                        if base_id:
                            logger.info(
                                "Backfilling gen config from %s for %s",
                                base_id,
                                self.config.model_name,
                            )
                            model.generation_config = GenerationConfig.from_pretrained(
                                base_id
                            )
                except Exception as gen_e:  # pylint: disable=broad-except
                    logger.warning(
                        "Failed to backfill generation_config for %s: %s",
                        self.config.model_name,
                        str(gen_e),
                    )

                self.asr_pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=tokenizer,
                    feature_extractor=feature_extractor,
                    device=self.effective_device,
                )

                cb.on_model_load_finished()

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
        language: str | None,
        task: str,
        return_timestamps_value: bool | str,
        progress_cb: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Process an audio file and return the transcription result.

        Args:
            audio_file_path: Input audio path.
            language: Optional language code.
            task: "transcribe" or "translate".
            return_timestamps_value: Whether/how to return timestamps.
            progress_cb: Optional progress reporter.
            cancellation_token: Optional cooperative cancellation token.

        Returns:
            dict[str, Any]: Result with text, optional chunks, runtime, and
            config used.

        Raises:
            TranscriptionError: If model loading or inference fails.
        """
        cb = progress_cb or NoOpProgress()
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        if self.asr_pipe is None:
            self._initialize_pipeline(progress_cb=cb)
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()

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
                    (
                        "Timestamp generation not properly configured for model %s; "
                        "disabling."
                    ),
                    self.config.model_name,
                )
                _return_timestamps_value = False

        # These are the arguments that will be passed to the pipeline
        # Suppress noisy warnings from HF transformers related to experimental
        # chunk_length and deprecations.
        warnings.filterwarnings(
            "ignore",
            message="Using `chunk_length_s` is very experimental*",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
        hf_logging.set_verbosity_error()

        # CRITICAL FIX: Disable chunk_length_s when using word-level timestamps
        # to avoid Transformers bug where all words get the same timestamp.
        # The manual chunking in pipeline.py handles audio splitting, so
        # Transformers should process each chunk without further internal chunking.
        chunk_length_value = self.config.chunk_length
        if _return_timestamps_value == "word":
            chunk_length_value = None
            logger.debug(
                "Disabling chunk_length_s for word-level timestamps to avoid "
                "Transformers internal chunking conflict"
            )

        pipeline_kwargs = {
            "chunk_length_s": chunk_length_value,
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
                (
                    "Translate requested but multilingual markers not found for "
                    "model %s; proceeding anyway."
                ),
                self.config.model_name,
            )

        # Older checkpoints may ship with generation configs that predate the
        # introduction of task/language mappings. Passing "task" or "language"
        # to such models triggers a ValueError. Only forward these parameters
        # when the generation config exposes the required attributes.
        gen_cfg = getattr(self.asr_pipe.model, "generation_config", None)
        has_task_mappings = False
        if gen_cfg is not None:
            has_task_mappings = any(
                getattr(gen_cfg, attr, None) is not None
                for attr in ("task_to_id", "lang_to_id")
            )

        if has_task_mappings:
            pipeline_kwargs["generate_kwargs"]["task"] = task
            # If translate task and no explicit language provided, default to English
            if language and language.lower() != "none":
                pipeline_kwargs["generate_kwargs"]["language"] = language
            elif task == "translate":
                pipeline_kwargs["generate_kwargs"]["language"] = "en"
        elif task != "transcribe" or language:
            logger.warning(
                "Generation config for model %s lacks task/language mappings; "
                "falling back to default transcription.",
                self.config.model_name,
            )

        try:
            logger.debug(
                "Calling ASR pipeline: audio=%s, chunk_length_s=%s, batch_size=%d, "
                "return_timestamps=%s",
                audio_file_path,
                chunk_length_value,
                self.config.batch_size,
                _return_timestamps_value,
            )
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            outputs = self.asr_pipe(str(audio_file_path), **pipeline_kwargs)
        except RuntimeError as e:
            # Check if this is the specific tensor size mismatch error in
            # timestamp extraction
            if "expanded size of the tensor" in str(
                e
            ) and "must match the existing size" in str(e):
                logger.warning(
                    (
                        "Word-level timestamp extraction failed due to "
                        "tensor size mismatch. Falling back to chunk-level "
                        "timestamps for %s: %s"
                    ),
                    audio_file_path,
                    str(e),
                )

                # Retry with chunk-level timestamps instead of word-level
                fallback_kwargs = pipeline_kwargs.copy()
                fallback_kwargs["return_timestamps"] = True  # chunk-level timestamps

                try:
                    if cancellation_token is not None:
                        cancellation_token.raise_if_cancelled()
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
                        "Failed to transcribe audio even with fallback: "
                        f"{str(fallback_e)}"
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

        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()

        # The pipeline may return 'chunks' or 'segments'. For consistency,
        # we normalize to a 'segments' key and also keep 'chunks' for
        # backward compatibility if it was the original key.
        chunks = outputs.get("chunks")
        segments = outputs.get("segments", chunks)
        logger.debug(
            "Raw ASR pipeline output: text_len=%d, chunks=%d, segments=%d, "
            "elapsed=%.2fs",
            len(outputs.get("text", "")),
            len(chunks) if chunks else 0,
            len(segments) if segments else 0,
            elapsed_time,
        )

        result = {
            "text": outputs["text"].strip(),
            "chunks": chunks,  # Keep for backward compatibility
            "segments": segments,  # Normalize to 'segments'
            "runtime_seconds": round(elapsed_time, 2),
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
        logger.debug(
            "Returning normalized result: text_len=%d, segments=%d, chunks=%d",
            len(result["text"]),
            len(result["segments"]) if result["segments"] else 0,
            len(result["chunks"]) if result["chunks"] else 0,
        )
        return result

    def close(self) -> None:
        """Release model resources and free accelerator caches.

        This method deletes the internal Transformers pipeline instance and
        attempts to free device memory on supported backends (CUDA/MPS). It is
        safe to call multiple times.

        Notes:
            - After calling this, the next invocation will lazily recreate the
              pipeline on demand.
        """
        try:
            if getattr(self, "asr_pipe", None) is not None:
                # Explicitly drop references to model/tokenizer/feature_extractor
                self.asr_pipe = None
        finally:
            # Best-effort device cache cleanup
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            try:
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            # Force immediate garbage collection to reclaim memory
            gc.collect()

    # Backwards-friendly alias
    release = close
