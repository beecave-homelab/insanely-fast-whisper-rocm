"""ASR backend integrations and configuration classes.

This module provides backend implementations for different ASR engines,
focusing on the Hugging Face Transformers integration.
"""

from __future__ import annotations

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

    def _process_with_manual_chunking(
        self,
        *,
        audio_file_path: str,
        language: str | None,
        task: str,
        return_timestamps_value: bool | str,
        cb: ProgressCallback,
        pipeline_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Process audio by splitting into chunks and batching for progress.

        This path enables accurate percentage progress based on total chunks.

        Args:
            audio_file_path: Original input audio path.
            language: Optional language code.
            task: "transcribe" or "translate".
            return_timestamps_value: Whether/how to return timestamps.
            cb: Progress callback to emit progress events.
            pipeline_kwargs: Keyword args forwarded to the HF pipeline.

        Raises:
            TranscriptionError: If the ASR model or associated components fail
                to load.

        Returns:
            dict[str, Any]: Merged result across all chunks.

        """
        from insanely_fast_whisper_api.audio.conversion import ensure_wav
        from insanely_fast_whisper_api.audio.processing import split_audio

        start_time = time.perf_counter()

        cb.on_audio_loading_started(audio_file_path)
        converted_path = ensure_wav(audio_file_path)
        cb.on_audio_loading_finished(duration_sec=None)

        chunks_with_offsets = split_audio(
            converted_path,
            chunk_duration=float(self.config.chunk_length),
            chunk_overlap=0.0,
            min_chunk_duration=1.0,
        )

        total_chunks = len(chunks_with_offsets)
        if total_chunks == 0:
            raise TranscriptionError("No audio chunks produced for transcription.")

        cb.on_chunking_started(total_chunks)

        def _shift_chunk_timestamps(
            items: list[dict[str, object]] | None, offset: float
        ) -> list[dict[str, object]]:
            if not items:
                return []
            shifted: list[dict[str, object]] = []
            for it in items:
                ts = it.get("timestamp") if isinstance(it, dict) else None
                if isinstance(ts, (list, tuple)) and len(ts) == 2:
                    start, end = ts
                    try:
                        start_f = float(start) if start is not None else None
                        end_f = float(end) if end is not None else None
                    except (TypeError, ValueError):
                        start_f, end_f = None, None
                    new_ts = (
                        (start_f + offset) if start_f is not None else None,
                        (end_f + offset) if end_f is not None else None,
                    )
                    new_item = dict(it)
                    new_item["timestamp"] = new_ts
                    shifted.append(new_item)
                else:
                    shifted.append(dict(it))
            return shifted

        combined_text_parts: list[str] = []
        combined_chunks: list[dict[str, object]] = []

        # Progress groups: decouple progress frequency from model batch_size.
        # We submit smaller groups to the pipeline so progress updates are more
        # frequent even if the user-specified batch_size is large.
        progress_group_size = max(1, int(self.config.progress_group_size))
        groups: list[list[tuple[str, float]]] = [
            chunks_with_offsets[i : i + progress_group_size]
            for i in range(0, total_chunks, progress_group_size)
        ]
        total_groups = len(groups)
        cb.on_inference_started(total_groups)

        completed_chunks = 0

        for group_index, group in enumerate(groups):
            batch_paths = [p for p, _off in group]
            batch_offsets = [_off for _p, _off in group]

            try:
                batch_outputs = self.asr_pipe(batch_paths, **pipeline_kwargs)
            except (
                RuntimeError,
                OSError,
                ValueError,
                MemoryError,
                TypeError,
                IndexError,
            ) as e:
                logger.error(
                    "Transcription failed for batch starting at %s: %s",
                    batch_paths[0] if batch_paths else converted_path,
                    str(e),
                    exc_info=True,
                )
                raise TranscriptionError(f"Failed to transcribe audio: {str(e)}") from e

            if isinstance(batch_outputs, dict):
                batch_outputs_list = [batch_outputs]
            else:
                batch_outputs_list = list(batch_outputs)

            for i, out in enumerate(batch_outputs_list):
                text_part = out.get("text", "") if isinstance(out, dict) else ""
                if isinstance(text_part, str) and text_part:
                    combined_text_parts.append(text_part.strip())
                out_chunks = out.get("chunks") if isinstance(out, dict) else None
                offset = float(batch_offsets[i]) if i < len(batch_offsets) else 0.0
                combined_chunks.extend(_shift_chunk_timestamps(out_chunks, offset))
                cb.on_chunk_done(completed_chunks)
                completed_chunks += 1
            cb.on_inference_batch_done(group_index)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        result = {
            "text": " ".join(combined_text_parts).strip(),
            "chunks": combined_chunks if return_timestamps_value else None,
            "runtime_seconds": round(elapsed_time, 2),
            "config_used": {
                "model": self.config.model_name,
                "device": self.effective_device,
                "batch_size": self.config.batch_size,
                "language": language or "auto",
                "dtype": self.config.dtype,
                "chunk_length_s": self.config.chunk_length,
                "progress_group_size": self.config.progress_group_size,
                "task": task,
                "return_timestamps": return_timestamps_value,
            },
        }
        # Stop progress before export begins so "Saved ..." lines print cleanly.
        cb.on_completed()
        return result

    def process_audio(
        self,
        audio_file_path: str,
        language: str | None,
        task: str,
        return_timestamps_value: bool | str,
        progress_cb: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Process an audio file and return the transcription result.

        Args:
            audio_file_path: Input audio path.
            language: Optional language code.
            task: "transcribe" or "translate".
            return_timestamps_value: Whether/how to return timestamps.
            progress_cb: Optional progress reporter.

        Returns:
            dict[str, Any]: Result with text, optional chunks, runtime, and
            config used.

        Raises:
            TranscriptionError: If model loading or inference fails.
        """
        cb = progress_cb or NoOpProgress()
        if self.asr_pipe is None:
            self._initialize_pipeline(progress_cb=cb)

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

        # Accurate progress path: manual chunking + batch loop
        return self._process_with_manual_chunking(
            audio_file_path=audio_file_path,
            language=language,
            task=task,
            return_timestamps_value=_return_timestamps_value,
            cb=cb,
            pipeline_kwargs=pipeline_kwargs,
        )
        # Convert to WAV if extension not among standard Whisper-friendly set
        from insanely_fast_whisper_api.audio.conversion import (
            ensure_wav,  # local import to avoid heavy deps at import time
        )

        cb.on_audio_loading_started(audio_file_path)
        converted_path = ensure_wav(audio_file_path)
        cb.on_audio_loading_finished(duration_sec=None)

        try:
            cb.on_inference_started(total_batches=None)
            outputs = self.asr_pipe(str(converted_path), **pipeline_kwargs)
            cb.on_inference_batch_done(0)
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
        cb.on_completed()
        return result
