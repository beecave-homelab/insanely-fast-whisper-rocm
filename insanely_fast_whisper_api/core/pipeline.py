"""ASR pipeline definition, including base classes and concrete implementations."""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from insanely_fast_whisper_api.audio import conversion as audio_conversion
from insanely_fast_whisper_api.audio import processing as audio_processing
from insanely_fast_whisper_api.audio import results as audio_results
from insanely_fast_whisper_api.core.asr_backend import ASRBackend
from insanely_fast_whisper_api.core.errors import TranscriptionError
from insanely_fast_whisper_api.core.progress import NoOpProgress, ProgressCallback
from insanely_fast_whisper_api.core.storage import BaseStorage, StorageFactory
from insanely_fast_whisper_api.utils import file_utils
from insanely_fast_whisper_api.utils.filename_generator import (
    FilenameGenerator,
    StandardFilenameStrategy,
    TaskType,
)

logger = logging.getLogger(__name__)

InputType = TypeVar("InputType")

# ---------------------------------------------------------------------------
# Lightweight configuration/result dataclasses for test compatibility
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:  # pylint: disable=too-many-instance-attributes
    """Stub config passed around by some higher-level APIs/tests."""

    model: str = "openai/whisper-base"
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 1
    chunk_length: int = 30


@dataclass
class TranscriptionResult:
    """Simplified representation of a transcription output used in tests."""

    text: str
    chunks: Any | None = None
    language: str = "en"


@dataclass
class ProgressEvent:
    """Dataclass to represent a progress event."""

    event_type: str  # e.g., "chunk_start", "chunk_complete", "pipeline_complete"
    pipeline_id: str
    file_path: str
    chunk_num: int | None = None
    total_chunks: int | None = None
    message: str | None = None
    result: dict[str, Any] | None = None  # For chunk_complete or pipeline_complete


class BasePipeline(ABC):
    """Base class for ASR pipelines using the Template Method pattern."""

    def __init__(
        self,
        asr_backend: ASRBackend,
        storage_backend: BaseStorage | None = None,
        save_transcriptions: bool = True,
        output_dir: str = "transcripts",
    ) -> None:
        """Initializes the BasePipeline.

        Args:
            asr_backend: The ASR backend to use for transcription.
            storage_backend: The storage backend for saving results.
            save_transcriptions: Whether to save transcriptions to disk.
            output_dir: The directory to save transcriptions in.
        """
        self.asr_backend = asr_backend
        self.storage_backend = (
            storage_backend if storage_backend else StorageFactory.create()
        )
        self.save_transcriptions = save_transcriptions
        self.output_dir = Path(output_dir)
        self._listeners: list[ProgressCallback] = []
        self.pipeline_id = str(uuid.uuid4())
        # Initialize filename generator with standard strategy
        self._filename_generator = FilenameGenerator(
            strategy=StandardFilenameStrategy()
        )

    def add_listener(self, callback: ProgressCallback) -> None:
        """Registers an observer for progress events."""
        self._listeners.append(callback)

    def _notify_listeners(self, event: ProgressEvent) -> None:
        """Notifies all registered listeners about an event."""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Error in progress listener: %s", e, exc_info=True)

    def process(
        self,
        audio_file_path: str,
        language: str | None,
        task: Literal["transcribe", "translate"],
        timestamp_type: Literal["chunk", "word"],
        original_filename: str | None = None,
        progress_callback: ProgressCallback | None = None,
        # Other common parameters for all pipelines
    ) -> dict[str, Any]:
        """Template method defining the overall ASR algorithm skeleton.

        Returns:
            A dictionary containing the final transcription result.

        Raises:
            TranscriptionError: If the pipeline fails at any stage.
        """
        start_time = time.perf_counter()
        self.pipeline_id = str(uuid.uuid4())  # New ID for each run
        input_path = Path(audio_file_path)
        absolute_audio_path = input_path.resolve()

        self._notify_listeners(
            ProgressEvent(
                event_type="pipeline_start",
                pipeline_id=self.pipeline_id,
                file_path=str(absolute_audio_path),
                message=f"Pipeline started for {absolute_audio_path}",
            )
        )

        progress_cb = progress_callback or NoOpProgress()

        try:
            prepared_data = self._prepare_input(input_path)
            processed_result = self._execute_asr(
                prepared_data,
                language,
                task,
                timestamp_type,
                progress_cb,
            )
            final_result = self._postprocess_output(
                processed_result, absolute_audio_path, task, original_filename
            )

            saved_file_path = None
            if self.save_transcriptions:
                saved_file_path = self._save_result(
                    final_result, absolute_audio_path, task, original_filename
                )
                if saved_file_path:
                    final_result["output_file_path"] = (
                        saved_file_path  # Add saved path to result
                    )

            end_time = time.perf_counter()
            total_duration = round(end_time - start_time, 2)
            final_result["pipeline_runtime_seconds"] = total_duration
            # Potentially add pipeline config to final_result

            self._notify_listeners(
                ProgressEvent(
                    event_type="pipeline_complete",
                    pipeline_id=self.pipeline_id,
                    file_path=str(absolute_audio_path),
                    message=f"Pipeline completed in {total_duration}s.",
                    result=final_result,
                )
            )
            progress_cb.on_completed()
            return final_result
        except Exception as e:
            logger.error(
                "Error during pipeline execution for %s: %s",
                audio_file_path,
                e,
                exc_info=True,
            )
            try:
                progress_cb.on_error(str(e))
            except Exception:  # pragma: no cover - defensive
                pass

            self._notify_listeners(
                ProgressEvent(
                    event_type="pipeline_error",
                    pipeline_id=self.pipeline_id,
                    file_path=str(absolute_audio_path),
                    message=f"Pipeline failed: {str(e)}",
                )
            )
            progress_cb.on_completed()
            # Re-raise or handle as appropriate. For now, re-raise TranscriptionError.
            if not isinstance(e, TranscriptionError):
                # Format string for logger and exception message for clarity
                error_message = f"Pipeline failed for {audio_file_path}: {str(e)}"
                logger.error(
                    error_message, exc_info=True
                )  # exc_info will log the original exception
                raise TranscriptionError(f"Pipeline failed: {str(e)}") from e
            # If it's already a TranscriptionError, just log and re-raise
            logger.error(
                "Pipeline failed with TranscriptionError for %s: %s",
                audio_file_path,
                e,
                exc_info=True,
            )  # Log before re-raising
            raise

    @abstractmethod
    def _prepare_input(self, audio_file_path: Path) -> InputType:
        """Prepare audio input (e.g., load, chunk).

        Args:
            audio_file_path: Path to the input audio file.

        Returns:
            Prepared input used by `_execute_asr`.

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """

    @abstractmethod
    def _execute_asr(
        self,
        prepared_data: InputType,
        language: str | None,
        task: str,
        timestamp_type: str,
        progress_callback: ProgressCallback,
    ) -> dict[str, Any]:
        """Execute the core ASR task using the backend. Returns raw ASR output."""

    @abstractmethod
    def _postprocess_output(
        self,
        asr_output: dict[str, Any],
        audio_file_path: Path,
        task: str,
        original_filename: str | None = None,
    ) -> dict[str, Any]:
        """Post-process ASR output (for example, format and add metadata).

        Returns:
            The final result dictionary.
        """

    def _save_result(
        self,
        result: dict[str, Any],
        audio_file_path: Path,
        task: str,
        original_filename: str | None = None,
    ) -> str | None:
        """Saves the transcription result using the storage backend.

        Returns:
            The path to the saved file, or None if saving failed.
        """
        if not self.storage_backend:
            logger.warning("No storage backend configured, skipping save.")
            return None

        try:
            current_task_type = TaskType(task.lower())
        except ValueError:
            logger.warning(
                (
                    "_save_result: Invalid task type '%s'. Defaulting to "
                    "transcribe for filename."
                ),
                task,
            )
            current_task_type = TaskType.TRANSCRIBE  # Fallback for filename generation

        # Generate filename using the centralized utility.
        # Use original_filename if provided, otherwise use the actual file path
        filename_path = (
            original_filename if original_filename else str(audio_file_path.absolute())
        )

        # The extension is implicitly handled by JsonStorage, so we pass "json"
        # to the generator for completeness, though JsonStorage might re-append it.
        # Or, storage.save could be made to not append if filename already has one.
        # For now, this ensures the generator creates a complete name like
        # 'file_transcribe_time.json'.
        filename_str = self._filename_generator.create_filename(
            audio_path=filename_path,
            task=current_task_type,
            extension="json",  # JsonStorage typically appends .json, ensure consistency
        )

        save_path_base = self.output_dir / filename_str
        # Note: JsonStorage.save expects a base path and appends its own
        # extension. To use the fully generated name, we might need to adjust
        # JsonStorage or pass `save_path_base.stem` if JsonStorage always adds
        # '.json'. For now, we pass the full name, assuming JsonStorage can
        # handle it or needs adjustment. If JsonStorage strictly appends .json,
        # save_path_base should be `self.output_dir / filename_str.removesuffix(
        # '.json')` or `self.output_dir / Path(filename_str).stem`. This needs
        # to be harmonized with how `JsonStorage.save()` works. Let's assume for
        # now that `JsonStorage` is flexible or will be adapted.

        try:
            # Pass the full path (directory + filename_with_extension) to the
            # storage backend. The `task` argument for storage.save might be
            # redundant if already in filename.
            saved_path = self.storage_backend.save(result, save_path_base, task)
            if saved_path:
                logger.info("Result saved to %s", saved_path)
            return saved_path
        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
        ) as e:  # More specific exception types
            logger.error(
                "Failed to save result for %s: %s", audio_file_path, e, exc_info=True
            )
            return None


class WhisperPipeline(BasePipeline):
    """Whisper-specific pipeline implementation."""

    def _prepare_input(self, audio_file_path: Path) -> str:
        """Prepare input for the Whisper pipeline.

        For basic Whisper, the input is just the file path. This method ensures
        the file exists before proceeding.

        Returns:
            The audio file path as a string.

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        logger.info("Preparing input: %s", audio_file_path)
        if not audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        return str(audio_file_path)

    def _execute_asr(
        self,
        prepared_data: str,  # This is the audio_file_path from _prepare_input
        language: str | None,
        task: str,
        timestamp_type: str,
        progress_callback: ProgressCallback,
    ) -> dict[str, Any]:
        """Execute ASR for a single audio file, handling chunking internally.

        Returns:
            The raw ASR output dictionary.

        Raises:
            TranscriptionError: If the backend fails to process the audio.
        """
        logger.info(
            "Executing ASR for: %s, task: %s, lang: %s, timestamps: %s",
            prepared_data,
            task,
            language,
            timestamp_type,
        )
        # Determine return_timestamps_value for the backend based on timestamp_type
        if timestamp_type == "word":
            return_timestamps_value: bool | str = "word"
        elif timestamp_type == "chunk":
            return_timestamps_value = (
                True  # For Whisper, True implies chunk-level timestamps
            )
        else:  # Default or unknown
            return_timestamps_value = False

        progress_callback.on_audio_loading_started(prepared_data)
        converted_path = audio_conversion.ensure_wav(prepared_data)
        progress_callback.on_audio_loading_finished(duration_sec=None)

        # Split the audio into chunks so we can provide deterministic progress
        # updates to observers (e.g. Gradio's progress bar). ``split_audio``
        # returns a list with the original path if chunking is unnecessary.
        chunk_data = audio_processing.split_audio(
            converted_path,
            chunk_duration=float(self.asr_backend.config.chunk_length),
            chunk_overlap=0.0,
        )
        total_chunks = len(chunk_data)
        # Store tuples of (result, start_time) for the merge step
        chunk_results: list[tuple[dict[str, Any], float]] = []

        if total_chunks == 0:
            raise TranscriptionError("No audio chunks produced for transcription.")

        progress_callback.on_chunking_started(total_chunks)
        progress_callback.on_inference_started(total_chunks)

        # Suppress premature completion events from the backend while we
        # orchestrate chunk-level progress here.
        class _ProgressProxy:
            def __init__(self, delegate: ProgressCallback) -> None:
                self._delegate = delegate

            def __getattr__(self, item: str) -> object:
                return cast(object, getattr(self._delegate, item))

            def on_completed(self) -> None:
                return

        progress_proxy = _ProgressProxy(progress_callback)

        try:
            for idx, (chunk_path, chunk_start_time) in enumerate(chunk_data, start=1):
                self._notify_listeners(
                    ProgressEvent(
                        event_type="chunk_start",
                        pipeline_id=self.pipeline_id,
                        file_path=prepared_data,
                        chunk_num=idx,
                        total_chunks=total_chunks,
                        message=(
                            f"Processing chunk {idx}/{total_chunks} for {prepared_data}"
                        ),
                    )
                )

                asr_raw_result = self.asr_backend.process_audio(
                    audio_file_path=chunk_path,
                    language=language,
                    task=task,
                    return_timestamps_value=return_timestamps_value,
                    progress_cb=progress_proxy,
                )

                self._notify_listeners(
                    ProgressEvent(
                        event_type="chunk_complete",
                        pipeline_id=self.pipeline_id,
                        file_path=prepared_data,
                        chunk_num=idx,
                        total_chunks=total_chunks,
                        result=asr_raw_result,
                        message=(
                            f"Completed chunk {idx}/{total_chunks} for {prepared_data}"
                        ),
                    )
                )
                chunk_results.append((asr_raw_result, chunk_start_time))
                completed_index = idx - 1
                try:
                    progress_callback.on_chunk_done(completed_index)
                    progress_callback.on_inference_batch_done(completed_index)
                except Exception:  # pragma: no cover - defensive
                    pass
        finally:
            cleanup_paths: list[str] = []
            if total_chunks > 1:
                cleanup_paths.extend([cd[0] for cd in chunk_data])
            if converted_path != prepared_data and converted_path not in cleanup_paths:
                cleanup_paths.append(converted_path)
            if cleanup_paths:
                file_utils.cleanup_temp_files(cleanup_paths)

        if not chunk_results:
            # This case should ideally not be hit if split_audio always returns
            # at least one item, but as a safeguard:
            return {"text": "", "chunks": []}

        if total_chunks > 1:
            combined = audio_results.merge_chunk_results(chunk_results)
        else:
            combined = chunk_results[0][0]

        progress_callback.on_completed()
        return combined

    def _postprocess_output(
        self,
        asr_output: dict[str, Any],
        audio_file_path: Path,
        task: str,
        original_filename: str | None = None,
    ) -> dict[str, Any]:
        """Post-processes the raw ASR output for Whisper.

        Returns:
            The post-processed result dictionary with added metadata.
        """
        logger.info("Postprocessing ASR output for: %s", audio_file_path)
        # Example: Add original file name, task, and a processing timestamp
        processed_result = asr_output.copy()
        # CRITICAL FIX: Store absolute path for audio file to ensure
        # post-processing steps (like stabilization) can locate the file
        # regardless of working directory changes.
        # Use original_filename if provided, otherwise use the absolute file path
        processed_result["original_file"] = (
            original_filename if original_filename else str(audio_file_path.resolve())
        )
        processed_result["task_type"] = task
        processed_result["processed_at"] = datetime.now(timezone.utc).isoformat()
        # Potentially refine chunk timestamps if needed, or add other metadata
        return processed_result
