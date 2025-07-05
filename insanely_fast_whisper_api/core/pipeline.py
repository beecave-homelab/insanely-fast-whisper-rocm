"""ASR pipeline definition, including base classes and concrete implementations."""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from insanely_fast_whisper_api.core.asr_backend import ASRBackend
from insanely_fast_whisper_api.core.errors import TranscriptionError
from insanely_fast_whisper_api.core.storage import BaseStorage, StorageFactory
from insanely_fast_whisper_api.utils.filename_generator import (
    FilenameGenerator,
    StandardFilenameStrategy,
    TaskType,
)

logger = logging.getLogger(__name__)


@dataclass
class ProgressEvent:
    """Dataclass to represent a progress event."""

    event_type: str  # e.g., "chunk_start", "chunk_complete", "pipeline_complete"
    pipeline_id: str
    file_path: str
    chunk_num: Optional[int] = None
    total_chunks: Optional[int] = None
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None  # For chunk_complete or pipeline_complete


ProgressCallback = Callable[[ProgressEvent], None]


class BasePipeline(ABC):
    """Base class for ASR pipelines using the Template Method pattern."""

    def __init__(
        self,
        asr_backend: ASRBackend,
        storage_backend: Optional[BaseStorage] = None,
        save_transcriptions: bool = True,
        output_dir: str = "transcripts",
    ):
        self.asr_backend = asr_backend
        self.storage_backend = (
            storage_backend if storage_backend else StorageFactory.create()
        )
        self.save_transcriptions = save_transcriptions
        self.output_dir = Path(output_dir)
        self._listeners: List[ProgressCallback] = []
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
        language: Optional[str],
        task: Literal["transcribe", "translate"],
        timestamp_type: Literal["chunk", "word"],
        original_filename: Optional[str] = None,
        # Other common parameters for all pipelines
    ) -> Dict[str, Any]:
        """Template method defining the overall ASR algorithm skeleton."""
        start_time = time.perf_counter()
        self.pipeline_id = str(uuid.uuid4())  # New ID for each run
        absolute_audio_path = Path(audio_file_path).resolve()

        self._notify_listeners(
            ProgressEvent(
                event_type="pipeline_start",
                pipeline_id=self.pipeline_id,
                file_path=str(absolute_audio_path),
                message=f"Pipeline started for {absolute_audio_path}",
            )
        )

        try:
            prepared_data = self._prepare_input(absolute_audio_path)
            processed_result = self._execute_asr(
                prepared_data, language, task, timestamp_type
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
            return final_result
        except Exception as e:
            logger.error(
                "Error during pipeline execution for %s: %s",
                audio_file_path,
                e,
                exc_info=True,
            )
            self._notify_listeners(
                ProgressEvent(
                    event_type="pipeline_error",
                    pipeline_id=self.pipeline_id,
                    file_path=str(absolute_audio_path),
                    message=f"Pipeline failed: {str(e)}",
                )
            )
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
    def _prepare_input(self, audio_file_path: Path) -> Any:
        """Prepare audio input (e.g., load, chunk). Returns data for _execute_asr."""

    @abstractmethod
    def _execute_asr(
        self,
        prepared_data: Any,
        language: Optional[str],
        task: str,
        timestamp_type: str,
    ) -> Dict[str, Any]:
        """Execute the core ASR task using the backend. Returns raw ASR output."""

    @abstractmethod
    def _postprocess_output(
        self,
        asr_output: Dict[str, Any],
        audio_file_path: Path,
        task: str,
        original_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Post-process ASR output (e.g., format, add metadata). Returns final result dict."""

    def _save_result(
        self,
        result: Dict[str, Any],
        audio_file_path: Path,
        task: str,
        original_filename: Optional[str] = None,
    ) -> Optional[str]:
        """Saves the transcription result using the storage backend."""
        if not self.storage_backend:
            logger.warning("No storage backend configured, skipping save.")
            return None

        try:
            current_task_type = TaskType(task.lower())
        except ValueError:
            logger.warning(
                "_save_result: Invalid task type '%s'. Defaulting to transcribe for filename.",
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
        # Note: JsonStorage.save expects a base path and appends its own extension.
        # To use the fully generated name including extension, we might need to adjust JsonStorage
        # or pass `save_path_base.stem` if JsonStorage *always* adds '.json'.
        # For now, we pass the full name, assuming JsonStorage can handle it or needs adjustment.
        # If JsonStorage strictly appends .json, save_path_base should be
        # `self.output_dir / filename_str.removesuffix('.json')`
        # or `self.output_dir / Path(filename_str).stem`
        # This needs to be harmonized with how `JsonStorage.save()` works.
        # Let's assume for now that `JsonStorage` is flexible or will be adapted.

        try:
            # Pass the full path (directory + filename_with_extension) to the storage backend.
            # The `task` argument for storage.save might be redundant if already in filename.
            saved_path = self.storage_backend.save(result, save_path_base, task)
            if saved_path:
                logger.info("Result saved to %s", saved_path)
            return saved_path
        except (
            OSError,
            IOError,
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

    def _prepare_input(self, audio_file_path: Path) -> Any:
        """For basic Whisper, input is just the file path.
        Chunking would happen here or in _execute_asr."""
        # This could be extended for chunking logic if not handled by the backend strategy
        logger.info("Preparing input: %s", audio_file_path)
        if not audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        # For now, prepared_data is just the path. If pipeline-level chunking
        # is implemented, this would return a list of chunk file paths or similar.
        return str(audio_file_path)

    def _execute_asr(
        self,
        prepared_data: str,  # This is the audio_file_path from _prepare_input
        language: Optional[str],
        task: str,
        timestamp_type: str,
    ) -> Dict[str, Any]:
        """Executes ASR for a single audio file (or a single chunk
        if chunking is done by this method)."""
        logger.info(
            "Executing ASR for: %s, task: %s, lang: %s, timestamps: %s",
            prepared_data,
            task,
            language,
            timestamp_type,
        )
        # Determine return_timestamps_value for the backend based on timestamp_type
        if timestamp_type == "word":
            return_timestamps_value: Union[bool, str] = "word"
        elif timestamp_type == "chunk":
            return_timestamps_value = (
                True  # For Whisper, True implies chunk-level timestamps
            )
        else:  # Default or unknown
            return_timestamps_value = False

        # This method assumes the asr_backend can handle a single file path.
        # If pipeline-level chunking was implemented in _prepare_input,
        # this method would loop through chunks and aggregate results.
        self._notify_listeners(
            ProgressEvent(
                event_type="chunk_start",  # Assuming non-chunked or backend handles chunks
                pipeline_id=self.pipeline_id,
                file_path=prepared_data,
                chunk_num=1,  # Placeholder
                total_chunks=1,  # Placeholder
                message=(  # Reformatted for line length
                    f"Processing chunk 1/1 for {prepared_data}"
                ),
            )
        )

        asr_raw_result = self.asr_backend.process_audio(
            audio_file_path=prepared_data,
            language=language,
            task=task,
            return_timestamps_value=return_timestamps_value,
        )

        self._notify_listeners(
            ProgressEvent(
                event_type="chunk_complete",  # Assuming non-chunked or backend handles chunks
                pipeline_id=self.pipeline_id,
                file_path=prepared_data,
                chunk_num=1,  # Placeholder
                total_chunks=1,  # Placeholder
                result=asr_raw_result,  # Added result
                message=(  # Reformatted for line length
                    f"Completed chunk 1/1 for {prepared_data}"
                ),
            )
        )
        return asr_raw_result

    def _postprocess_output(
        self,
        asr_output: Dict[str, Any],
        audio_file_path: Path,
        task: str,
        original_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Post-processes the raw ASR output for Whisper."""
        logger.info("Postprocessing ASR output for: %s", audio_file_path)
        # Example: Add original file name, task, and a processing timestamp
        processed_result = asr_output.copy()
        # Use original_filename if provided, otherwise use the actual file path
        processed_result["original_file"] = (
            original_filename if original_filename else str(audio_file_path)
        )
        processed_result["task_type"] = task
        processed_result["processed_at"] = datetime.now(timezone.utc).isoformat()
        # Potentially refine chunk timestamps if needed, or add other metadata
        return processed_result
