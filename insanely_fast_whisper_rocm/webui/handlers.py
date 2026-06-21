"""Handler functions for Insanely Fast Whisper API WebUI.

This module contains the core logic for handling transcription requests
and exporting results in the WebUI. It serves as an intermediary between
the UI components and the ASR pipeline.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import gradio as gr

from insanely_fast_whisper_rocm.audio.processing import extract_audio_from_video
from insanely_fast_whisper_rocm.core.asr_backend import HuggingFaceBackendConfig
from insanely_fast_whisper_rocm.core.cancellation import CancellationToken
from insanely_fast_whisper_rocm.core.errors import (
    DiarizationError,
    OutOfMemoryError,
    TranscriptionCancelledError,
    TranscriptionError,
)
from insanely_fast_whisper_rocm.core.formatters import FORMATTERS
from insanely_fast_whisper_rocm.core.integrations.stable_ts import stabilize_timestamps
from insanely_fast_whisper_rocm.core.orchestrator import create_orchestrator
from insanely_fast_whisper_rocm.core.progress import ProgressCallback
from insanely_fast_whisper_rocm.utils import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEMUCS,
    DEFAULT_DEVICE,
    DEFAULT_DIARIZATION_DEVICE,
    DEFAULT_DIARIZE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_STABILIZE,
    DEFAULT_TIMESTAMP_TYPE,
    DEFAULT_TRANSCRIPTS_DIR,
    DEFAULT_VAD,
    DEFAULT_VAD_THRESHOLD,
    constants,
)
from insanely_fast_whisper_rocm.utils.filename_generator import (
    FilenameGenerator,
    StandardFilenameStrategy,
    TaskType,
)
from insanely_fast_whisper_rocm.webui.zip_creator import (
    BatchZipBuilder,
    ZipConfiguration,
)

# Configure logger
logger = logging.getLogger("insanely_fast_whisper_rocm.webui.handlers")

# Ensure default transcripts dir exists for WebUI direct saves (if any outside pipeline)
Path(constants.DEFAULT_TRANSCRIPTS_DIR).mkdir(parents=True, exist_ok=True)

# Global instance of FilenameGenerator with standard strategy for WebUI
# This can be configured or made more flexible if needed later.
STANDARD_FILENAME_STRATEGY = StandardFilenameStrategy()
WEBUI_FILENAME_GENERATOR = FilenameGenerator(strategy=STANDARD_FILENAME_STRATEGY)


@dataclass
class TranscriptionConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for the transcription process."""

    model: str = DEFAULT_MODEL
    device: str = DEFAULT_DEVICE
    batch_size: int = DEFAULT_BATCH_SIZE
    timestamp_type: Literal["chunk", "word"] = DEFAULT_TIMESTAMP_TYPE
    language: str = DEFAULT_LANGUAGE
    task: Literal["transcribe", "translate"] = "transcribe"
    dtype: str = "float16"
    chunk_length: int = 30
    chunk_duration: float | None = None
    chunk_overlap: float | None = None
    # Stabilization options
    stabilize: bool = DEFAULT_STABILIZE
    demucs: bool = DEFAULT_DEMUCS
    vad: bool = DEFAULT_VAD
    vad_threshold: float = DEFAULT_VAD_THRESHOLD
    # Diarization options
    diarize: bool = DEFAULT_DIARIZE
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    diarization_device: str = DEFAULT_DIARIZATION_DEVICE


@dataclass
class FileHandlingConfig:
    """Configuration for file handling."""

    save_transcriptions: bool = True
    temp_uploads_dir: str = DEFAULT_TRANSCRIPTS_DIR


def _prepare_temp_downloadable_file(
    raw_data: dict[str, Any],
    format_type: str,
    original_audio_stem: str,
    temp_dir: Path,
    task: TaskType,
) -> str:
    """Generate and persist a temporary downloadable file for the WebUI.

    Generates content for TXT, SRT, or JSON, saves it to a temporary file, and
    returns the file path.

    Args:
        raw_data: The raw transcription result data.
        format_type: Target output format, e.g. "txt", "srt", or "json".
        original_audio_stem: Stem of the original audio file name.
        temp_dir: Directory to write the temporary file to.
        task: The task used for filename generation.

    Returns:
        str: Absolute path to the generated temporary file.

    Raises:
        ValueError: If a formatter for the given format is not available.
        OSError: If writing the temporary file fails.
    """
    formatter = FORMATTERS.get(format_type)
    if not formatter:
        raise ValueError(f"No formatter available for type: {format_type}")

    content = formatter.format(raw_data)

    # Use the global WEBUI_FILENAME_GENERATOR for consistent naming
    # Filename will be like: audio_stem_task_timestamp.format
    # We need a unique name, timestamp is good.
    # The generator itself handles the timestamp.
    # Generator expects a path-like string for stem extraction
    filename = WEBUI_FILENAME_GENERATOR.create_filename(
        audio_path=original_audio_stem,
        task=task,
        extension=format_type,
    )

    temp_file_path = (
        temp_dir / f"temp_dl_{filename}"
    )  # Prefix to avoid clashes if needed
    temp_file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        abs_path = str(temp_file_path.resolve())
        logger.info("Created temporary download file: %s", abs_path)
        return abs_path
    except OSError as e:
        logger.error(
            "Failed to create temporary download file %s: %s", temp_file_path, e
        )
        raise


def _to_gradio_file_value(path: str | Path | None) -> str | None:
    """Convert a filesystem path to a Gradio-safe absolute file path.

    Gradio's File component copies files by absolute path.
    Relative paths cause the copy step to hang silently.

    Args:
        path: Path returned by the pipeline or export builder.

    Returns:
        Absolute string path for Gradio file/download components, or None.
    """
    if not path:
        return None
    return str(Path(path).resolve())


def _is_stabilization_corrupt(segments: list[dict]) -> bool:
    """Check if the stabilized segments appear to be corrupt.

    Returns:
        bool: True if the segments are likely corrupt, False otherwise.
    """
    if not segments or len(segments) < 2:
        return False

    # Heuristic: If more than 50% of segments have identical timestamps,
    # it's likely a sign of timestamp collapse.
    first_timestamp = (segments[0].get("start"), segments[0].get("end"))
    identical_count = sum(
        1 for seg in segments if (seg.get("start"), seg.get("end")) == first_timestamp
    )

    return (identical_count / len(segments)) > 0.5


def _build_ui_json_preview(
    raw_result: dict[str, Any],
    *,
    json_file_path: str | None,
) -> str:
    """Build a JSON preview string for the WebUI export tab.

    Args:
        raw_result: Raw transcription result from the pipeline.
        json_file_path: Path to the saved JSON file on disk.

    Returns:
        Pretty-formatted JSON text for UI display.
    """
    if json_file_path:
        json_path = Path(json_file_path)
        try:
            return json_path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Failed to read JSON export preview from %s", json_path)

    return json.dumps(raw_result, indent=2, ensure_ascii=False)


def _review_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return transcript segments used by the speaker review UI.

    Args:
        result: Raw transcription result.

    Returns:
        Segment dictionaries from ``segments`` or ``chunks``.
    """
    segments = result.get("segments") or result.get("chunks") or []
    return segments if isinstance(segments, list) else []


def build_speaker_review_state(result: dict[str, Any] | None) -> dict[str, Any]:
    """Build editable speaker-review metadata for a transcription result.

    Args:
        result: Raw transcription result with optional diarization labels.

    Returns:
        Speaker map, segment choices, and review queue metadata.
    """
    if not isinstance(result, dict):
        return {
            "speaker_map": {},
            "segment_choices": [],
            "review_queue": [],
            "overlap_count": 0,
        }

    speakers = {
        str(segment.get("speaker"))
        for segment in _review_segments(result)
        if segment.get("speaker")
    }
    speaker_map = {
        speaker: result.get("speaker_names", {}).get(speaker, speaker)
        for speaker in sorted(speakers)
    }
    segment_choices = [
        _format_review_segment_choice(index, segment, speaker_map)
        for index, segment in enumerate(_review_segments(result))
    ]
    review_queue = [
        index
        for index, segment in enumerate(_review_segments(result))
        if _is_low_confidence_segment(segment)
    ]
    overlap_count = sum(
        1 for segment in _review_segments(result) if segment.get("overlap")
    )
    return {
        "speaker_map": speaker_map,
        "segment_choices": segment_choices,
        "review_queue": review_queue,
        "overlap_count": overlap_count,
    }


def build_speaker_review_summary(result: dict[str, Any] | None) -> str:
    """Return a readable speaker-review summary for the WebUI.

    Args:
        result: Raw transcription result.

    Returns:
        Multiline summary for the speaker review panel.
    """
    state = build_speaker_review_state(result)
    speaker_map = state["speaker_map"]
    if not speaker_map:
        return "No diarized speaker labels available yet."

    lines = [f"{speaker} -> {name}" for speaker, name in speaker_map.items()]
    review_count = len(state["review_queue"])
    if review_count:
        lines.append(f"{review_count} low-confidence turn(s) need review.")
    if state["overlap_count"]:
        lines.append(f"{state['overlap_count']} overlapping turn(s) marked.")
    return "\n".join(lines)


def apply_speaker_rename(
    result: dict[str, Any] | None,
    speaker_id: str,
    speaker_name: str,
) -> tuple[str, str, dict[str, Any] | None]:
    """Rename a diarization speaker for readable transcript exports.

    Args:
        result: Raw transcription result.
        speaker_id: Internal diarization speaker ID.
        speaker_name: Human-readable speaker name.

    Returns:
        Updated transcript preview, JSON preview, and raw result state.
    """
    if not isinstance(result, dict) or not speaker_id:
        return "", "{}", result

    updated = _copy_result_with_speaker_names(result)
    updated["speaker_names"][speaker_id] = speaker_name.strip() or speaker_id
    _apply_display_speakers(updated)
    return _review_outputs(updated)


def retag_transcript_segment(
    result: dict[str, Any] | None,
    segment_choice: str,
    speaker_id: str,
) -> tuple[str, str, dict[str, Any] | None]:
    """Assign one transcript segment to a different speaker.

    Args:
        result: Raw transcription result.
        segment_choice: Segment selector label generated by the review UI.
        speaker_id: Target internal speaker ID.

    Returns:
        Updated transcript preview, JSON preview, and raw result state.
    """
    if not isinstance(result, dict) or not speaker_id:
        return "", "{}", result

    index = _parse_review_segment_index(segment_choice)
    updated = _copy_result_with_speaker_names(result)
    for key in ("segments", "chunks"):
        values = updated.get(key)
        if isinstance(values, list) and 0 <= index < len(values):
            values[index] = {**values[index], "speaker": speaker_id}
    _apply_display_speakers(updated)
    return _review_outputs(updated)


def split_review_segment_speaker(
    result: dict[str, Any] | None,
    segment_choice: str,
    speaker_name: str,
) -> tuple[str, str, dict[str, Any] | None]:
    """Split one transcript segment into a new speaker label.

    Args:
        result: Raw transcription result.
        segment_choice: Segment selector label generated by the review UI.
        speaker_name: Optional readable name for the new speaker.

    Returns:
        Updated transcript preview, JSON preview, and raw result state.
    """
    if not isinstance(result, dict):
        return "", "{}", result

    index = _parse_review_segment_index(segment_choice)
    updated = _copy_result_with_speaker_names(result)
    new_speaker = _next_speaker_id(updated)
    for key in ("segments", "chunks"):
        values = updated.get(key)
        if isinstance(values, list) and 0 <= index < len(values):
            values[index] = {**values[index], "speaker": new_speaker}

    updated["speaker_names"][new_speaker] = speaker_name.strip() or new_speaker
    _apply_display_speakers(updated)
    return _review_outputs(updated)


def merge_review_speakers(
    result: dict[str, Any] | None,
    source_speaker: str,
    target_speaker: str,
) -> tuple[str, str, dict[str, Any] | None]:
    """Merge one speaker label into another across transcript turns.

    Args:
        result: Raw transcription result.
        source_speaker: Internal speaker ID to replace.
        target_speaker: Internal speaker ID to keep.

    Returns:
        Updated transcript preview, JSON preview, and raw result state.
    """
    if not isinstance(result, dict) or not source_speaker or not target_speaker:
        return "", "{}", result

    updated = _copy_result_with_speaker_names(result)
    for key in ("segments", "chunks"):
        values = updated.get(key)
        if not isinstance(values, list):
            continue
        updated[key] = [
            {
                **segment,
                "speaker": target_speaker
                if segment.get("speaker") == source_speaker
                else segment.get("speaker"),
            }
            for segment in values
        ]
    updated["speaker_names"].pop(source_speaker, None)
    _apply_display_speakers(updated)
    return _review_outputs(updated)


def refresh_review_downloads(
    result: dict[str, Any] | None,
    temp_uploads_dir: str,
    task: str,
) -> tuple[Any, Any, Any, Any]:
    """Regenerate downloadable files from the reviewed transcript state.

    Args:
        result: Reviewed transcription result.
        temp_uploads_dir: Directory used for WebUI exports.
        task: Current transcription task.

    Returns:
        Updates for ZIP, TXT, SRT, and JSON download components.
    """
    hidden_update = gr.update(visible=False, value=None)
    if not isinstance(result, dict):
        return hidden_update, hidden_update, hidden_update, hidden_update

    temp_dir = Path(temp_uploads_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    stem = _review_export_stem(result)
    task_type = TaskType(task)
    txt_update = srt_update = json_update = hidden_update

    try:
        txt_update = gr.update(
            value=_prepare_temp_downloadable_file(
                result, "txt", stem, temp_dir, task_type
            ),
            visible=True,
        )
    except (OSError, ValueError) as exc:
        logger.warning("Failed to refresh reviewed TXT download: %s", exc)

    try:
        srt_update = gr.update(
            value=_prepare_temp_downloadable_file(
                result, "srt", stem, temp_dir, task_type
            ),
            visible=True,
        )
    except (OSError, ValueError) as exc:
        logger.warning("Failed to refresh reviewed SRT download: %s", exc)

    try:
        json_update = gr.update(
            value=_prepare_temp_downloadable_file(
                result, "json", stem, temp_dir, task_type
            ),
            visible=True,
        )
    except (OSError, ValueError) as exc:
        logger.warning("Failed to refresh reviewed JSON download: %s", exc)

    try:
        zip_config = ZipConfiguration(
            temp_dir=str(temp_dir),
            organize_by_format=False,
            include_summary=False,
        )
        zip_builder = BatchZipBuilder(config=zip_config)
        zip_builder.create(
            filename=f"{stem}_reviewed_ALL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )
        zip_builder.add_batch_files(
            {f"{stem}.wav": result}, formats=["txt", "srt", "json"]
        )
        zip_path, _ = zip_builder.build()
        zip_update = gr.update(value=_to_gradio_file_value(zip_path), visible=True)
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        FileNotFoundError,
        zipfile.BadZipFile,
    ) as exc:
        logger.warning("Failed to refresh reviewed ZIP download: %s", exc)
        zip_update = hidden_update

    return zip_update, txt_update, srt_update, json_update


def _format_review_segment_choice(
    index: int,
    segment: dict[str, Any],
    speaker_map: dict[str, str],
) -> str:
    """Return a stable segment selector label."""
    start = segment.get("start")
    end = segment.get("end")
    speaker = segment.get("speaker") or "Unassigned"
    label = speaker_map.get(str(speaker), str(speaker))
    text = str(segment.get("text", "")).strip().replace("\n", " ")
    badges = []
    if _is_low_confidence_segment(segment):
        badges.append("low confidence")
    if segment.get("overlap"):
        badges.append("overlap")
    marker = f" ({', '.join(badges)})" if badges else ""
    return f"{index}: [{start}-{end}] {label}{marker}: {text[:80]}"


def _parse_review_segment_index(segment_choice: str) -> int:
    """Parse the selected review segment index.

    Returns:
        Selected segment index, or ``-1`` when parsing fails.
    """
    try:
        return int(str(segment_choice).split(":", 1)[0])
    except (TypeError, ValueError):
        return -1


def _is_low_confidence_segment(segment: dict[str, Any]) -> bool:
    """Return whether a segment should enter the review queue."""
    confidence = segment.get("speaker_confidence")
    if confidence is None:
        confidence = segment.get("confidence")
    try:
        return float(confidence) < 0.65
    except (TypeError, ValueError):
        return False


def _copy_result_with_speaker_names(result: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow editable copy of a transcription result."""
    copied = {
        **result,
        "speaker_names": dict(result.get("speaker_names") or {}),
    }
    for key in ("segments", "chunks"):
        values = result.get(key)
        if isinstance(values, list):
            copied[key] = [dict(segment) for segment in values]
    return copied


def _next_speaker_id(result: dict[str, Any]) -> str:
    """Return the next available ``SPEAKER_NN`` identifier."""
    used = {
        str(segment.get("speaker"))
        for segment in _review_segments(result)
        if segment.get("speaker")
    }
    index = 0
    while True:
        candidate = f"SPEAKER_{index:02d}"
        if candidate not in used:
            return candidate
        index += 1


def _review_export_stem(result: dict[str, Any]) -> str:
    """Return a stable filename stem for reviewed exports."""
    audio_path = result.get("audio_file_path")
    if isinstance(audio_path, str) and audio_path:
        return Path(audio_path).stem

    output_path = result.get("output_file_path")
    if isinstance(output_path, str) and output_path:
        return Path(output_path).stem

    return "reviewed_transcript"


def _apply_display_speakers(result: dict[str, Any]) -> None:
    """Attach readable speaker names while preserving internal IDs."""
    speaker_names = result.get("speaker_names") or {}
    for key in ("segments", "chunks"):
        values = result.get(key)
        if not isinstance(values, list):
            continue
        for segment in values:
            speaker = segment.get("speaker")
            if speaker:
                segment["speaker_display"] = speaker_names.get(speaker, speaker)


def _review_outputs(
    result: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Return transcript, JSON, and state outputs for review edits."""
    display_txt_formatter = FORMATTERS.get("txt")
    transcription = (
        display_txt_formatter.format(result)
        if display_txt_formatter
        else result.get("text", "")
    )
    return (
        transcription,
        json.dumps(result, indent=2, ensure_ascii=False),
        result,
    )


def transcribe(
    audio_file_path: str,
    config: TranscriptionConfig,
    file_config: FileHandlingConfig,
    progress_tracker_instance: gr.Progress | None = None,
    current_file_idx: int = 0,
    total_files_for_session: int = 1,
) -> dict[str, Any]:
    """Transcribe an audio file using the ASRPipeline.

    Args:
        audio_file_path: Path to the input audio file
        config: Transcription configuration object
        file_config: File handling configuration object
        progress_tracker_instance: Optional Gradio progress tracker for UI updates.
        current_file_idx: 0-based index of the current file being processed.
        total_files_for_session: Total number of files in the current batch.

    Returns:
        Dictionary containing the transcription results and metadata

    Raises:
        TranscriptionError: If the transcription process fails
        TranscriptionCancelledError: If the transcription is cancelled by user
    """
    try:
        logger.info(
            "Starting transcription for file: %s (File %d/%d)",
            audio_file_path,
            current_file_idx + 1,
            total_files_for_session,
        )
        original_file_name_for_desc = Path(audio_file_path).name

        cancellation_token = CancellationToken()

        def _ensure_not_cancelled() -> None:
            """Raise an exception if the transcription has been cancelled."""
            if progress_tracker_instance is not None and getattr(
                progress_tracker_instance, "cancelled", False
            ):
                cancellation_token.cancel()
            cancellation_token.raise_if_cancelled()

        _ensure_not_cancelled()

        # --- Video detection & audio extraction ---
        temp_files: list[str] = []
        if Path(audio_file_path).suffix.lower() in constants.SUPPORTED_VIDEO_FORMATS:
            logger.info("Detected video input – extracting audio track…")
            try:
                extracted_path = extract_audio_from_video(audio_file_path)
                temp_files.append(extracted_path)
                audio_file_path = extracted_path  # Use extracted WAV for processing
                logger.info("Audio extracted to temporary file: %s", extracted_path)
            except RuntimeError as conv_err:
                logger.error("Video conversion failed: %s", conv_err)
                raise TranscriptionError(str(conv_err)) from conv_err

        _ensure_not_cancelled()

        # Initial progress update for this file's segment
        base_progress = current_file_idx / total_files_for_session
        if progress_tracker_instance is not None:
            progress_tracker_instance(
                base_progress,
                desc=(
                    f"Starting file {current_file_idx + 1}/{total_files_for_session}: "
                    f"{original_file_name_for_desc}"
                ),
            )

        _ensure_not_cancelled()

        # Build backend config and use the orchestrator for transcription
        backend_config = HuggingFaceBackendConfig(
            model_name=config.model,
            device=config.device,
            dtype=config.dtype,
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
            progress_group_size=constants.DEFAULT_PROGRESS_GROUP_SIZE,
        )
        result: dict[str, Any]

        _ensure_not_cancelled()

        orchestrator = create_orchestrator()

        def _warning_callback(message: str) -> None:
            """Log orchestrator recovery warnings and update UI.

            Args:
                message: The warning message from the orchestrator.
            """
            logger.info("Orchestrator warning: %s", message)
            if progress_tracker_instance is not None:
                if message.startswith("Attempt "):
                    progress_tracker_instance(None, desc=message)
                else:
                    progress_tracker_instance(None, desc=f"⚠️ {message}")
            if "falling back to cpu" in message.lower():
                logger.info("CPU fallback decision made by orchestrator")

        class WebUIProgressCallback(ProgressCallback):
            """Progress callback implementation for Gradio WebUI updates.

            This callback translates pipeline progress events into Gradio
            progress tracker updates for real-time UI feedback.
            """

            def __init__(
                self,
                tracker: gr.Progress,
                base: float,
                total: int,
                idx: int,
                name: str,
                cancel_token: CancellationToken,
            ) -> None:
                """Initialize the WebUI progress callback.

                Args:
                    tracker: Gradio progress tracker instance.
                    base: Base progress value for this file in the batch.
                    total: Total number of files being processed.
                    idx: Index of the current file being processed.
                    name: Name of the file being processed.
                    cancel_token: Cancellation token for cooperative cancellation.
                """
                self.tracker = tracker
                self.base = base
                self.total = total
                self.idx = idx
                self.name = name
                self.cancel_token = cancel_token
                self._total_chunks: int | None = None

            def _update(self, fraction: float | None, message: str) -> None:
                """Update the Gradio progress tracker.

                Args:
                    fraction: Optional progress fraction (0.0-1.0).
                    message: Progress message to display.
                """
                if self.cancel_token.cancelled:
                    return
                if getattr(self.tracker, "cancelled", False):
                    self.cancel_token.cancel()
                    return

                desc = f"{message} ({self.name})"
                if fraction is not None:
                    overall = self.base + (fraction / self.total)
                    self.tracker(overall, desc=desc)
                else:
                    self.tracker(None, desc=desc)

            def on_model_load_started(self) -> None:
                """Handle model load start event."""
                self._update(None, "Loading model...")

            def on_model_load_finished(self) -> None:
                """Handle model load complete event."""
                self._update(None, "Model loaded")

            def on_audio_loading_started(self, path: str) -> None:  # noqa: ARG002
                """Handle audio loading start event.

                Args:
                    path: Path to the audio file being loaded.
                """
                self._update(None, "Preparing audio...")

            def on_audio_loading_finished(self, duration_sec: float | None) -> None:  # noqa: ARG002
                """Handle audio loading complete event.

                Args:
                    duration_sec: Duration of the loaded audio in seconds.
                """
                self._update(None, "Audio ready")

            def on_chunking_started(self, total_chunks: int | None) -> None:
                """Handle audio chunking start event.

                Args:
                    total_chunks: Total number of audio chunks to process.
                """
                self._total_chunks = total_chunks
                self._update(0.0, "Starting transcription...")

            def on_chunk_done(self, index: int) -> None:
                """Handle chunk processing complete event.

                Args:
                    index: Index of the completed chunk.
                """
                if not self._total_chunks:
                    self._update(None, "Transcribing...")
                    return

                chunk_num = index + 1
                fraction = min(chunk_num / self._total_chunks, 1.0)
                self._update(
                    fraction,
                    f"Processing chunk {chunk_num}/{self._total_chunks}",
                )

            def on_inference_started(self, total_batches: int | None) -> None:  # noqa: ARG002
                """Handle inference start event.

                Args:
                    total_batches: Total number of inference batches.
                """
                return

            def on_inference_batch_done(self, index: int) -> None:  # noqa: ARG002
                """Handle inference batch complete event.

                Args:
                    index: Index of the completed batch.
                """
                return

            def on_postprocess_started(self, name: str) -> None:
                """Handle post-processing start event.

                Args:
                    name: Name of the post-processing step.
                """
                self._update(None, f"Post-processing: {name}")

            def on_postprocess_finished(self, name: str) -> None:
                """Handle post-processing complete event.

                Args:
                    name: Name of the completed post-processing step.
                """
                self._update(None, f"Post-processing done: {name}")

            def on_export_started(self, total_items: int) -> None:
                """Handle export start event.

                Args:
                    total_items: Total number of items to export.
                """
                self._update(None, f"Exporting {total_items} file(s)...")

            def on_export_item_done(self, index: int, label: str) -> None:
                """Handle export item complete event.

                Args:
                    index: Index of the completed export item.
                    label: Label of the exported item.
                """
                self._update(None, f"Exported: {label} ({index + 1})")

            def on_completed(self) -> None:
                """Handle transcription complete event."""
                self._update(1.0, "Transcription complete")

            def on_error(self, message: str) -> None:
                """Handle error event.

                Args:
                    message: Error message to display.
                """
                self._update(None, f"Error: {message}")

        webui_cb = None
        if progress_tracker_instance is not None:
            webui_cb = WebUIProgressCallback(
                tracker=progress_tracker_instance,
                base=base_progress,
                total=total_files_for_session,
                idx=current_file_idx,
                name=original_file_name_for_desc,
                cancel_token=cancellation_token,
            )

        _ensure_not_cancelled()
        try:
            result = orchestrator.run_transcription(
                audio_path=audio_file_path,
                backend_config=backend_config,
                task=config.task,
                language=(
                    config.language
                    if config.language and config.language.lower() != "none"
                    else None
                ),
                timestamp_type=config.timestamp_type,
                progress_callback=webui_cb,
                warning_callback=_warning_callback,
                save_transcriptions=file_config.save_transcriptions,
                output_dir=file_config.temp_uploads_dir,
            )
        except OutOfMemoryError as oom:
            error_msg = (
                f"Transcription failed due to insufficient memory. Try: "
                f"(1) selecting a smaller model, (2) reducing batch size "
                f"manually, "
                f"or (3) processing shorter audio segments. "
                f"Current settings: model={config.model}, "
                f"batch_size={config.batch_size}, "
                f"chunk_length={config.chunk_length}"
            )
            logger.error(error_msg)
            raise TranscriptionError(error_msg) from oom

        if progress_tracker_instance is not None and isinstance(result, dict):
            attempts = result.get("orchestrator_attempts")
            if isinstance(attempts, list) and len(attempts) > 1:
                summary_parts: list[str] = []
                for attempt in attempts:
                    cfg = attempt.get("config")
                    if not isinstance(cfg, dict):
                        continue
                    device = cfg.get("device")
                    dtype = cfg.get("dtype")
                    batch_size = cfg.get("batch_size")
                    chunk_length = cfg.get("chunk_length")
                    attempt_no = attempt.get("attempt")
                    summary_parts.append(
                        f"{attempt_no}) {device}/{dtype} "
                        + f"bs={batch_size} cl={chunk_length}"
                    )

                if summary_parts:
                    progress_tracker_instance(
                        None,
                        desc=(
                            "Attempts: "
                            + " | ".join(summary_parts)
                            + f" ({original_file_name_for_desc})"
                        ),
                    )

        _ensure_not_cancelled()

        # Conditionally apply timestamp stabilization
        if config.stabilize:
            logger.info("Applying timestamp stabilization...")
            if progress_tracker_instance is not None:
                # Show an indeterminate progress while stabilization runs
                desc = (
                    f"Stabilizing timestamps (demucs={config.demucs}, "
                    f"vad={config.vad}, threshold={config.vad_threshold}) "
                    f"for {original_file_name_for_desc}"
                )
                progress_tracker_instance(None, desc=desc)
            # Relay detailed stabilization progress messages to the UI

            def _stab_progress(msg: str) -> None:
                """Update progress during timestamp stabilization.

                Args:
                    msg: Progress message from the stabilization process.
                """
                if cancellation_token.cancelled:
                    return
                if progress_tracker_instance is not None and getattr(
                    progress_tracker_instance, "cancelled", False
                ):
                    cancellation_token.cancel()
                    return
                if progress_tracker_instance is not None:
                    progress_tracker_instance(
                        None,
                        desc=f"{msg} ({original_file_name_for_desc})",
                    )

            _ensure_not_cancelled()
            original_result = result

            heartbeat_stop = threading.Event()
            heartbeat_thread: threading.Thread | None = None
            if progress_tracker_instance is not None:
                demucs_label = "demucs" if config.demucs else "no-demucs"
                vad_label = "vad" if config.vad else "no-vad"
                heartbeat_desc = (
                    "Stabilizing timestamps "
                    f"({demucs_label}, {vad_label}) ({original_file_name_for_desc})"
                )

                def _heartbeat() -> None:
                    """Periodically update progress during long operations.

                    This function runs in a background thread and updates the progress
                    tracker every 5 seconds with a heartbeat message to keep the UI
                    responsive during long-running stabilization operations.
                    """
                    while not heartbeat_stop.is_set():
                        if cancellation_token.cancelled:
                            return
                        if getattr(progress_tracker_instance, "cancelled", False):
                            cancellation_token.cancel()
                            return
                        progress_tracker_instance(None, desc=heartbeat_desc)
                        time.sleep(5)

                heartbeat_thread = threading.Thread(
                    target=_heartbeat,
                    daemon=True,
                )
                heartbeat_thread.start()

            try:
                stabilized_result = stabilize_timestamps(
                    result,
                    demucs=config.demucs,
                    vad=config.vad,
                    vad_threshold=config.vad_threshold,
                    progress_cb=_stab_progress,
                )
            finally:
                heartbeat_stop.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=1)

            _ensure_not_cancelled()

            if _is_stabilization_corrupt(stabilized_result.get("segments", [])):
                logger.warning(
                    "Stabilization produced corrupted timestamps. Falling back "
                    + "to original transcription."
                )
                result = original_result
            else:
                result = stabilized_result
            if progress_tracker_instance is not None:
                progress_tracker_instance(
                    None,
                    desc=(f"Stabilization complete for {original_file_name_for_desc}"),
                )

            _ensure_not_cancelled()

        # Optional diarization post-processing
        if config.diarize:
            _ensure_not_cancelled()
            logger.info(
                "Diarization enabled for %s (device=%s, num=%s, min=%s, max=%s)",
                original_file_name_for_desc,
                config.diarization_device,
                config.num_speakers,
                config.min_speakers,
                config.max_speakers,
            )
            try:
                from insanely_fast_whisper_rocm.core.integrations.diarization import (
                    diarize as diarize_result,
                )

                if progress_tracker_instance is not None:
                    progress_tracker_instance(
                        None, desc="Running speaker diarization..."
                    )
                result = diarize_result(
                    result,
                    audio_path=audio_file_path,
                    num_speakers=config.num_speakers,
                    min_speakers=config.min_speakers,
                    max_speakers=config.max_speakers,
                    device=config.diarization_device,
                    hf_token=constants.HF_TOKEN,
                )
                if result.get("diarized"):
                    logger.info(
                        "Diarization completed for %s", original_file_name_for_desc
                    )
                else:
                    logger.warning(
                        "Diarization did not annotate result for %s",
                        original_file_name_for_desc,
                    )
            except DiarizationError as exc:
                logger.warning("Diarization failed: %s", exc)
                result = {
                    **result,
                    "diarized": False,
                    "diarization_error": str(exc),
                }
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("Diarization failed: %s", exc)
                result = {
                    **result,
                    "diarized": False,
                    "diarization_error": str(exc),
                }

            _ensure_not_cancelled()

        logger.info("Transcription completed successfully for %s", audio_file_path)

        # Final progress update for this file's segment upon successful completion
        if progress_tracker_instance is not None:
            final_progress_for_this_file_segment = (
                current_file_idx + 1
            ) / total_files_for_session
            progress_tracker_instance(
                final_progress_for_this_file_segment,
                desc=(
                    f"Completed file {current_file_idx + 1}/{total_files_for_session}: "
                    f"{original_file_name_for_desc}"
                ),
            )

        return result

    except TranscriptionCancelledError as exc:
        logger.info("Transcription cancelled for %s", audio_file_path)
        raise exc
    except Exception as e:
        logger.error("Error during transcription: %s", str(e))
        raise TranscriptionError(f"Transcription failed: {str(e)}") from e


def process_transcription_request(  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    audio_paths: list[str],
    transcription_config: TranscriptionConfig,
    file_handling_config: FileHandlingConfig,
    progress_tracker: gr.Progress | None = None,
) -> tuple[
    str,
    str,
    str | dict[str, Any],
    Any,
    Any,
    Any,
    Any,
    Any,
]:
    """Process one or more files and prepare WebUI outputs.

    Process transcription for one or more audio files, generate results, and
    prepare Gradio UI component updates using `BatchZipBuilder`.

    Args:
        audio_paths: List of paths to audio files to transcribe.
        transcription_config: Configuration for transcription.
        file_handling_config: Configuration for file handling.
        progress_tracker: Optional progress tracker for real-time progress updates.

    Returns:
        Tuple of Gradio UI component updates for status, transcription output,
        JSON output, raw result, and download widgets.

    Raises:
        TranscriptionCancelledError: If a user-triggered cancellation stops processing.

    """
    all_results_data = []
    processed_files_summary = []

    # Initialize default Gradio button updates (hidden) early so error paths can
    # safely reference them.
    dl_btn_hidden_update = gr.update(visible=False, value=None)

    # output_base_dir is where pipeline saves JSON and where our ZIPs will go.
    output_base_dir = Path(file_handling_config.temp_uploads_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    num_files = len(audio_paths)
    current_task_type = TaskType(transcription_config.task)  # For filename generator

    for idx, audio_file_path_str in enumerate(audio_paths):
        audio_file_path = Path(audio_file_path_str)
        file_name_for_log = audio_file_path.name

        # Progress updates are now handled within the 'transcribe' function
        # based on current_file_idx and total_files_for_session.

        try:
            # Pass the main progress tracker to the transcribe function
            result_dict = transcribe(
                str(audio_file_path),
                transcription_config,
                file_handling_config,
                progress_tracker_instance=progress_tracker,
                current_file_idx=idx,
                total_files_for_session=num_files,
            )

            # The asr_pipeline.process() returns the transcription data directly.
            # Stabilization is performed inside transcribe() when enabled, so we
            # do not repeat it here to avoid duplicate work.
            raw_transcription_result = result_dict
            # This is the path to the JSON file saved by the pipeline
            json_file_path_from_pipeline = result_dict.get("output_file_path")

            if (
                not json_file_path_from_pipeline
                and file_handling_config.save_transcriptions
            ):
                # This case should ideally not happen if pipeline guarantees
                # output_file_path
                logger.error(
                    (
                        "JSON file path missing from pipeline for %s despite "
                        "save_transcriptions=True. Generating fallback."
                    ),
                    file_name_for_log,
                )
                # Fallback: generate filename for JSON if pipeline didn't provide path
                # This JSON is for our records/data, not necessarily for direct download
                # button if pipeline failed to save
                fallback_json_filename = WEBUI_FILENAME_GENERATOR.create_filename(
                    audio_path=str(audio_file_path),
                    task=current_task_type,
                    extension="json",
                )
                json_file_path_from_pipeline = str(
                    output_base_dir / fallback_json_filename
                )
                with open(json_file_path_from_pipeline, "w", encoding="utf-8") as f:
                    json.dump(raw_transcription_result, f, indent=2, ensure_ascii=False)
                logger.info("Fallback: Saved JSON to %s", json_file_path_from_pipeline)

            # No longer saving individual TXT/SRT here. They'll be generated on-the-fly
            # for download or created by BatchZipBuilder within ZIPs.

            all_results_data.append({
                "audio_original_path": str(audio_file_path),  # Store full path
                "audio_original_stem": audio_file_path.stem,
                "raw_result": raw_transcription_result,
                # Path to pipeline-saved JSON
                "json_file_path": json_file_path_from_pipeline,
            })
            processed_files_summary.append(
                f"{file_name_for_log}: Transcribed successfully."
            )

            if progress_tracker is not None:
                progress_tracker(
                    (idx + 1) / num_files,
                    desc=f"Completed file {idx + 1}/{num_files}: {file_name_for_log}",
                )

        except TranscriptionCancelledError:
            logger.info(
                "Transcription cancelled by user during processing of %s",
                file_name_for_log,
            )
            raise
        except TranscriptionError as e:
            logger.error("Error transcribing %s: %s", file_name_for_log, e)
            processed_files_summary.append(f"{file_name_for_log}: Error - {e}")
            all_results_data.append({
                "audio_original_path": str(audio_file_path),
                "error": str(e),
            })
            if progress_tracker is not None:
                progress_tracker(
                    (idx + 1) / num_files,
                    desc=(
                        f"Error processing file {idx + 1}/{num_files}: "
                        f"{file_name_for_log}"
                    ),
                )
            if num_files > 1:
                continue
            status_output_val = f"Error processing {file_name_for_log}."
            transcription_output_val = f"Error processing {file_name_for_log}: {e}"
            json_output_val = {"error": str(e), "file": file_name_for_log}
            raw_result_state_val = None
            return (
                status_output_val,
                transcription_output_val,
                json_output_val,
                raw_result_state_val,
                dl_btn_hidden_update,
                dl_btn_hidden_update,
                dl_btn_hidden_update,
                dl_btn_hidden_update,
            )
        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            AttributeError,
            KeyError,
        ) as e:
            logger.error(
                "Unexpected error processing %s: %s",
                file_name_for_log,
                e,
                exc_info=True,
            )
            processed_files_summary.append(
                f"{file_name_for_log}: Unexpected Error - {e}"
            )
            all_results_data.append({
                "audio_original_path": str(audio_file_path),
                "error": str(e),
            })
            if progress_tracker is not None:
                progress_tracker(
                    (idx + 1) / num_files,
                    desc=(
                        f"Critical error file {idx + 1}/{num_files}: "
                        f"{file_name_for_log}"
                    ),
                )
            if num_files > 1:
                continue
            status_output_val = (
                f"Unexpected error while processing {file_name_for_log}."
            )
            transcription_output_val = f"Unexpected error with {file_name_for_log}: {e}"
            json_output_val = {
                "error": str(e),
                "file": file_name_for_log,
                "details": "Check logs.",
            }
            raw_result_state_val = None
            return (
                status_output_val,
                transcription_output_val,
                json_output_val,
                raw_result_state_val,
                dl_btn_hidden_update,  # zip_btn_update
                dl_btn_hidden_update,  # txt_btn_update
                dl_btn_hidden_update,  # srt_btn_update
                dl_btn_hidden_update,  # json_btn_update
            )

    if not all_results_data:
        return (
            "No files processed.",
            "No files processed.",
            {},
            {},
            dl_btn_hidden_update,  # zip_btn_update
            dl_btn_hidden_update,  # txt_btn_update
            dl_btn_hidden_update,  # srt_btn_update
            dl_btn_hidden_update,  # json_btn_update
        )

    # Initialize default Gradio button updates (hidden)
    zip_btn_update = txt_btn_update = srt_btn_update = json_btn_update = (
        dl_btn_hidden_update
    )

    successful_results = [res for res in all_results_data if "error" not in res]

    if not successful_results:
        error_summary_msg = "\n".join(processed_files_summary)
        transcription_output_val = (
            f"All {num_files} files failed to process.\nDetails:\n{error_summary_msg}"
        )
        json_output_val = {
            "summary": processed_files_summary,
            "errors": [res for res in all_results_data if "error" in res],
        }
        raw_result_state_val = None
        return (
            f"All {num_files} files failed.",
            transcription_output_val,
            json_output_val,
            raw_result_state_val,
            dl_btn_hidden_update,
            dl_btn_hidden_update,
            dl_btn_hidden_update,
            dl_btn_hidden_update,
        )

    # Common configuration timestamp for all zip files
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    if num_files == 1:
        first_success = successful_results[0]
        status_output_val = (
            f"Completed file 1/1: {Path(first_success['audio_original_path']).name}"
        )
        # Use FORMATTERS for display text
        display_txt_formatter = FORMATTERS.get("txt")
        transcription_output_val = (
            display_txt_formatter.format(first_success["raw_result"])
            if display_txt_formatter
            else "Could not format text output."
        )
        json_output_val = _build_ui_json_preview(
            first_success["raw_result"],
            json_file_path=None,
        )
        raw_result_state_val = first_success["raw_result"]

        # Individual file downloads - wrap each in try/except to prevent hangs
        try:
            txt_btn_update = gr.update(
                value=_prepare_temp_downloadable_file(
                    first_success["raw_result"],
                    "txt",
                    first_success["audio_original_stem"],
                    output_base_dir,
                    current_task_type,
                ),
                visible=True,
            )
        except Exception as txt_e:
            logger.error("Failed to prepare TXT download: %s", txt_e, exc_info=True)
            txt_btn_update = dl_btn_hidden_update

        try:
            srt_btn_update = gr.update(
                value=_prepare_temp_downloadable_file(
                    first_success["raw_result"],
                    "srt",
                    first_success["audio_original_stem"],
                    output_base_dir,
                    current_task_type,
                ),
                visible=True,
            )
        except Exception as srt_e:
            logger.error("Failed to prepare SRT download: %s", srt_e, exc_info=True)
            srt_btn_update = dl_btn_hidden_update

        try:
            json_btn_update = gr.update(
                value=_prepare_temp_downloadable_file(
                    first_success["raw_result"],
                    "json",
                    first_success["audio_original_stem"],
                    output_base_dir,
                    current_task_type,
                ),
                visible=True,
            )
        except Exception as json_e:
            logger.error("Failed to prepare JSON download: %s", json_e, exc_info=True)
            json_btn_update = dl_btn_hidden_update

        # "Download All (ZIP)" for single file
        try:
            single_zip_config = ZipConfiguration(
                temp_dir=str(output_base_dir),
                organize_by_format=False,
                include_summary=False,
            )
            single_zip_builder = BatchZipBuilder(config=single_zip_config)
            # Use original audio stem for a more descriptive ZIP name
            zip_filename = (
                f"{first_success['audio_original_stem']}_ALL_{timestamp_str}.zip"
            )

            # Data for builder:
            # { "original_audio_path_for_internal_naming" : raw_result_data }
            # The key is used by BatchZipBuilder to name files inside the zip if
            # not organizing by format.
            # Path(first_success['audio_original_path']).name might be better
            # if stem is too simple.
            data_for_builder = {
                Path(first_success["audio_original_path"]).name: first_success[
                    "raw_result"
                ]
            }

            single_zip_builder.create(filename=zip_filename)
            single_zip_builder.add_batch_files(
                data_for_builder, formats=["txt", "srt", "json"]
            )
            single_all_zip_path, _ = single_zip_builder.build()

            zip_btn_update = gr.update(
                value=_to_gradio_file_value(single_all_zip_path),
                visible=True,
            )
            logger.info(
                "Prepared single file downloads and ALL_ZIP=%s", single_all_zip_path
            )
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            FileNotFoundError,
            zipfile.BadZipFile,
        ) as e:
            logger.error(
                "Failed to create ALL ZIP for single file: %s", e, exc_info=True
            )
            # zip_btn_update remains hidden (dl_btn_hidden)
            processed_files_summary.append(
                f"{first_success['audio_original_stem']}: Failed to create ZIP - {e}"
            )

    elif num_files > 0:  # Multiple files (at least one success)
        successful_transcriptions = len(successful_results)  # Initialize here

        # Initialize button states for multi-file case
        txt_btn_update = dl_btn_hidden_update
        srt_btn_update = dl_btn_hidden_update
        json_btn_update = dl_btn_hidden_update
        zip_btn_update = dl_btn_hidden_update

        # Prepare data for BatchZipBuilder: Dict[str, Dict[str, Any]]
        # Key: path/to/original_audio.mp3 (used by builder for internal naming)
        # Value: raw_transcription_result

        # Summary message
        if successful_transcriptions == num_files:
            status_output_val = (
                f"Completed {successful_transcriptions}/{num_files} files."
            )
            transcription_output_val = (
                f"Successfully processed {num_files} files. Results packaged."
            )
        else:
            status_output_val = (
                f"Completed {successful_transcriptions}/{num_files} files "
                f"with {num_files - successful_transcriptions} failures."
            )
            transcription_output_val = (
                f"Processed {num_files} files. "
                + f"{successful_transcriptions} successful, "
                + f"{num_files - successful_transcriptions} failed.\n"
                + "Successful results packaged.\nSummary:\n"
                + "\n".join(processed_files_summary)
            )
        json_output_val = json.dumps(
            {
                "summary": processed_files_summary,
                "output_directory": str(output_base_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
        raw_result_state_val = None

        # Log the content of the first successful result for debugging
        if successful_results:
            first_raw = successful_results[0]["raw_result"]
            if isinstance(first_raw, dict):
                text = first_raw.get("text")
                chunks = first_raw.get("chunks")
                segments = first_raw.get("segments")
                logger.debug(
                    "First successful raw_result for multi-file: keys=%s "
                    + "text_len=%s chunks=%s segments=%s",
                    sorted(first_raw.keys()),
                    len(text) if isinstance(text, str) else None,
                    len(chunks) if isinstance(chunks, list) else None,
                    len(segments) if isinstance(segments, list) else None,
                )
            else:
                logger.debug(
                    "First successful raw_result for multi-file: type=%s",
                    type(first_raw),
                )

        # 1. Download All (ZIP) - contains txt, srt, json, organized by format
        try:
            all_zip_config = ZipConfiguration(
                temp_dir=str(output_base_dir),
                organize_by_format=True,
                include_summary=True,
            )  # Enable summary for main zip
            all_zip_builder = BatchZipBuilder(config=all_zip_config)
            all_zip_filename = f"batch_archive_{timestamp_str}_all_formats.zip"
            all_zip_builder.create(batch_id=timestamp_str, filename=all_zip_filename)
            all_zip_builder.add_batch_files(
                {
                    res["audio_original_path"]: res["raw_result"]
                    for res in successful_results
                },
                formats=["txt", "srt", "json"],
            )

            all_zip_path, _ = all_zip_builder.build()  # build() adds summary

            zip_btn_update = gr.update(
                value=_to_gradio_file_value(all_zip_path),
                visible=True,
            )
            logger.info(
                "Prepared ALL ZIP: %s, Files: %s", all_zip_path, len(successful_results)
            )
            json_output_val = json.dumps(
                {
                    "summary": processed_files_summary,
                    "output_directory": str(output_base_dir),
                    "zip_archive_all": Path(all_zip_path).name,
                },
                indent=2,
                ensure_ascii=False,
            )
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            FileNotFoundError,
            zipfile.BadZipFile,
        ) as e:
            logger.error("Failed to create ALL ZIP for batch: %s", e, exc_info=True)
            # zip_btn_update remains hidden

        # 2. Download All TXT (ZIP)
        txt_files_exist = bool(successful_results)  # Simpler check
        logger.debug("Multi-file txt_files_exist: %s", txt_files_exist)

        if txt_files_exist:
            try:
                txt_zip_config = ZipConfiguration(
                    temp_dir=str(output_base_dir), organize_by_format=False
                )  # Flat for single type
                txt_zip_builder = BatchZipBuilder(config=txt_zip_config)
                txt_zip_filename = f"batch_archive_{timestamp_str}_txt_only.zip"
                txt_zip_builder.create(
                    batch_id=timestamp_str, filename=txt_zip_filename
                )
                txt_zip_builder.add_batch_files(
                    {
                        res["audio_original_path"]: res["raw_result"]
                        for res in successful_results
                    },
                    formats=["txt"],
                )

                txt_zip_path, _ = txt_zip_builder.build()  # build() adds summary
                txt_btn_update = gr.update(
                    value=_to_gradio_file_value(txt_zip_path),
                    visible=True,
                )
                logger.info(
                    "Prepared TXT ZIP: %s, Files: %s",
                    txt_zip_path,
                    len(successful_results),
                )
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                FileNotFoundError,
                zipfile.BadZipFile,
            ) as e:
                logger.error("Failed to create TXT ZIP for batch: %s", e, exc_info=True)

        # 3. Download All SRT (ZIP)
        srt_files_exist = bool(successful_results)  # Simpler check
        logger.debug("Multi-file srt_files_exist: %s", srt_files_exist)

        if srt_files_exist:
            try:
                srt_zip_config = ZipConfiguration(
                    temp_dir=str(output_base_dir), organize_by_format=False
                )
                srt_zip_builder = BatchZipBuilder(config=srt_zip_config)
                srt_zip_filename = f"batch_archive_{timestamp_str}_srt_only.zip"
                srt_zip_builder.create(
                    batch_id=timestamp_str, filename=srt_zip_filename
                )
                srt_zip_builder.add_batch_files(
                    {
                        res["audio_original_path"]: res["raw_result"]
                        for res in successful_results
                    },
                    formats=["srt"],
                )

                srt_zip_path, _ = srt_zip_builder.build()  # build() adds summary
                srt_btn_update = gr.update(
                    value=_to_gradio_file_value(srt_zip_path),
                    visible=True,
                )
                logger.info(
                    "Prepared SRT ZIP: %s, Files: %s",
                    srt_zip_path,
                    len(successful_results),
                )
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                FileNotFoundError,
                zipfile.BadZipFile,
            ) as e:
                logger.error("Failed to create SRT ZIP for batch: %s", e, exc_info=True)

        # 4. Download All JSON (ZIP)
        # JSON files are generated from raw_result by BatchZipBuilder
        json_files_exist = bool(successful_results)  # Simpler check
        logger.debug("Multi-file json_files_exist: %s", json_files_exist)
        # No need to log sample JSON output as BatchZipBuilder handles its
        # creation directly from raw_result

        if json_files_exist:
            try:
                json_zip_config = ZipConfiguration(
                    temp_dir=str(output_base_dir), organize_by_format=False
                )
                json_zip_builder = BatchZipBuilder(config=json_zip_config)
                json_zip_filename = f"batch_archive_{timestamp_str}_json_only.zip"
                json_zip_builder.create(
                    batch_id=timestamp_str, filename=json_zip_filename
                )
                json_zip_builder.add_batch_files(
                    {
                        res["audio_original_path"]: res["raw_result"]
                        for res in successful_results
                    },
                    formats=["json"],
                )

                json_zip_path, _ = json_zip_builder.build()  # build() adds summary
                json_btn_update = gr.update(
                    value=_to_gradio_file_value(json_zip_path),
                    visible=True,
                )
                logger.info(
                    "Prepared JSON ZIP: %s, Files: %s",
                    json_zip_path,
                    len(successful_results),
                )
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                FileNotFoundError,
                zipfile.BadZipFile,
            ) as e:
                logger.error(
                    "Failed to create JSON ZIP for batch: %s", e, exc_info=True
                )

    # This `else` case for num_files == 0 should be caught by `if not all_results_data:`
    # or `if not successful_results:`. Adding defensively.
    else:
        status_output_val = "No valid results to process."
        transcription_output_val = "No valid results to process."
        json_output_val = {"error": "No results"}
        raw_result_state_val = None
        # All buttons remain hidden (dl_btn_hidden)

    if progress_tracker is not None:
        # Final update to 100% if all files processed (or attempted)
        progress_tracker(1.0, desc="Done")

    logger.info(
        "WebUI response summary: transcription_text_len=%s json_keys=%s state=%s ",
        (
            len(transcription_output_val)
            if isinstance(transcription_output_val, str)
            else None
        ),
        sorted(json_output_val.keys()) if isinstance(json_output_val, dict) else None,
        type(raw_result_state_val).__name__,
    )

    return (
        status_output_val,
        transcription_output_val,
        json_output_val,
        raw_result_state_val,
        zip_btn_update,
        txt_btn_update,
        srt_btn_update,
        json_btn_update,
    )
