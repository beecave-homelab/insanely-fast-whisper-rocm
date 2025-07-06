"""Handler functions for Insanely Fast Whisper API WebUI.

This module contains the core logic for handling transcription requests
and exporting results in the WebUI. It serves as an intermediary between
the UI components and the ASR pipeline.
"""

import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import gradio as gr

from insanely_fast_whisper_api import ASRPipeline
from insanely_fast_whisper_api.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)
from insanely_fast_whisper_api.core.errors import TranscriptionError
from insanely_fast_whisper_api.core.pipeline import ProgressEvent
from insanely_fast_whisper_api.utils import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_TIMESTAMP_TYPE,
    DEFAULT_TRANSCRIPTS_DIR,
    constants,
)
from insanely_fast_whisper_api.utils.filename_generator import (
    FilenameGenerator,
    StandardFilenameStrategy,
    TaskType,
)
from insanely_fast_whisper_api.core.formatters import FORMATTERS
from insanely_fast_whisper_api.webui.zip_creator import (
    BatchZipBuilder,
    ZipConfiguration,
)

# Configure logger
logger = logging.getLogger("insanely_fast_whisper_api.webui.handlers")

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
    chunk_duration: Optional[float] = None
    chunk_overlap: Optional[float] = None


@dataclass
class FileHandlingConfig:
    """Configuration for file handling."""

    save_transcriptions: bool = True
    temp_uploads_dir: str = DEFAULT_TRANSCRIPTS_DIR


def _prepare_temp_downloadable_file(
    raw_data: Dict[str, Any],
    format_type: str,  # "txt" or "srt"
    original_audio_stem: str,
    temp_dir: Path,
    task: TaskType,
) -> str:
    """
    Generates content for TXT or SRT, saves it to a temporary file,
    and returns the file path.
    """
    formatter = FORMATTERS.get(format_type)
    if not formatter:
        raise ValueError(f"No formatter available for type: {format_type}")

    content = formatter.format(raw_data)

    # Use the global WEBUI_FILENAME_GENERATOR for consistent naming
    # Filename will be like: audio_stem_task_timestamp.format
    # We need a unique name, timestamp is good.
    # The generator itself handles the timestamp.
    filename = WEBUI_FILENAME_GENERATOR.create_filename(
        audio_path=original_audio_stem,  # Generator expects a path-like string for stem extraction
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
        logger.info("Created temporary download file: %s", temp_file_path)
        return str(temp_file_path)
    except IOError as e:
        logger.error(
            "Failed to create temporary download file %s: %s", temp_file_path, e
        )
        raise


def transcribe(
    audio_file_path: str,
    config: TranscriptionConfig,
    file_config: FileHandlingConfig,
    progress_tracker_instance: Optional[gr.Progress] = None,
    current_file_idx: int = 0,
    total_files_for_session: int = 1,
) -> Dict[str, Any]:
    """
    Transcribe an audio file using the ASRPipeline.

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
    """
    try:
        logger.info(
            "Starting transcription for file: %s (File %d/%d)",
            audio_file_path,
            current_file_idx + 1,
            total_files_for_session,
        )
        original_file_name_for_desc = Path(audio_file_path).name

        # Initial progress update for this file's segment
        base_progress = current_file_idx / total_files_for_session
        if progress_tracker_instance is not None:
            progress_tracker_instance(
                base_progress,
                desc=f"Starting file {current_file_idx + 1}/{total_files_for_session}: {original_file_name_for_desc}",
            )

        # Initialize the ASR backend
        backend_config = HuggingFaceBackendConfig(
            model_name=config.model,
            device=config.device,
            dtype=config.dtype,
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )
        backend = HuggingFaceBackend(config=backend_config)

        # Initialize the ASR pipeline
        asr_pipeline = ASRPipeline(
            asr_backend=backend,
            save_transcriptions=file_config.save_transcriptions,
            output_dir=file_config.temp_uploads_dir,
        )

        # Adapt progress_callback to the new listener pattern
        if progress_tracker_instance is not None:
            # progress_listener needs access to base_progress, current_file_idx, total_files_for_session
            # and original_file_name_for_desc for consistent messaging.

            def progress_listener(event: ProgressEvent):
                # Use original_file_name_for_desc captured from outer scope
                nonlocal base_progress

                if event.message:
                    desc_message = f"{event.message} ({original_file_name_for_desc})"
                else:
                    desc_message = (
                        f"{event.event_type} for {original_file_name_for_desc}"
                    )

                current_event_fraction_within_file = (
                    None  # 0.0 to 1.0 for the current file's internal progress
                )
                if event.event_type == "pipeline_start":
                    current_event_fraction_within_file = 0.0
                elif (
                    event.event_type == "chunk_start"
                    and event.total_chunks
                    and event.total_chunks > 0
                    and event.chunk_num is not None
                ):
                    current_event_fraction_within_file = (
                        event.chunk_num - 1
                    ) / event.total_chunks
                elif (
                    event.event_type == "chunk_complete"
                    and event.total_chunks
                    and event.total_chunks > 0
                    and event.chunk_num is not None
                ):
                    current_event_fraction_within_file = (
                        event.chunk_num / event.total_chunks
                    )
                elif event.event_type == "pipeline_complete":
                    current_event_fraction_within_file = 1.0

                if current_event_fraction_within_file is not None:
                    # Scale current_event_fraction (0-1 for this file) to its portion of the total progress.
                    # Each file contributes 1/total_files_for_session to the total progress.
                    overall_fraction = base_progress + (
                        current_event_fraction_within_file / total_files_for_session
                    )
                    progress_tracker_instance(overall_fraction, desc=desc_message)
                else:
                    # For other events, show indeterminate progress or just update description
                    progress_tracker_instance(None, desc=desc_message)

            asr_pipeline.add_listener(progress_listener)

        # App-level chunking parameters (config.chunk_duration, config.chunk_overlap)
        # are not directly used by the new pipeline in this basic setup.
        # If that specific chunking logic is needed, it has to be implemented
        # within the WhisperPipeline._prepare_input or as a pre-processing step.
        # The current HuggingFaceBackend relies on the model's internal chunking (chunk_length_s).
        if config.chunk_duration is not None and config.chunk_overlap is not None:
            logger.warning(
                "App-level chunk_duration and chunk_overlap are set in config, "
                "but the current refactored pipeline does not use them directly. "
                "Transcription will use the model's internal chunking."
            )

        # Run the transcription using the new process method
        result = asr_pipeline.process(
            audio_file_path=audio_file_path,
            language=(
                config.language
                if config.language and config.language.lower() != "none"
                else None
            ),
            task=config.task,
            timestamp_type=config.timestamp_type,
        )

        logger.info("Transcription completed successfully for %s", audio_file_path)

        # Final progress update for this file's segment upon successful completion
        if progress_tracker_instance is not None:
            final_progress_for_this_file_segment = (
                current_file_idx + 1
            ) / total_files_for_session
            progress_tracker_instance(
                final_progress_for_this_file_segment,
                desc=f"Completed file {current_file_idx + 1}/{total_files_for_session}: {original_file_name_for_desc}",
            )

        return result

    except Exception as e:
        logger.error("Error during transcription: %s", str(e))
        raise TranscriptionError(f"Transcription failed: {str(e)}") from e


def process_transcription_request(  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    audio_paths: List[str],
    transcription_config: TranscriptionConfig,
    file_handling_config: FileHandlingConfig,
    progress_tracker: Optional[gr.Progress] = None,
) -> Tuple[
    str, Any, Any, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]
]:
    """
    Process transcription for one or more audio files, generate results,
    and prepare Gradio UI component updates using BatchZipBuilder.
    """
    all_results_data = []
    processed_files_summary = []

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
            result_dict = transcribe(
                str(audio_file_path),
                transcription_config,
                file_handling_config,
                progress_tracker_instance=progress_tracker,  # Pass the main progress tracker
                current_file_idx=idx,
                total_files_for_session=num_files,
            )

            # FIX: The asr_pipeline.process() returns the transcription data directly,
            # not wrapped in a "raw_result" field. result_dict IS the raw result.
            raw_transcription_result = result_dict
            # This is the path to the JSON file saved by the pipeline
            json_file_path_from_pipeline = result_dict.get("output_file_path")

            if (
                not json_file_path_from_pipeline
                and file_handling_config.save_transcriptions
            ):
                # This case should ideally not happen if pipeline guarantees output_file_path
                logger.error(
                    "JSON file path missing from pipeline for %s despite save_transcriptions=True. Generating fallback.",
                    file_name_for_log,
                )
                # Fallback: generate filename for JSON if pipeline didn't provide path
                # This JSON is for our records/data, not necessarily for direct download button if pipeline failed to save
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

            # No longer saving individual TXT/SRT here. They'll be generated on-the-fly for download
            # or created by BatchZipBuilder within ZIPs.

            all_results_data.append(
                {
                    "audio_original_path": str(audio_file_path),  # Store full path
                    "audio_original_stem": audio_file_path.stem,
                    "raw_result": raw_transcription_result,
                    "json_file_path": json_file_path_from_pipeline,  # Path to pipeline-saved JSON
                }
            )
            processed_files_summary.append(
                f"{file_name_for_log}: Transcribed successfully."
            )

            if progress_tracker is not None:
                progress_tracker(
                    (idx + 1) / num_files,
                    desc=f"Completed file {idx+1}/{num_files}: {file_name_for_log}",
                )

        except TranscriptionError as e:
            logger.error("Error transcribing %s: %s", file_name_for_log, e)
            processed_files_summary.append(f"{file_name_for_log}: Error - {e}")
            all_results_data.append(
                {"audio_original_path": str(audio_file_path), "error": str(e)}
            )
            if progress_tracker is not None:
                progress_tracker(
                    (idx + 1) / num_files,
                    desc=f"Error processing file {idx+1}/{num_files}: {file_name_for_log}",
                )
            if num_files > 1:
                continue
            transcription_output_val = f"Error processing {file_name_for_log}: {e}"
            json_output_val = {"error": str(e), "file": file_name_for_log}
            raw_result_state_val = {"error": str(e)}
            dl_btn_hidden = gr.DownloadButton(visible=False, interactive=False)
            return (
                transcription_output_val,
                json_output_val,
                raw_result_state_val,
                dl_btn_hidden,
                dl_btn_hidden,
                dl_btn_hidden,
                dl_btn_hidden,
            )
        except (
            IOError,
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
            all_results_data.append(
                {"audio_original_path": str(audio_file_path), "error": str(e)}
            )
            if progress_tracker is not None:
                progress_tracker(
                    (idx + 1) / num_files,
                    desc=f"Critical error file {idx+1}/{num_files}: {file_name_for_log}",
                )
            if num_files > 1:
                continue
            transcription_output_val = f"Unexpected error with {file_name_for_log}: {e}"
            json_output_val = {
                "error": str(e),
                "file": file_name_for_log,
                "details": "Check logs.",
            }
            raw_result_state_val = {"error": str(e), "details": "Check logs."}
            dl_btn_hidden_update = gr.update(
                visible=False, value=None, interactive=False
            )
            return (
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
            {},
            {},
            dl_btn_hidden_update,  # zip_btn_update
            dl_btn_hidden_update,  # txt_btn_update
            dl_btn_hidden_update,  # srt_btn_update
            dl_btn_hidden_update,  # json_btn_update
        )

    # Initialize default Gradio button updates (hidden)
    dl_btn_hidden_update = gr.update(visible=False, value=None, interactive=False)
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
        raw_result_state_val = {"error_summary": processed_files_summary}
        return (
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
        # Use FORMATTERS for display text
        display_txt_formatter = FORMATTERS.get("txt")
        transcription_output_val = (
            display_txt_formatter.format(first_success["raw_result"])
            if display_txt_formatter
            else "Could not format text output."
        )
        json_output_val = first_success["raw_result"]
        raw_result_state_val = (
            {  # State for potential re-use, not directly for downloads now
                "raw_result": first_success["raw_result"],
                "json_file_path": first_success["json_file_path"],
            }
        )

        # Individual file downloads
        txt_btn_update = gr.update(
            value=_prepare_temp_downloadable_file(
                first_success["raw_result"],
                "txt",
                first_success["audio_original_stem"],
                output_base_dir,
                current_task_type,
            ),
            visible=True,
            interactive=True,
            label="Download .txt",
        )
        srt_btn_update = gr.update(
            value=_prepare_temp_downloadable_file(
                first_success["raw_result"],
                "srt",
                first_success["audio_original_stem"],
                output_base_dir,
                current_task_type,
            ),
            visible=True,
            interactive=True,
            label="Download .srt",
        )
        # JSON button points to the already saved pipeline JSON
        json_btn_update = gr.update(
            value=first_success["json_file_path"],
            visible=True,
            interactive=True,
            label="Download .json",
        )

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

            # Data for builder: { "original_audio_path_for_internal_naming" : raw_result_data }
            # The key is used by BatchZipBuilder to name files inside the zip if not organizing by format.
            # Path(first_success['audio_original_path']).name might be better if stem is too simple.
            data_for_builder = {
                Path(first_success["audio_original_path"]).name: first_success[
                    "raw_result"
                ]
            }

            with single_zip_builder.create(filename=zip_filename) as builder:
                builder.add_batch_files(
                    data_for_builder, formats=["txt", "srt", "json"]
                )
                single_all_zip_path, _ = builder.build()

            zip_btn_update = gr.update(
                value=single_all_zip_path,  # Use the returned path
                visible=True,
                interactive=True,
                label="Download All (ZIP)",
            )
            raw_result_state_val["all_zip"] = single_all_zip_path
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

        # Prepare data for BatchZipBuilder: Dict[str, Dict[str, Any]]
        # Key: path/to/original_audio.mp3 (used by builder for internal naming)
        # Value: raw_transcription_result
        batch_data_for_builder = {
            res["audio_original_path"]: res["raw_result"] for res in successful_results
        }

        # Summary message
        if successful_transcriptions == num_files:
            transcription_output_val = (
                f"Successfully processed {num_files} files. Results packaged."
            )
        else:
            transcription_output_val = (
                f"Processed {num_files} files. {successful_transcriptions} successful, "
                f"{num_files - successful_transcriptions} failed.\n"
                f"Successful results packaged.\n"
                f"Summary:\n" + "\n".join(processed_files_summary)
            )
        json_output_val = {
            "summary": processed_files_summary,
            "output_directory": str(output_base_dir),
        }
        # raw_result_state_val will hold paths to the created ZIPs
        raw_result_state_val = {}

        # Log the content of the first successful result for debugging
        if successful_results:
            logger.debug(
                "First successful raw_result for multi-file: %s",
                json.dumps(successful_results[0]["raw_result"], indent=2),
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
            all_zip_builder.add_summary(include_stats=True)

            ALL_ZIP_PATH, _ = all_zip_builder.build()  # Call build() to get the path

            zip_btn_update = gr.update(
                value=ALL_ZIP_PATH,  # Use the returned path
                visible=True,
                interactive=True,
                label=f"Download All ({len(successful_results)} files) as ZIP",
            )
            raw_result_state_val["all_zip"] = ALL_ZIP_PATH
            logger.info(
                "Prepared ALL ZIP: %s, Files: %s", ALL_ZIP_PATH, len(successful_results)
            )
            json_output_val["zip_archive_all"] = Path(ALL_ZIP_PATH).name
        except (
            IOError,
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
                txt_zip_builder.add_summary(include_stats=True)

                txt_zip_path, _ = txt_zip_builder.build()
                txt_btn_update = gr.update(
                    value=txt_zip_path,
                    visible=True,
                    interactive=True,
                    label="Download All TXT (ZIP)",
                )
                raw_result_state_val["txt_zip"] = txt_zip_path
                logger.info(
                    "Prepared TXT ZIP: %s, Files: %s",
                    txt_zip_path,
                    len(successful_results),
                )
            except (
                IOError,
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
                srt_zip_builder.add_summary(include_stats=True)

                srt_zip_path, _ = srt_zip_builder.build()
                srt_btn_update = gr.update(
                    value=srt_zip_path,
                    visible=True,
                    interactive=True,
                    label="Download All SRT (ZIP)",
                )
                raw_result_state_val["srt_zip"] = srt_zip_path
                logger.info(
                    "Prepared SRT ZIP: %s, Files: %s",
                    srt_zip_path,
                    len(successful_results),
                )
            except (
                IOError,
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
        # No need to log sample JSON output as BatchZipBuilder handles its creation directly from raw_result

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
                json_zip_builder.add_summary(include_stats=True)

                json_zip_path, _ = json_zip_builder.build()
                json_btn_update = gr.update(
                    value=json_zip_path,
                    visible=True,
                    interactive=True,
                    label="Download All JSON (ZIP)",
                )
                raw_result_state_val["json_zip"] = json_zip_path
                logger.info(
                    "Prepared JSON ZIP: %s, Files: %s",
                    json_zip_path,
                    len(successful_results),
                )
            except (
                IOError,
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
        transcription_output_val = "No valid results to process."
        json_output_val = {"error": "No results"}
        raw_result_state_val = {"error": "No results"}
        # All buttons remain hidden (dl_btn_hidden)

    if progress_tracker is not None:
        # Final update to 100% if all files processed (or attempted)
        final_desc = "All files processed."
        if not successful_results and num_files > 0:
            final_desc = (
                f"Processing complete. {len(successful_results)}/{num_files} succeeded."
            )
        elif not all_results_data and num_files > 0:  # Should be caught earlier
            final_desc = "No files were processed."

        progress_tracker(1.0, desc=final_desc)

    return (
        transcription_output_val,
        json_output_val,
        raw_result_state_val,
        zip_btn_update,
        txt_btn_update,
        srt_btn_update,
        json_btn_update,
    )
