"""Gradio UI components for the Insanely Fast Whisper API.

This module provides the main UI components and layout for the web interface,
including file upload, processing controls, and result display components.
"""

import logging
from typing import List

import gradio as gr

from insanely_fast_whisper_api.utils.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_TIMESTAMP_TYPE,
    DEFAULT_TRANSCRIPTS_DIR,
    MAX_BATCH_SIZE,
    MIN_BATCH_SIZE,
    SUPPORTED_AUDIO_FORMATS,
)
from insanely_fast_whisper_api.webui.handlers import (
    FileHandlingConfig,
    TranscriptionConfig,
    process_transcription_request,
)

# Configure logger
logger = logging.getLogger("insanely_fast_whisper_api.webui.ui")


def _create_model_config_ui():
    """Helper function to create model configuration UI components."""
    with gr.Accordion("Model Configuration", open=True):
        model = gr.Textbox(value=DEFAULT_MODEL, label="Model")
        device = gr.Textbox(value=DEFAULT_DEVICE, label="Device (e.g., 0, cpu, mps)")
        batch_size = gr.Slider(
            minimum=MIN_BATCH_SIZE,
            maximum=MAX_BATCH_SIZE,
            step=1,
            value=DEFAULT_BATCH_SIZE,
            label="Batch Size",
        )
    return model, device, batch_size


def _create_processing_options_ui():
    """Helper function to create processing options UI components."""
    with gr.Accordion("Processing Options", open=False):
        dtype = gr.Dropdown(
            choices=["float16", "float32"],
            value="float16",
            label="Precision",
            info="Lower precision (float16) is faster but may be less accurate",
        )
        better_transformer = gr.Checkbox(
            value=False,
            label="Use BetterTransformer",
            info="May improve performance on some systems",
        )
        chunk_length = gr.Slider(
            minimum=10,
            maximum=60,
            step=5,
            value=30,
            label="Processing Chunk Length (seconds)",
            info=(
                "Length of audio segments for model processing. "
                "Longer chunks may be more accurate but use more memory"
            ),
        )
    return dtype, better_transformer, chunk_length


def _create_task_config_ui():
    """Helper function to create task configuration UI components."""
    with gr.Accordion("Task Configuration", open=True):
        timestamp_type = gr.Radio(
            choices=["chunk", "word"],
            label="Timestamp Type",
            value=DEFAULT_TIMESTAMP_TYPE,
        )
        language = gr.Textbox(
            value=DEFAULT_LANGUAGE,
            label="Language in ISO format (use 'None' for auto detection)",
            placeholder="en, fr, de, etc.",
        )
        task = gr.Radio(
            choices=["transcribe", "translate"],
            label="Task",
            value="transcribe",
        )
    return timestamp_type, language, task


def _create_file_handling_ui():
    """Helper function to create file handling UI components."""
    with gr.Accordion("File Handling", open=False):
        save_transcriptions = gr.Checkbox(
            value=True, label="Save transcriptions to disk"
        )
        temp_uploads_dir = gr.Textbox(
            value=DEFAULT_TRANSCRIPTS_DIR,
            label="Save directory",
            info="Directory to save transcription results",
        )
    return save_transcriptions, temp_uploads_dir


def _process_transcription_request_wrapper(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    audio_paths: List[str],
    model_name: str,
    device: str,
    batch_size: int,
    timestamp_type: str,
    language: str,
    task: str,
    dtype: str,
    better_transformer: bool,
    whisper_chunk_length: int,
    save_transcriptions: bool,
    temp_uploads_dir: str,
    progress: gr.Progress = gr.Progress(),
):
    """Wrapper to adapt Gradio inputs to process_transcription_request."""
    transcription_cfg = TranscriptionConfig(
        model=model_name,
        device=device,
        batch_size=batch_size,
        timestamp_type=timestamp_type,
        language=language,
        task=task,
        dtype=dtype,
        better_transformer=better_transformer,
        chunk_length=whisper_chunk_length,
        chunk_duration=None,
        chunk_overlap=None,
    )
    file_handling_cfg = FileHandlingConfig(
        save_transcriptions=save_transcriptions, temp_uploads_dir=temp_uploads_dir
    )
    return process_transcription_request(
        audio_paths=audio_paths,
        transcription_config=transcription_cfg,
        file_handling_config=file_handling_cfg,
        progress_tracker=progress,
    )


def create_ui_components():  # pylint: disable=too-many-locals
    """Create and return Gradio UI components with all parameters."""
    with gr.Blocks(title="Insanely Fast Whisper - Local WebUI") as demo:
        gr.Markdown("# 🎙️ Insanely Fast Whisper - Local WebUI")
        gr.Markdown(
            "Transcribe or translate audio files using Whisper models directly in your browser."
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Audio input
                audio_input = gr.File(
                    label="Upload Audio File(s)",
                    type="filepath",
                    file_count="multiple",
                    file_types=list(SUPPORTED_AUDIO_FORMATS),
                )

                # Model configuration
                model, device, batch_size = _create_model_config_ui()

                # Processing options
                dtype, better_transformer, chunk_length = (
                    _create_processing_options_ui()
                )

                # Task configuration
                timestamp_type, language, task = _create_task_config_ui()

                # File handling
                save_transcriptions, temp_uploads_dir = _create_file_handling_ui()

                submit_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=3):
                # Outputs
                with gr.Tabs():
                    with gr.TabItem("Transcription / Summary"):
                        transcription_output = gr.Textbox(
                            label="Transcription Result / Processing Summary",
                            lines=15,
                            interactive=False,
                        )
                    with gr.TabItem("Raw JSON / Details"):
                        json_output = gr.JSON(
                            label="Raw JSON Output / Detailed Results", visible=True
                        )

                raw_result_state = gr.State()

                with gr.Row():
                    download_txt_btn = gr.DownloadButton(
                        "Download TXT", visible=False, interactive=False
                    )
                    download_srt_btn = gr.DownloadButton(
                        "Download SRT", visible=False, interactive=False
                    )
                    download_json_btn = gr.DownloadButton(
                        "Download JSON", visible=False, interactive=False
                    )
                download_zip_btn = gr.DownloadButton(
                    "Download All as ZIP", visible=False, interactive=False
                )

        # Event handling
        submit_btn.click(
            fn=_process_transcription_request_wrapper,
            inputs=[
                audio_input,
                model,
                device,
                batch_size,
                timestamp_type,
                language,
                task,
                dtype,
                better_transformer,
                chunk_length,
                save_transcriptions,
                temp_uploads_dir,
            ],
            outputs=[
                transcription_output,
                json_output,
                raw_result_state,
                download_zip_btn,
                download_txt_btn,
                download_srt_btn,
                download_json_btn,
            ],
            api_name="transcribe_audio_v2",
        )

    return demo
