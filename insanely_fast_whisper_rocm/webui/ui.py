"""Gradio UI components for the Insanely Fast Whisper API.

This module provides the main UI components and layout for the web interface,
including file upload, processing controls, and result display components.
"""

import logging
import threading
from collections.abc import Generator
from queue import Empty, Queue
from typing import Any, Literal, cast

import gradio as gr

from insanely_fast_whisper_rocm.utils.constants import (
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
    MAX_BATCH_SIZE,
    MAX_SPEAKERS,
    MIN_BATCH_SIZE,
    MIN_SPEAKERS,
    SUPPORTED_UPLOAD_FORMATS,
)
from insanely_fast_whisper_rocm.webui.handlers import (
    FileHandlingConfig,
    TranscriptionConfig,
    apply_speaker_rename,
    build_speaker_review_state,
    build_speaker_review_summary,
    merge_review_speakers,
    process_transcription_request,
    refresh_review_downloads,
    retag_transcript_segment,
    split_review_segment_speaker,
)

# Configure logger
logger = logging.getLogger("insanely_fast_whisper_rocm.webui.ui")

_STATUS_STREAM_DONE = object()


class _StatusStreamingProgressProxy:
    """Forward progress updates and stream status descriptions.

    This proxy keeps Gradio's progress behavior intact while forwarding textual
    progress descriptions into a queue so the status textbox can be updated in
    real-time.
    """

    def __init__(
        self,
        progress: gr.Progress,
        status_queue: Queue[object],
    ) -> None:
        """Initialize the status-streaming proxy.

        Args:
            progress: Gradio progress instance used by backend processing.
            status_queue: Queue where status description strings are published.
        """
        self._progress = progress
        self._status_queue = status_queue

    @property
    def cancelled(self) -> bool:
        """Return whether the underlying Gradio progress was cancelled."""
        return bool(getattr(self._progress, "cancelled", False))

    def __call__(
        self,
        progress_value: float | None = None,
        *,
        desc: str | None = None,
    ) -> None:
        """Forward progress updates and queue status text.

        Args:
            progress_value: Fractional progress in [0, 1] or None.
            desc: Human-readable progress description.
        """
        self._progress(progress_value, desc=desc)
        if desc:
            self._status_queue.put(desc)


def _create_model_config_ui(
    default_model: str = DEFAULT_MODEL,
) -> tuple[gr.Textbox, gr.Textbox, gr.Slider]:
    """Helper to create model configuration UI components with a default model.

    Returns:
        tuple[gr.Textbox, gr.Textbox, gr.Slider]: The model, device, and
        batch size controls.
    """
    with gr.Accordion("Model Configuration", open=True):
        model = gr.Textbox(value=default_model, label="Model")
        device = gr.Textbox(value=DEFAULT_DEVICE, label="Device (e.g., 0, cpu, mps)")
        batch_size = gr.Slider(
            minimum=MIN_BATCH_SIZE,
            maximum=MAX_BATCH_SIZE,
            step=1,
            value=DEFAULT_BATCH_SIZE,
            label="Batch Size",
        )
    return model, device, batch_size


def _create_processing_options_ui() -> tuple[gr.Dropdown, gr.Slider]:
    """Helper function to create processing options UI components.

    Returns:
        tuple[gr.Dropdown, gr.Slider]: The dtype dropdown and chunk length slider.
    """
    with gr.Accordion("Processing Options", open=False):
        dtype = gr.Dropdown(
            choices=["float16", "float32"],
            value="float16",
            label="Precision",
            info="Lower precision (float16) is faster but may be less accurate",
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
    return dtype, chunk_length


def _create_stabilization_ui(
    *,
    default_stabilize: bool = DEFAULT_STABILIZE,
    default_demucs: bool = DEFAULT_DEMUCS,
    default_vad: bool = DEFAULT_VAD,
    default_vad_threshold: float = DEFAULT_VAD_THRESHOLD,
) -> tuple[gr.Checkbox, gr.Checkbox, gr.Checkbox, gr.Slider]:
    """Helper function to create timestamp stabilization UI components.

    Returns:
        tuple[gr.Checkbox, gr.Checkbox, gr.Checkbox, gr.Slider]: Stabilize,
        Demucs, VAD toggles and the VAD threshold slider.
    """
    with gr.Accordion("Timestamp Stabilization", open=False):
        stabilize = gr.Checkbox(
            value=default_stabilize,
            label="Enable word-level stabilization (--stabilize)",
        )
        demucs = gr.Checkbox(
            value=default_demucs, label="Use Demucs noise reduction (--demucs)"
        )
        vad = gr.Checkbox(value=default_vad, label="Enable VAD (--vad)")
        vad_threshold = gr.Slider(
            minimum=0.1,
            maximum=0.9,
            step=0.05,
            value=default_vad_threshold,
            label="VAD Threshold (--vad-threshold)",
        )
    return stabilize, demucs, vad, vad_threshold


def _create_diarization_ui(
    *,
    default_diarize: bool = DEFAULT_DIARIZE,
    default_min_speakers: int = MIN_SPEAKERS,
    default_max_speakers: int = MAX_SPEAKERS,
) -> tuple[gr.Checkbox, gr.Radio, gr.Slider, gr.Slider, gr.Slider, gr.Radio]:
    """Helper function to create speaker diarization UI components.

    Returns:
        tuple[gr.Checkbox, gr.Radio, gr.Slider, gr.Slider, gr.Slider, gr.Radio]:
        Diarize toggle, speaker-count mode, exact/min/max speaker sliders,
        and advanced diarization device radio.
    """
    with gr.Accordion("Speaker Diarization", open=False):
        diarize = gr.Checkbox(
            value=default_diarize,
            label="Enable speaker diarization (--diarize)",
        )
        speaker_count_mode = gr.Radio(
            choices=["Auto", "Exact", "Range"],
            value="Auto",
            label="Speaker count",
        )
        num_speakers = gr.Slider(
            minimum=0,
            maximum=MAX_SPEAKERS,
            step=1,
            value=0,
            label="Exact speakers",
        )
        min_speakers = gr.Slider(
            minimum=MIN_SPEAKERS,
            maximum=MAX_SPEAKERS,
            step=1,
            value=default_min_speakers,
            label="Min speakers",
        )
        max_speakers = gr.Slider(
            minimum=MIN_SPEAKERS,
            maximum=MAX_SPEAKERS,
            step=1,
            value=default_max_speakers,
            label="Max speakers",
        )
        with gr.Accordion("Advanced diarization runtime", open=False):
            diarization_device = gr.Radio(
                choices=["cpu", "cuda"],
                value=DEFAULT_DIARIZATION_DEVICE,
                label="Diarization device",
            )
    return (
        diarize,
        speaker_count_mode,
        num_speakers,
        min_speakers,
        max_speakers,
        diarization_device,
    )


def _create_task_config_ui() -> tuple[gr.Radio, gr.Textbox, gr.Radio]:
    """Helper function to create task configuration UI components.

    Returns:
        tuple[gr.Radio, gr.Textbox, gr.Radio]: Timestamp type, language, and
        task controls.
    """
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


def _create_file_handling_ui() -> tuple[gr.Checkbox, gr.Textbox]:
    """Helper function to create file handling UI components.

    Returns:
        tuple[gr.Checkbox, gr.Textbox]: Save transcriptions toggle and save
        directory input.
    """
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


def _process_transcription_request_wrapper(
    audio_paths: list[str],
    model_name: str,
    device: str,
    batch_size: int,
    timestamp_type: str,
    language: str,
    task: str,
    dtype: str,
    whisper_chunk_length: int,
    # Stabilization params
    stabilize: bool,
    demucs: bool,
    vad: bool,
    vad_threshold: float,
    # Diarization params
    diarize: bool,
    speaker_count_mode: str,
    num_speakers: int,
    min_speakers: int,
    max_speakers: int,
    diarization_device: str,
    save_transcriptions: bool,
    temp_uploads_dir: str,
    progress: gr.Progress | None = None,
) -> Generator[tuple[object, ...], None, None]:
    """Wrapper to adapt Gradio inputs to process_transcription_request.

    Yields:
        Intermediate and final UI output tuples expected by the Gradio click
        handler.

    Raises:
        RuntimeError: If transcription completes without returning UI outputs.
    """
    if progress is None:
        progress = gr.Progress()

    status_queue: Queue[object] = Queue()
    progress_proxy = _StatusStreamingProgressProxy(progress, status_queue)

    transcription_cfg = TranscriptionConfig(
        model=model_name,
        device=device,
        batch_size=batch_size,
        timestamp_type=cast("Literal['chunk', 'word']", timestamp_type),
        language=language,
        task=cast("Literal['transcribe', 'translate']", task),
        dtype=dtype,
        chunk_length=whisper_chunk_length,
        chunk_duration=None,
        chunk_overlap=None,
    )
    file_handling_cfg = FileHandlingConfig(
        save_transcriptions=save_transcriptions, temp_uploads_dir=temp_uploads_dir
    )
    # Inject stabilization options
    transcription_cfg.stabilize = stabilize
    transcription_cfg.demucs = demucs
    transcription_cfg.vad = vad
    transcription_cfg.vad_threshold = vad_threshold
    # Inject diarization options
    transcription_cfg.diarize = diarize
    if speaker_count_mode == "Exact":
        transcription_cfg.num_speakers = num_speakers if num_speakers > 0 else None
        transcription_cfg.min_speakers = None
        transcription_cfg.max_speakers = None
    elif speaker_count_mode == "Range":
        transcription_cfg.num_speakers = None
        transcription_cfg.min_speakers = min_speakers
        transcription_cfg.max_speakers = max_speakers
    else:
        transcription_cfg.num_speakers = None
        transcription_cfg.min_speakers = None
        transcription_cfg.max_speakers = None
    transcription_cfg.diarization_device = diarization_device

    final_result: tuple[object, ...] | None = None
    final_error: Exception | None = None

    def _run_transcription() -> None:
        """Run transcription in background and signal stream completion."""
        nonlocal final_result, final_error
        try:
            final_result = process_transcription_request(
                audio_paths=audio_paths,
                transcription_config=transcription_cfg,
                file_handling_config=file_handling_cfg,
                progress_tracker=cast(Any, progress_proxy),
            )
        except Exception as exc:  # pragma: no cover - defensive passthrough
            final_error = exc
        finally:
            status_queue.put(_STATUS_STREAM_DONE)

    worker = threading.Thread(target=_run_transcription, daemon=True)
    worker.start()

    while True:
        try:
            queued_status = status_queue.get(timeout=0.1)
        except Empty:
            if final_result is not None or final_error is not None:
                break
            continue

        if queued_status is _STATUS_STREAM_DONE:
            break

        if isinstance(queued_status, str):
            yield (
                queued_status,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )

    worker.join(timeout=1)

    if final_error is not None:
        raise final_error

    if final_result is None:
        raise RuntimeError("Transcription finished without returning UI outputs.")

    yield final_result


def _build_transcription_start_status(audio_paths: list[str]) -> tuple[object, ...]:
    """Build immediate UI updates shown right after button click.

    Args:
        audio_paths: Selected input paths from the upload widget.

    Returns:
        A tuple of UI updates that resets monitor outputs and hides downloads.
    """
    hidden_download_update = gr.update(value=None, visible=False)

    if not audio_paths:
        return (
            "No files selected. Upload at least one file to start transcription.",
            "",
            "",
            None,
            hidden_download_update,
            hidden_download_update,
            hidden_download_update,
            hidden_download_update,
        )

    first_name = audio_paths[0].split("/")[-1]
    start_status: str
    if len(audio_paths) == 1:
        start_status = f"Starting transcription for: {first_name}"
    else:
        start_status = (
            f"Starting transcription for {len(audio_paths)} files (first: {first_name})"
        )

    return (
        start_status,
        "",
        "",
        None,
        hidden_download_update,
        hidden_download_update,
        hidden_download_update,
        hidden_download_update,
    )


def _refresh_speaker_review_controls(
    raw_result: dict[str, Any] | None,
) -> tuple[object, object, object, str]:
    """Refresh speaker review selectors after transcription or edits.

    Args:
        raw_result: Raw transcription state.

    Returns:
        Gradio updates for segment/speaker controls and summary text.
    """
    state = build_speaker_review_state(raw_result)
    speaker_choices = list(state["speaker_map"])
    segment_choices = state["segment_choices"]
    return (
        gr.update(
            choices=segment_choices,
            value=segment_choices[0] if segment_choices else None,
        ),
        gr.update(
            choices=speaker_choices,
            value=speaker_choices[0] if speaker_choices else None,
        ),
        gr.update(
            choices=speaker_choices,
            value=speaker_choices[1] if len(speaker_choices) > 1 else None,
        ),
        build_speaker_review_summary(raw_result),
    )


def create_ui_components(
    *,
    default_model: str = DEFAULT_MODEL,
    default_stabilize: bool = DEFAULT_STABILIZE,
    default_demucs: bool = DEFAULT_DEMUCS,
    default_vad: bool = DEFAULT_VAD,
    default_vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    default_diarize: bool = DEFAULT_DIARIZE,
) -> gr.Blocks:  # pylint: disable=too-many-locals
    """Create and return Gradio UI components with all parameters.

    Returns:
        gr.Blocks: The configured Gradio Blocks interface instance.
    """
    with gr.Blocks(
        title="Insanely Fast Whisper - Local WebUI",
    ) as demo:
        gr.Markdown("# 🎙️ Insanely Fast Whisper - Local WebUI")
        gr.Markdown(
            "Transcribe or translate audio and video files using Whisper models "
            + "directly in your browser."
        )

        with gr.Row(elem_classes=["ifw-shell"]):
            with gr.Column(scale=2):
                # Audio input
                audio_input = gr.File(
                    label="Upload Audio File(s)",
                    type="filepath",
                    file_count="multiple",
                    file_types=list(SUPPORTED_UPLOAD_FORMATS),
                )

                # Model configuration
                model, device, batch_size = _create_model_config_ui(default_model)

                # Processing options
                dtype, chunk_length = _create_processing_options_ui()

                # Timestamp stabilization options
                stabilize_opt, demucs_opt, vad_opt, vad_threshold_opt = (
                    _create_stabilization_ui(
                        default_stabilize=default_stabilize,
                        default_demucs=default_demucs,
                        default_vad=default_vad,
                        default_vad_threshold=default_vad_threshold,
                    )
                )

                # Speaker diarization options
                (
                    diarize_opt,
                    speaker_count_mode_opt,
                    num_speakers_opt,
                    min_speakers_opt,
                    max_speakers_opt,
                    diarization_device_opt,
                ) = _create_diarization_ui(default_diarize=default_diarize)

                # Task configuration
                timestamp_type, language, task = _create_task_config_ui()

                # File handling
                save_transcriptions, temp_uploads_dir = _create_file_handling_ui()

                submit_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=3, elem_classes=["ifw-monitor"]):
                # Outputs
                gr.Markdown("Result Monitor", elem_classes=["ifw-monitor-title"])
                with gr.Column(elem_classes=["ifw-monitor-body"]):
                    status_output = gr.Textbox(
                        label="Status",
                        value="Ready for transcription.",
                        lines=2,
                        max_lines=3,
                        interactive=False,
                        elem_classes=["ifw-status"],
                    )
                    with gr.Tabs():
                        with gr.TabItem("Transcript"):
                            transcription_output = gr.Textbox(
                                label="Transcript Output",
                                lines=13,
                                interactive=False,
                                elem_classes=["ifw-transcript"],
                            )
                        with gr.TabItem("Export JSON"):
                            json_output = gr.Code(
                                label="Saved JSON Export Preview",
                                language="json",
                                lines=16,
                                interactive=False,
                                elem_classes=["ifw-json"],
                            )
                        with gr.TabItem("Speaker Review"):
                            speaker_review_summary = gr.Textbox(
                                label="Speaker map",
                                value="No diarized speaker labels available yet.",
                                lines=5,
                                interactive=False,
                            )
                            review_segment = gr.Dropdown(
                                choices=[],
                                label="Transcript segment",
                                interactive=True,
                            )
                            with gr.Row():
                                review_speaker = gr.Dropdown(
                                    choices=[],
                                    label="Speaker",
                                    interactive=True,
                                )
                                speaker_name = gr.Textbox(
                                    label="Readable name",
                                    placeholder="Host, Guest, Support agent",
                                )
                            with gr.Row():
                                rename_speaker_btn = gr.Button("Rename all")
                                retag_segment_btn = gr.Button("Retag segment")
                                split_segment_btn = gr.Button("Split as new")
                            with gr.Row():
                                merge_source_speaker = gr.Dropdown(
                                    choices=[],
                                    label="Merge from",
                                    interactive=True,
                                )
                                merge_speaker_btn = gr.Button("Merge into selected")

                raw_result_state = gr.State()

                with gr.Group(elem_classes=["ifw-downloads"]):
                    with gr.Row(equal_height=True):
                        download_txt_btn = gr.File(
                            label="Download TXT",
                            visible=False,
                            interactive=False,
                            scale=1,
                            min_width=180,
                        )
                        download_srt_btn = gr.File(
                            label="Download SRT",
                            visible=False,
                            interactive=False,
                            scale=1,
                            min_width=180,
                        )
                        download_json_btn = gr.File(
                            label="Download JSON",
                            visible=False,
                            interactive=False,
                            scale=1,
                            min_width=180,
                        )
                    with gr.Row():
                        download_zip_btn = gr.File(
                            label="Download All as ZIP",
                            visible=False,
                            interactive=False,
                            scale=1,
                        )

        # Event handling
        submit_event = submit_btn.click(
            fn=_build_transcription_start_status,
            inputs=[audio_input],
            outputs=[
                status_output,
                transcription_output,
                json_output,
                raw_result_state,
                download_zip_btn,
                download_txt_btn,
                download_srt_btn,
                download_json_btn,
            ],
            queue=False,
            show_progress="hidden",
        )

        submit_event.then(
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
                chunk_length,
                # Stabilization options (match wrapper order)
                stabilize_opt,
                demucs_opt,
                vad_opt,
                vad_threshold_opt,
                # Diarization options (match wrapper order)
                diarize_opt,
                speaker_count_mode_opt,
                num_speakers_opt,
                min_speakers_opt,
                max_speakers_opt,
                diarization_device_opt,
                save_transcriptions,
                temp_uploads_dir,
            ],
            outputs=[
                status_output,
                transcription_output,
                json_output,
                raw_result_state,
                download_zip_btn,
                download_txt_btn,
                download_srt_btn,
                download_json_btn,
            ],
            api_name="transcribe_audio_v2",
            show_progress="minimal",
            show_progress_on=[transcription_output],
        ).then(
            fn=_refresh_speaker_review_controls,
            inputs=[raw_result_state],
            outputs=[
                review_segment,
                review_speaker,
                merge_source_speaker,
                speaker_review_summary,
            ],
            queue=False,
        ).then(
            fn=refresh_review_downloads,
            inputs=[raw_result_state, temp_uploads_dir, task],
            outputs=[
                download_zip_btn,
                download_txt_btn,
                download_srt_btn,
                download_json_btn,
            ],
            queue=False,
        )

        rename_speaker_btn.click(
            fn=apply_speaker_rename,
            inputs=[raw_result_state, review_speaker, speaker_name],
            outputs=[transcription_output, json_output, raw_result_state],
            queue=False,
        ).then(
            fn=_refresh_speaker_review_controls,
            inputs=[raw_result_state],
            outputs=[
                review_segment,
                review_speaker,
                merge_source_speaker,
                speaker_review_summary,
            ],
            queue=False,
        ).then(
            fn=refresh_review_downloads,
            inputs=[raw_result_state, temp_uploads_dir, task],
            outputs=[
                download_zip_btn,
                download_txt_btn,
                download_srt_btn,
                download_json_btn,
            ],
            queue=False,
        )

        retag_segment_btn.click(
            fn=retag_transcript_segment,
            inputs=[raw_result_state, review_segment, review_speaker],
            outputs=[transcription_output, json_output, raw_result_state],
            queue=False,
        ).then(
            fn=_refresh_speaker_review_controls,
            inputs=[raw_result_state],
            outputs=[
                review_segment,
                review_speaker,
                merge_source_speaker,
                speaker_review_summary,
            ],
            queue=False,
        ).then(
            fn=refresh_review_downloads,
            inputs=[raw_result_state, temp_uploads_dir, task],
            outputs=[
                download_zip_btn,
                download_txt_btn,
                download_srt_btn,
                download_json_btn,
            ],
            queue=False,
        )

        split_segment_btn.click(
            fn=split_review_segment_speaker,
            inputs=[raw_result_state, review_segment, speaker_name],
            outputs=[transcription_output, json_output, raw_result_state],
            queue=False,
        ).then(
            fn=_refresh_speaker_review_controls,
            inputs=[raw_result_state],
            outputs=[
                review_segment,
                review_speaker,
                merge_source_speaker,
                speaker_review_summary,
            ],
            queue=False,
        ).then(
            fn=refresh_review_downloads,
            inputs=[raw_result_state, temp_uploads_dir, task],
            outputs=[
                download_zip_btn,
                download_txt_btn,
                download_srt_btn,
                download_json_btn,
            ],
            queue=False,
        )

        merge_speaker_btn.click(
            fn=merge_review_speakers,
            inputs=[raw_result_state, merge_source_speaker, review_speaker],
            outputs=[transcription_output, json_output, raw_result_state],
            queue=False,
        ).then(
            fn=_refresh_speaker_review_controls,
            inputs=[raw_result_state],
            outputs=[
                review_segment,
                review_speaker,
                merge_source_speaker,
                speaker_review_summary,
            ],
            queue=False,
        ).then(
            fn=refresh_review_downloads,
            inputs=[raw_result_state, temp_uploads_dir, task],
            outputs=[
                download_zip_btn,
                download_txt_btn,
                download_srt_btn,
                download_json_btn,
            ],
            queue=False,
        )

    demo.css = """
        .ifw-shell {
            background: linear-gradient(180deg, #171717 0%, #101010 100%);
        }
        .ifw-monitor {
            border: 1px solid #2b2b2b;
            border-radius: 18px;
            background: linear-gradient(180deg, #181818 0%, #121212 100%);
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
            overflow: hidden;
        }
        .ifw-monitor-body {
            padding: 14px 14px 0 14px;
        }
        .ifw-monitor-title {
            margin: 0;
            padding: 14px 14px 0 14px;
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #ff7a1a;
            opacity: 0.9;
        }
        .ifw-status textarea {
            font-size: 0.82rem;
            line-height: 1.35;
            color: #f3c9a6;
        }
        .ifw-downloads {
            border-top: 1px solid #262626;
            background: rgba(255, 255, 255, 0.02);
            padding: 12px 14px 14px 14px;
        }
        .ifw-downloads .gradio-file {
            min-height: 74px;
        }
        .ifw-monitor .tab-nav {
            padding: 0 2px;
        }
        .ifw-json textarea,
        .ifw-transcript textarea {
            font-size: 0.94rem;
            line-height: 1.55;
        }
        """
    return demo
