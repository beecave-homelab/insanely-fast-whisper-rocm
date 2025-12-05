"""Tests for insanely_fast_whisper_rocm.webui.ui module.

This module contains tests for Gradio UI component creation and layout.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import gradio as gr

from insanely_fast_whisper_rocm.webui.ui import (
    _create_file_handling_ui,
    _create_model_config_ui,
    _create_processing_options_ui,
    _create_stabilization_ui,
    _create_task_config_ui,
    _process_transcription_request_wrapper,
    create_ui_components,
)


class TestCreateModelConfigUI:
    """Test suite for _create_model_config_ui function."""

    def test_create_model_config_ui__returns_three_components(self) -> None:
        """Test that _create_model_config_ui returns model, device, and batch_size."""
        with gr.Blocks():
            model, device, batch_size = _create_model_config_ui()

            assert isinstance(model, gr.Textbox)
            assert isinstance(device, gr.Textbox)
            assert isinstance(batch_size, gr.Slider)

    def test_create_model_config_ui__uses_custom_default_model(self) -> None:
        """Test that _create_model_config_ui uses provided default_model."""
        custom_model = "openai/whisper-large-v3"

        with gr.Blocks():
            model, _, _ = _create_model_config_ui(default_model=custom_model)

            assert model.value == custom_model

    def test_create_model_config_ui__slider_has_correct_range(self) -> None:
        """Test that batch_size slider has correct min/max values."""
        from insanely_fast_whisper_rocm.utils.constants import (
            MAX_BATCH_SIZE,
            MIN_BATCH_SIZE,
        )

        with gr.Blocks():
            _, _, batch_size = _create_model_config_ui()

            assert batch_size.minimum == MIN_BATCH_SIZE
            assert batch_size.maximum == MAX_BATCH_SIZE


class TestCreateProcessingOptionsUI:
    """Test suite for _create_processing_options_ui function."""

    def test_create_processing_options_ui__returns_two_components(self) -> None:
        """Test that _create_processing_options_ui returns dtype and chunk_length."""
        with gr.Blocks():
            dtype, chunk_length = _create_processing_options_ui()

            assert isinstance(dtype, gr.Dropdown)
            assert isinstance(chunk_length, gr.Slider)

    def test_create_processing_options_ui__dtype_has_correct_choices(self) -> None:
        """Test that dtype dropdown has float16 and float32 choices."""
        with gr.Blocks():
            dtype, _ = _create_processing_options_ui()

            choices_values = [
                c[0] if isinstance(c, tuple) else c for c in dtype.choices
            ]
            assert "float16" in choices_values
            assert "float32" in choices_values

    def test_create_processing_options_ui__chunk_length_range(self) -> None:
        """Test that chunk_length slider has correct range."""
        with gr.Blocks():
            _, chunk_length = _create_processing_options_ui()

            assert chunk_length.minimum == 10
            assert chunk_length.maximum == 60
            assert chunk_length.step == 5


class TestCreateStabilizationUI:
    """Test suite for _create_stabilization_ui function."""

    def test_create_stabilization_ui__returns_four_components(self) -> None:
        """Test that _create_stabilization_ui returns all controls."""
        with gr.Blocks():
            stabilize, demucs, vad, vad_threshold = _create_stabilization_ui()

            assert isinstance(stabilize, gr.Checkbox)
            assert isinstance(demucs, gr.Checkbox)
            assert isinstance(vad, gr.Checkbox)
            assert isinstance(vad_threshold, gr.Slider)

    def test_create_stabilization_ui__uses_custom_defaults(self) -> None:
        """Test that _create_stabilization_ui uses provided defaults."""
        with gr.Blocks():
            stabilize, demucs, vad, vad_threshold = _create_stabilization_ui(
                default_stabilize=True,
                default_demucs=True,
                default_vad=True,
                default_vad_threshold=0.5,
            )

            assert stabilize.value is True
            assert demucs.value is True
            assert vad.value is True
            assert vad_threshold.value == 0.5

    def test_create_stabilization_ui__vad_threshold_range(self) -> None:
        """Test that vad_threshold slider has correct range."""
        with gr.Blocks():
            _, _, _, vad_threshold = _create_stabilization_ui()

            assert vad_threshold.minimum == 0.1
            assert vad_threshold.maximum == 0.9
            assert vad_threshold.step == 0.05


class TestCreateTaskConfigUI:
    """Test suite for _create_task_config_ui function."""

    def test_create_task_config_ui__returns_three_components(self) -> None:
        """Test that _create_task_config_ui returns timestamp_type, language, task."""
        with gr.Blocks():
            timestamp_type, language, task = _create_task_config_ui()

            assert isinstance(timestamp_type, gr.Radio)
            assert isinstance(language, gr.Textbox)
            assert isinstance(task, gr.Radio)

    def test_create_task_config_ui__timestamp_type_choices(self) -> None:
        """Test that timestamp_type has chunk and word choices."""
        with gr.Blocks():
            timestamp_type, _, _ = _create_task_config_ui()

            choices_values = [
                c[0] if isinstance(c, tuple) else c for c in timestamp_type.choices
            ]
            assert "chunk" in choices_values
            assert "word" in choices_values

    def test_create_task_config_ui__task_choices(self) -> None:
        """Test that task has transcribe and translate choices."""
        with gr.Blocks():
            _, _, task = _create_task_config_ui()

            choices_values = [c[0] if isinstance(c, tuple) else c for c in task.choices]
            assert "transcribe" in choices_values
            assert "translate" in choices_values


class TestCreateFileHandlingUI:
    """Test suite for _create_file_handling_ui function."""

    def test_create_file_handling_ui__returns_two_components(self) -> None:
        """Test that _create_file_handling_ui returns checkbox and textbox."""
        with gr.Blocks():
            save_transcriptions, temp_uploads_dir = _create_file_handling_ui()

            assert isinstance(save_transcriptions, gr.Checkbox)
            assert isinstance(temp_uploads_dir, gr.Textbox)

    def test_create_file_handling_ui__default_values(self) -> None:
        """Test that _create_file_handling_ui has correct defaults."""
        from insanely_fast_whisper_rocm.utils.constants import DEFAULT_TRANSCRIPTS_DIR

        with gr.Blocks():
            save_transcriptions, temp_uploads_dir = _create_file_handling_ui()

            assert save_transcriptions.value is True
            assert temp_uploads_dir.value == DEFAULT_TRANSCRIPTS_DIR


class TestProcessTranscriptionRequestWrapper:
    """Test suite for _process_transcription_request_wrapper function."""

    @patch("insanely_fast_whisper_rocm.webui.ui.process_transcription_request")
    def test_wrapper__calls_handler_with_correct_config(
        self, mock_process: MagicMock
    ) -> None:
        """Test that wrapper creates correct config objects and calls handler."""
        mock_process.return_value = ("text", {}, {}, Mock(), Mock(), Mock(), Mock())

        _process_transcription_request_wrapper(
            audio_paths=["test.wav"],
            model_name="openai/whisper-tiny",
            device="cpu",
            batch_size=8,
            timestamp_type="word",
            language="en",
            task="transcribe",
            dtype="float16",
            whisper_chunk_length=30,
            stabilize=True,
            demucs=False,
            vad=True,
            vad_threshold=0.35,
            save_transcriptions=True,
            temp_uploads_dir="/tmp/test",
            progress=None,
        )

        # Verify process_transcription_request was called
        assert mock_process.called
        call_args = mock_process.call_args

        # Check transcription config
        transcription_cfg = call_args.kwargs["transcription_config"]
        assert transcription_cfg.model == "openai/whisper-tiny"
        assert transcription_cfg.device == "cpu"
        assert transcription_cfg.batch_size == 8
        assert transcription_cfg.timestamp_type == "word"
        assert transcription_cfg.language == "en"
        assert transcription_cfg.task == "transcribe"
        assert transcription_cfg.dtype == "float16"
        assert transcription_cfg.chunk_length == 30
        assert transcription_cfg.stabilize is True
        assert transcription_cfg.demucs is False
        assert transcription_cfg.vad is True
        assert transcription_cfg.vad_threshold == 0.35

        # Check file handling config
        file_handling_cfg = call_args.kwargs["file_handling_config"]
        assert file_handling_cfg.save_transcriptions is True
        assert file_handling_cfg.temp_uploads_dir == "/tmp/test"

    @patch("insanely_fast_whisper_rocm.webui.ui.process_transcription_request")
    @patch("insanely_fast_whisper_rocm.webui.ui.gr.Progress")
    def test_wrapper__creates_progress_when_none(
        self, mock_progress_cls: MagicMock, mock_process: MagicMock
    ) -> None:
        """Test that wrapper creates Progress instance when progress is None."""
        mock_process.return_value = ("text", {}, {}, Mock(), Mock(), Mock(), Mock())
        mock_progress_instance = Mock()
        mock_progress_cls.return_value = mock_progress_instance

        _process_transcription_request_wrapper(
            audio_paths=["test.wav"],
            model_name="openai/whisper-tiny",
            device="cpu",
            batch_size=8,
            timestamp_type="word",
            language="en",
            task="transcribe",
            dtype="float16",
            whisper_chunk_length=30,
            stabilize=False,
            demucs=False,
            vad=False,
            vad_threshold=0.35,
            save_transcriptions=True,
            temp_uploads_dir="/tmp/test",
            progress=None,
        )

        # Verify Progress was created
        mock_progress_cls.assert_called_once()

    @patch("insanely_fast_whisper_rocm.webui.ui.process_transcription_request")
    def test_wrapper__uses_provided_progress(self, mock_process: MagicMock) -> None:
        """Test that wrapper uses provided progress tracker."""
        mock_process.return_value = ("text", {}, {}, Mock(), Mock(), Mock(), Mock())
        progress_tracker = Mock(spec=gr.Progress)

        _process_transcription_request_wrapper(
            audio_paths=["test.wav"],
            model_name="openai/whisper-tiny",
            device="cpu",
            batch_size=8,
            timestamp_type="word",
            language="en",
            task="transcribe",
            dtype="float16",
            whisper_chunk_length=30,
            stabilize=False,
            demucs=False,
            vad=False,
            vad_threshold=0.35,
            save_transcriptions=True,
            temp_uploads_dir="/tmp/test",
            progress=progress_tracker,
        )

        # Verify the provided progress tracker was used
        call_args = mock_process.call_args
        assert call_args.kwargs["progress_tracker"] is progress_tracker


class TestCreateUIComponents:
    """Test suite for create_ui_components function."""

    def test_create_ui_components__returns_blocks(self) -> None:
        """Test that create_ui_components returns a Gradio Blocks instance."""
        demo = create_ui_components()

        assert isinstance(demo, gr.Blocks)

    def test_create_ui_components__uses_custom_defaults(self) -> None:
        """Test that create_ui_components applies custom default parameters."""
        custom_model = "openai/whisper-large-v3"

        demo = create_ui_components(
            default_model=custom_model,
            default_stabilize=True,
            default_demucs=True,
            default_vad=True,
            default_vad_threshold=0.5,
        )

        assert isinstance(demo, gr.Blocks)

    def test_create_ui_components__has_correct_title(self) -> None:
        """Test that the UI has the correct title."""
        demo = create_ui_components()

        assert demo.title == "Insanely Fast Whisper - Local WebUI"

    @patch("insanely_fast_whisper_rocm.webui.ui._create_model_config_ui")
    @patch("insanely_fast_whisper_rocm.webui.ui._create_processing_options_ui")
    @patch("insanely_fast_whisper_rocm.webui.ui._create_stabilization_ui")
    @patch("insanely_fast_whisper_rocm.webui.ui._create_task_config_ui")
    @patch("insanely_fast_whisper_rocm.webui.ui._create_file_handling_ui")
    def test_create_ui_components__calls_all_ui_helpers(
        self,
        mock_file_handling: MagicMock,
        mock_task_config: MagicMock,
        mock_stabilization: MagicMock,
        mock_processing: MagicMock,
        mock_model_config: MagicMock,
    ) -> None:
        """Test that create_ui_components calls all helper functions."""
        # Setup mocks to return dummy components
        mock_model_config.return_value = (
            gr.Textbox(),
            gr.Textbox(),
            gr.Slider(0, 10),
        )
        mock_processing.return_value = (gr.Dropdown(), gr.Slider(0, 10))
        mock_stabilization.return_value = (
            gr.Checkbox(),
            gr.Checkbox(),
            gr.Checkbox(),
            gr.Slider(0, 1),
        )
        mock_task_config.return_value = (gr.Radio(), gr.Textbox(), gr.Radio())
        mock_file_handling.return_value = (gr.Checkbox(), gr.Textbox())

        create_ui_components(
            default_model="test-model",
            default_stabilize=True,
            default_demucs=False,
            default_vad=True,
            default_vad_threshold=0.4,
        )

        # Verify all helpers were called
        mock_model_config.assert_called_once_with("test-model")
        mock_processing.assert_called_once()
        mock_stabilization.assert_called_once_with(
            default_stabilize=True,
            default_demucs=False,
            default_vad=True,
            default_vad_threshold=0.4,
        )
        mock_task_config.assert_called_once()
        mock_file_handling.assert_called_once()
