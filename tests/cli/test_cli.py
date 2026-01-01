"""Comprehensive tests for the CLI functionality."""

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from insanely_fast_whisper_rocm import constants
from insanely_fast_whisper_rocm.cli.cli import cli, main
from insanely_fast_whisper_rocm.cli.facade import CLIFacade, cli_facade
from insanely_fast_whisper_rocm.core.asr_backend import HuggingFaceBackendConfig
from insanely_fast_whisper_rocm.core.errors import (
    DeviceNotFoundError,
    TranscriptionError,
)
from insanely_fast_whisper_rocm.core.progress import ProgressCallback


class TestCLIFacade:
    """Test the CLI facade functionality."""

    def test_facade_initialization(self) -> None:
        """Test that the facade initializes correctly."""
        facade = CLIFacade()
        assert facade.backend is None
        assert facade._current_config is None

    def test_get_env_config(self) -> None:
        """Test environment configuration retrieval."""
        facade = CLIFacade()
        config = facade.get_env_config()

        assert "model" in config
        assert "device" in config
        assert "batch_size" in config
        assert "language" in config
        assert config["model"] == constants.DEFAULT_MODEL
        assert config["batch_size"] == constants.DEFAULT_BATCH_SIZE

    def test_create_backend_config_defaults(self) -> None:
        """Test backend configuration creation with defaults."""
        facade = CLIFacade()
        config = facade._create_backend_config(
            model=constants.DEFAULT_MODEL,
            device=constants.DEFAULT_DEVICE,
            dtype="float16",
            batch_size=constants.DEFAULT_BATCH_SIZE,
            chunk_length=constants.DEFAULT_CHUNK_LENGTH,
            progress_group_size=constants.DEFAULT_PROGRESS_GROUP_SIZE,
        )

        assert isinstance(config, HuggingFaceBackendConfig)
        assert config.model_name == constants.DEFAULT_MODEL
        assert config.batch_size == constants.DEFAULT_BATCH_SIZE
        assert config.dtype == "float16"
        assert config.chunk_length == 30

    def test_create_backend_config_custom(self) -> None:
        """Test backend configuration creation with custom parameters."""
        facade = CLIFacade()
        config = facade._create_backend_config(
            model="openai/whisper-large-v3",
            device="cuda:0",
            dtype="float32",
            batch_size=8,
            chunk_length=15,
            progress_group_size=constants.DEFAULT_PROGRESS_GROUP_SIZE,
        )

        assert config.model_name == "openai/whisper-large-v3"
        assert config.device == "cuda:0"
        assert config.dtype == "float32"
        assert config.batch_size == 8
        assert config.chunk_length == 15
        assert config.progress_group_size == constants.DEFAULT_PROGRESS_GROUP_SIZE

    def test_create_backend_config_force_cpu(self) -> None:
        """Test backend configuration with force_cpu option."""
        facade = CLIFacade()
        config = facade._create_backend_config(
            model=constants.DEFAULT_MODEL,
            device="cuda:0",
            dtype="float16",
            batch_size=constants.DEFAULT_BATCH_SIZE,
            chunk_length=constants.DEFAULT_CHUNK_LENGTH,
            progress_group_size=constants.DEFAULT_PROGRESS_GROUP_SIZE,
        )

        # In current implementation, CPU normalization is applied at processing time,
        # not at config creation. Ensure fields are present.
        assert isinstance(config.model_name, str)
        assert isinstance(config.chunk_length, int)

    def test_create_backend_config_cpu_adjustments(self) -> None:
        """Test that CPU device gets parameter adjustments."""
        facade = CLIFacade()
        config = facade._create_backend_config(
            model=constants.DEFAULT_MODEL,
            device="cpu",
            dtype="float16",
            batch_size=16,
            chunk_length=30,
            progress_group_size=constants.DEFAULT_PROGRESS_GROUP_SIZE,
        )

        assert config.device == "cpu"
        # Adjustments are applied during processing, not at config creation
        assert isinstance(config.batch_size, int)

    @patch.object(CLIFacade, "orchestrator_factory")
    def test_transcribe_audio_success(self, mock_orchestrator_factory: Mock) -> None:
        """Test successful audio transcription."""
        # Mock the orchestrator
        mock_orch = Mock()
        mock_orch.run_transcription.return_value = {
            "text": "Test transcription",
            "chunks": [],
            "runtime_seconds": 1.5,
            "config_used": {},
        }
        mock_orchestrator_factory.return_value = mock_orch

        facade = CLIFacade()
        result = facade.process_audio(
            audio_file_path=Path("test.mp3"),
            model="openai/whisper-tiny",
            task="transcribe",
        )

        assert result["text"] == "Test transcription"
        assert "runtime_seconds" in result
        mock_orch.run_transcription.assert_called_once()

    @patch.object(CLIFacade, "orchestrator_factory")
    def test_transcribe_audio_backend_reuse(
        self, mock_orchestrator_factory: Mock
    ) -> None:
        """Test that orchestrator is used for repeated calls."""
        mock_orch = Mock()
        mock_orch.run_transcription.return_value = {"text": "Test", "chunks": []}
        mock_orchestrator_factory.return_value = mock_orch

        facade = CLIFacade()

        # First call
        facade.process_audio(Path("test1.mp3"))
        # Second call with same config
        facade.process_audio(Path("test2.mp3"))

        # Orchestrator factory should be called each time process_audio is called
        assert mock_orchestrator_factory.call_count == 2
        assert mock_orch.run_transcription.call_count == 2


@pytest.fixture(autouse=True)
def _stub_cli_facade(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Install a lightweight CLI facade for tests.

    Args:
        monkeypatch: Pytest helper providing attribute overrides.

    Yields:
        None: Allows callers to execute with the stub facade active.

    """

    class _StubFacade(CLIFacade):
        def __init__(self) -> None:
            super().__init__(check_file_exists=False)

        def process_audio(
            self,
            audio_file_path: Path,
            model: str | None = None,
            device: str | None = None,
            dtype: str = "float16",
            batch_size: int | None = None,
            chunk_length: int = 30,
            progress_group_size: int | None = None,
            language: str | None = None,
            task: str = "transcribe",
            return_timestamps_value: bool | str = True,
            progress_cb: ProgressCallback | None = None,
        ) -> dict[str, Any]:
            del device, dtype, batch_size, chunk_length, progress_group_size
            del language, task, return_timestamps_value, progress_cb

            return {
                "text": "stub transcription",
                "chunks": [],
                "runtime_seconds": 1.0,
                "config_used": {"model": model or "stub-model"},
                "audio_file_path": str(audio_file_path),
            }

    stub = _StubFacade()
    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.cli.commands.cli_facade", stub, raising=False
    )
    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.cli.facade.cli_facade", stub, raising=False
    )
    yield


class TestCLICommands:
    """Test CLI commands functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.test_audio_file = (
            Path(__file__).parents[2] / "tests" / "audio" / "fixtures" / "test_clip.mp3"
        )

    def test_cli_group_help(self) -> None:
        """Test that CLI group shows help correctly."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Insanely Fast Whisper API" in result.output
        assert "transcribe" in result.output

    def test_cli_version(self) -> None:
        """Test CLI version display."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert constants.API_VERSION in result.output

    def test_transcribe_help(self) -> None:
        """Test transcribe command help."""
        result = self.runner.invoke(cli, ["transcribe", "--help"])
        assert result.exit_code == 0
        # Help text should include command name and core options
        assert "transcribe" in result.output
        assert "--model" in result.output
        assert "--device" in result.output
        assert "--output" in result.output

    def test_transcribe_missing_file(self) -> None:
        """Test transcribe command with missing audio file."""
        result = self.runner.invoke(cli, ["transcribe", "nonexistent.mp3"])
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_transcribe_success(self, mock_process: Mock) -> None:
        """Test successful transcription command."""
        # Mock successful transcription
        mock_process.return_value = {
            "text": "This is a test transcription.",
            "chunks": [
                {"text": "This is a test transcription.", "timestamp": [0.0, 2.0]}
            ],
            "runtime_seconds": 1.23,
            "config_used": {"model": "openai/whisper-tiny"},
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli,
                [
                    "transcribe",
                    str(tmp_path),
                    "--model",
                    "openai/whisper-tiny",
                    "--device",
                    "cpu",
                ],
            )

            assert result.exit_code == 0
            # We don't assert exact messaging; rely on exit code and file export tests
            assert result.exit_code == 0

            # Verify facade was called with correct parameters
            mock_process.assert_called_once()
            call_args = mock_process.call_args
            assert call_args[1]["model"] == "openai/whisper-tiny"
            assert call_args[1]["device"] == "cpu"

        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_transcribe_with_output_file(self, mock_process: Mock) -> None:
        """Test transcription with output file."""
        mock_process.return_value = {
            "text": "Test output",
            "chunks": [],
            "runtime_seconds": 1.0,
            "config_used": {},
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
            audio_path = Path(audio_file.name)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_file:
            output_path = Path(output_file.name)

        try:
            result = self.runner.invoke(
                cli,
                [
                    "transcribe",
                    str(audio_path),
                    "--output",
                    str(output_path),
                    "--export-format",
                    "json",
                    "--no-stabilize",
                ],
            )

            assert result.exit_code == 0
            assert f"\U0001f4be Saved JSON to: {output_path}" in result.output

            # Verify output file was created and contains expected data
            assert output_path.exists()
            with open(output_path, encoding="utf-8") as f:
                saved_data = json.load(f)

            assert saved_data["text"] == "Test output"
            assert saved_data["transcribe"] == "Test output"
            assert saved_data["metadata"]["config_used"] == {}
            assert "timestamp" in saved_data["metadata"]

        finally:
            audio_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_transcribe_device_error(self, mock_process: Mock) -> None:
        """Test transcription with device error."""
        mock_process.side_effect = DeviceNotFoundError("CUDA device not available")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli, ["transcribe", str(tmp_path), "--device", "cuda:0"]
            )

            assert result.exit_code == 1
            assert "Device error" in result.output
            assert "CUDA device not available" in result.output
            assert "--device cpu" in result.output

        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_transcribe_transcription_error(self, mock_process: Mock) -> None:
        """Test transcription with transcription error."""
        mock_process.side_effect = TranscriptionError("Model loading failed")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(cli, ["transcribe", str(tmp_path)])

            assert result.exit_code == 1
            assert "Transcription error" in result.output
            assert "Model loading failed" in result.output

        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_transcribe_unexpected_error(self, mock_process: Mock) -> None:
        """Test transcription with unexpected error."""
        mock_process.side_effect = RuntimeError("Unexpected error")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(cli, ["transcribe", str(tmp_path)])

            assert result.exit_code == 1
            assert "Unexpected error" in result.output

        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_transcribe_all_options(self, mock_process: Mock) -> None:
        """Test transcribe command with all options."""
        mock_process.return_value = {
            "text": "Full options test",
            "chunks": [],
            "runtime_seconds": 2.5,
            "config_used": {},
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli,
                [
                    "transcribe",
                    str(tmp_path),
                    "--model",
                    "openai/whisper-large-v3",
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--batch-size",
                    "4",
                    "--progress-group-size",
                    "8",
                    "--chunk-length",
                    "20",
                    "--language",
                    "en",
                ],
            )

            assert result.exit_code == 0

            # Verify key parameters were passed correctly
            call_args = mock_process.call_args[1]
            assert call_args["model"] == "openai/whisper-large-v3"
            assert call_args["device"] == "cpu"
            assert call_args["dtype"] == "float32"
            assert call_args["batch_size"] == 4
            assert call_args.get("chunk_length") == 20
            assert call_args["progress_group_size"] == 8
            assert call_args["language"] == "en"

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_translate_help(self) -> None:
        """Test translate command help."""
        result = self.runner.invoke(cli, ["translate", "--help"])
        assert result.exit_code == 0
        # Help text should include command name and core options
        assert "translate" in result.output
        assert "--model" in result.output
        assert "--device" in result.output
        assert "--output" in result.output

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_translate_success(self, mock_process: Mock) -> None:
        """Test successful translation command."""
        # Mock successful translation
        mock_process.return_value = {
            "text": "This is a test translation.",
            "chunks": [
                {"text": "This is a test translation.", "timestamp": [0.0, 2.0]}
            ],
            "runtime_seconds": 1.23,
            "config_used": {"model": "openai/whisper-tiny"},
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli,
                [
                    "translate",
                    str(tmp_path),
                    "--model",
                    "openai/whisper-tiny",
                    "--device",
                    "cpu",
                ],
            )

            assert result.exit_code == 0

            # Verify facade was called with correct parameters
            mock_process.assert_called_once()
            call_args = mock_process.call_args
            assert call_args[1]["model"] == "openai/whisper-tiny"
            assert call_args[1]["device"] == "cpu"
            assert call_args[1]["task"] == "translate"

        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_transcribe_legacy_export_json(self, mock_process: Mock) -> None:
        """Test legacy --export-json flag."""
        mock_process.return_value = {
            "text": "Test",
            "chunks": [],
            "runtime_seconds": 1.0,
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli, ["transcribe", str(tmp_path), "--export-json"]
            )
            assert result.exit_code == 0
            # Legacy flag should set export format to json
            # The actual verification happens in _handle_output_and_benchmarks
            mock_process.assert_called_once()
            # Just verify the command completed successfully
            assert result.exit_code == 0
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_transcribe_legacy_export_txt(self, mock_process: Mock) -> None:
        """Test legacy --export-txt flag."""
        mock_process.return_value = {
            "text": "Test",
            "chunks": [],
            "runtime_seconds": 1.0,
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli, ["transcribe", str(tmp_path), "--export-txt"]
            )
            assert result.exit_code == 0
            mock_process.assert_called_once()
            assert result.exit_code == 0
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_transcribe_legacy_export_srt(self, mock_process: Mock) -> None:
        """Test legacy --export-srt flag."""
        mock_process.return_value = {
            "text": "Test",
            "chunks": [],
            "runtime_seconds": 1.0,
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli, ["transcribe", str(tmp_path), "--export-srt"]
            )
            assert result.exit_code == 0
            mock_process.assert_called_once()
            assert result.exit_code == 0
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_transcribe_legacy_export_all(self, mock_process: Mock) -> None:
        """Test legacy --export-all flag."""
        mock_process.return_value = {
            "text": "Test",
            "chunks": [],
            "runtime_seconds": 1.0,
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli, ["transcribe", str(tmp_path), "--export-all"]
            )
            assert result.exit_code == 0
            mock_process.assert_called_once()
            assert result.exit_code == 0
        finally:
            tmp_path.unlink(missing_ok=True)


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_main_function(self) -> None:
        """Test the main entry point function."""
        with patch("insanely_fast_whisper_rocm.cli.cli.cli") as mock_cli:
            main()
            mock_cli.assert_called_once()

    def test_global_facade_instance(self) -> None:
        """Test that the global facade instance is properly initialized."""
        assert isinstance(cli_facade, CLIFacade)
        assert cli_facade.backend is None
        assert cli_facade._current_config is None

    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path(__file__).parent.joinpath("test.mp3").exists(),
        reason="Test audio file not available",
    )
    def test_real_transcription(self) -> None:
        """Integration test with real audio file (if available)."""
        runner = CliRunner()
        test_audio = Path(__file__).parent / "test.mp3"

        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(test_audio),
                "--model",
                "openai/whisper-tiny",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ],
        )

        # Should complete successfully (though may take time)
        assert result.exit_code == 0
        assert "Transcription completed!" in result.output
        assert len(result.output.strip()) > 0


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_facade_device_validation(self) -> None:
        """Test that device errors from the backend propagate."""
        facade = CLIFacade()
        test_audio = (
            Path(__file__).parents[2] / "tests" / "audio" / "fixtures" / "test_clip.mp3"
        )

        mock_orch = Mock()
        mock_orch.run_transcription.side_effect = DeviceNotFoundError(
            "CUDA device not available"
        )

        # We must patch CLIFacade's orchestrator_factory since it's used in
        # process_audio
        with patch.object(
            facade,
            "orchestrator_factory",
            return_value=mock_orch,
        ):
            with pytest.raises(TranscriptionError) as exc_info:
                facade.process_audio(audio_file_path=test_audio, device="cuda:0")
            assert "CUDA device not available" in str(exc_info.value)

    def test_facade_transcription_error(self) -> None:
        """Test that orchestrator transcription errors propagate."""
        mock_orch = Mock()
        mock_orch.run_transcription.side_effect = TranscriptionError("Model failed")
        test_audio = (
            Path(__file__).parents[2] / "tests" / "audio" / "fixtures" / "test_clip.mp3"
        )

        facade = CLIFacade()
        with patch.object(
            facade,
            "orchestrator_factory",
            return_value=mock_orch,
        ):
            with pytest.raises(TranscriptionError):
                facade.process_audio(test_audio)

    def test_cpu_parameter_adjustments(self) -> None:
        """Test that CPU device gets parameter adjustments for better stability."""
        mock_orch = Mock()
        mock_orch.run_transcription.return_value = {"text": "CPU test", "chunks": []}
        test_audio = (
            Path(__file__).parents[2] / "tests" / "audio" / "fixtures" / "test_clip.mp3"
        )

        facade = CLIFacade()
        with patch.object(
            facade,
            "orchestrator_factory",
            return_value=mock_orch,
        ):
            # Test with CPU device - should adjust chunk_length and batch_size
            facade.process_audio(
                audio_file_path=test_audio,
                device="cpu",
                chunk_length=30,  # Should be reduced to 15 for CPU
                batch_size=8,  # Should be reduced to 2 for CPU
            )

            # Verify orchestrator was called with adjusted parameters
            mock_orch.run_transcription.assert_called_once()
            call_args = mock_orch.run_transcription.call_args[1]
            config = call_args["backend_config"]
            assert config.chunk_length == 15  # Reduced from 30
            assert config.batch_size == 2  # Reduced from 8

    def test_cpu_parameter_adjustments_min_values(self) -> None:
        """Test CPU parameter adjustments respect maximum values for stability."""
        mock_orch = Mock()
        mock_orch.run_transcription.return_value = {"text": "CPU test", "chunks": []}
        test_audio = (
            Path(__file__).parents[2] / "tests" / "audio" / "fixtures" / "test_clip.mp3"
        )

        facade = CLIFacade()
        with patch.object(
            facade,
            "orchestrator_factory",
            return_value=mock_orch,
        ):
            # Test with values larger than CPU limits - should be reduced
            facade.process_audio(
                audio_file_path=test_audio,
                device="cpu",
                chunk_length=20,  # Should be reduced to 15 for CPU
                batch_size=4,  # Should be reduced to 2 for CPU
            )

            # Verify orchestrator was called with adjusted parameters
            mock_orch.run_transcription.assert_called_once()
            call_args = mock_orch.run_transcription.call_args[1]
            config = call_args["backend_config"]
            assert config.chunk_length == 15  # Reduced from 20 to max 15 for CPU
            assert config.batch_size == 2  # Reduced from 4 to max 2 for CPU


class TestBackwardCompatibility:
    """Test backward compatibility of CLI interface."""

    def test_command_structure_preserved(self) -> None:
        """Test that the command structure matches the original."""
        runner = CliRunner()

        # Test that transcribe command exists
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert result.exit_code == 0

        # Test that all original options are available
        help_output = result.output
        expected_options = [
            "--model",
            "--device",
            "--dtype",
            "--batch-size",
            "--progress-group-size",
            "--chunk-length",
            "--language",
            "--output",
            "--export-format",
        ]

        for option in expected_options:
            assert option in help_output, f"Option {option} missing from help"

    def test_option_defaults_preserved(self) -> None:
        """Test that option defaults match the original implementation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])

        # Check that defaults are shown and match constants
        normalized_output = "".join(result.output.split())
        assert "".join(str(constants.DEFAULT_MODEL).split()) in normalized_output
        assert "".join(str(constants.DEFAULT_DEVICE).split()) in normalized_output
        assert "".join(str(constants.DEFAULT_BATCH_SIZE).split()) in normalized_output

    def test_error_messages_consistent(self) -> None:
        """Test that error messages are consistent with original."""
        runner = CliRunner()

        # Test missing file error
        result = runner.invoke(cli, ["transcribe", "nonexistent.mp3"])
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()


class TestVideoProcessing:
    """Test video file processing functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("insanely_fast_whisper_rocm.cli.commands.extract_audio_from_video")
    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_video_file_extraction(
        self, mock_process: Mock, mock_extract: Mock
    ) -> None:
        """Test that video files are extracted to audio before processing."""
        # Mock successful video extraction
        mock_extract.return_value = Path("extracted_audio.wav")
        mock_process.return_value = {
            "text": "Video transcription",
            "chunks": [],
            "runtime_seconds": 2.0,
        }

        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli, ["transcribe", str(tmp_path), "--device", "cpu"]
            )
            assert result.exit_code == 0

            # Verify video extraction was called
            mock_extract.assert_called_once_with(video_path=tmp_path)

            # Verify audio processing was called with extracted audio
            mock_process.assert_called_once()
            call_args = mock_process.call_args[1]
            assert call_args["audio_file_path"] == Path("extracted_audio.wav")

        finally:
            tmp_path.unlink(missing_ok=True)
            # Clean up the extracted audio file mock
            if Path("extracted_audio.wav").exists():
                Path("extracted_audio.wav").unlink(missing_ok=True)


class TestQuietMode:
    """Test quiet mode functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_quiet_mode_logging_reduction(self, mock_process: Mock) -> None:
        """Test that quiet mode reduces logging verbosity."""
        mock_process.return_value = {
            "text": "Test",
            "chunks": [],
            "runtime_seconds": 1.0,
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(cli, ["transcribe", str(tmp_path), "--quiet"])
            assert result.exit_code == 0
            # Quiet mode should complete successfully
            mock_process.assert_called_once()
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.stabilize_timestamps")
    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_stabilize_quiet_mode_suppress_output(
        self, mock_process: Mock, mock_stabilize: Mock
    ) -> None:
        """Test that stabilize in quiet mode suppresses output."""
        mock_process.return_value = {
            "text": "Test",
            "chunks": [],
            "runtime_seconds": 1.0,
        }
        mock_stabilize.return_value = {"text": "Test", "chunks": [], "stabilized": True}

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli, ["transcribe", str(tmp_path), "--stabilize", "--quiet"]
            )
            assert result.exit_code == 0
            mock_stabilize.assert_called_once()
        finally:
            tmp_path.unlink(missing_ok=True)


class TestStableTS:
    """Test stable-ts post-processing functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("insanely_fast_whisper_rocm.cli.commands.stabilize_timestamps")
    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_stabilize_timestamps_basic(
        self, mock_process: Mock, mock_stabilize: Mock
    ) -> None:
        """Test basic timestamp stabilization."""
        mock_process.return_value = {
            "text": "Test",
            "chunks": [],
            "runtime_seconds": 1.0,
        }
        mock_stabilize.return_value = {"text": "Test", "chunks": [], "stabilized": True}

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli, ["transcribe", str(tmp_path), "--stabilize"]
            )
            assert result.exit_code == 0
            mock_stabilize.assert_called_once()
            call_args = mock_stabilize.call_args
            assert call_args[0][0]["text"] == "Test"
            assert call_args[1]["demucs"] is constants.DEFAULT_DEMUCS
            assert call_args[1]["vad"] is constants.DEFAULT_VAD
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.stabilize_timestamps")
    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_stabilize_timestamps_with_demucs(
        self, mock_process: Mock, mock_stabilize: Mock
    ) -> None:
        """Test timestamp stabilization with demucs."""
        mock_process.return_value = {
            "text": "Test",
            "chunks": [],
            "runtime_seconds": 1.0,
        }
        mock_stabilize.return_value = {"text": "Test", "chunks": [], "stabilized": True}

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli, ["transcribe", str(tmp_path), "--stabilize", "--demucs"]
            )
            assert result.exit_code == 0
            mock_stabilize.assert_called_once()
            call_args = mock_stabilize.call_args
            assert call_args[1]["demucs"] is True
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.stabilize_timestamps")
    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_stabilize_timestamps_with_vad(
        self, mock_process: Mock, mock_stabilize: Mock
    ) -> None:
        """Test timestamp stabilization with VAD."""
        mock_process.return_value = {
            "text": "Test",
            "chunks": [],
            "runtime_seconds": 1.0,
        }
        mock_stabilize.return_value = {"text": "Test", "chunks": [], "stabilized": True}

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli,
                [
                    "transcribe",
                    str(tmp_path),
                    "--stabilize",
                    "--vad",
                    "--vad-threshold",
                    "0.5",
                ],
            )
            assert result.exit_code == 0
            mock_stabilize.assert_called_once()
            call_args = mock_stabilize.call_args
            assert call_args[1]["vad"] is True
            assert call_args[1]["vad_threshold"] == 0.5
        finally:
            tmp_path.unlink(missing_ok=True)


class TestDebugErrorHandling:
    """Test debug mode error logging."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_device_error_debug_logging(self, mock_process: Mock) -> None:
        """Test device error logging in debug mode."""
        from insanely_fast_whisper_rocm.cli.errors import DeviceNotFoundError

        mock_process.side_effect = DeviceNotFoundError("CUDA device not available")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(cli, ["transcribe", str(tmp_path), "--debug"])
            assert result.exit_code == 1
            # Should exit with error code 1
            assert "Device error" in result.output
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_transcription_error_debug_logging(self, mock_process: Mock) -> None:
        """Test transcription error logging in debug mode."""
        from insanely_fast_whisper_rocm.cli.errors import TranscriptionError

        mock_process.side_effect = TranscriptionError("Model failed")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(cli, ["transcribe", str(tmp_path), "--debug"])
            assert result.exit_code == 1
            assert "Transcription error" in result.output
        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    def test_unexpected_error_debug_logging(self, mock_process: Mock) -> None:
        """Test unexpected error logging in debug mode."""
        mock_process.side_effect = Exception("Unexpected error")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(cli, ["transcribe", str(tmp_path), "--debug"])
            assert result.exit_code == 1
            assert "Unexpected error" in result.output
        finally:
            tmp_path.unlink(missing_ok=True)


class TestBenchmarkCollection:
    """Test benchmark collection functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("insanely_fast_whisper_rocm.cli.commands.cli_facade.process_audio")
    @patch("insanely_fast_whisper_rocm.benchmarks.collector.BenchmarkCollector")
    def test_benchmark_collection_with_extra(
        self, mock_collector_class: Mock, mock_process: Mock
    ) -> None:
        """Test benchmark collection with extra metadata."""
        mock_process.return_value = {
            "text": "Test",
            "chunks": [],
            "runtime_seconds": 1.0,
            "config_used": {},
        }
        mock_collector = Mock()
        mock_collector.collect.return_value = Path("benchmark.json")
        mock_collector_class.return_value = mock_collector

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli,
                [
                    "transcribe",
                    str(tmp_path),
                    "--benchmark",
                    "--benchmark-extra",
                    "key1=value1",
                    "--benchmark-extra",
                    "key2=value2",
                ],
            )
            assert result.exit_code == 0
            mock_collector.collect.assert_called_once()
            call_args = mock_collector.collect.call_args
            # Check that extra dict was parsed correctly
            assert call_args[1]["extra"]["key1"] == "value1"
            assert call_args[1]["extra"]["key2"] == "value2"
            assert "Benchmark saved" in result.output
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])
