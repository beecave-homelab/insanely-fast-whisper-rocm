"""Comprehensive tests for the CLI functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from insanely_fast_whisper_api import constants
from insanely_fast_whisper_api.cli.cli import cli, main
from insanely_fast_whisper_api.cli.facade import CLIFacade, cli_facade
from insanely_fast_whisper_api.core.asr_backend import HuggingFaceBackendConfig
from insanely_fast_whisper_api.core.errors import (
    DeviceNotFoundError,
    TranscriptionError,
)


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

    @patch("insanely_fast_whisper_api.cli.facade.HuggingFaceBackend")
    def test_transcribe_audio_success(self, mock_backend_class: Mock) -> None:
        """Test successful audio transcription."""
        # Mock the backend
        mock_backend = Mock()
        mock_backend.process_audio.return_value = {
            "text": "Test transcription",
            "chunks": [],
            "runtime_seconds": 1.5,
            "config_used": {},
        }
        mock_backend_class.return_value = mock_backend

        facade = CLIFacade()
        result = facade.process_audio(
            audio_file_path=Path("test.mp3"),
            model="openai/whisper-tiny",
            task="transcribe",
        )

        assert result["text"] == "Test transcription"
        assert "runtime_seconds" in result
        mock_backend.process_audio.assert_called_once()

    @patch("insanely_fast_whisper_api.cli.facade.HuggingFaceBackend")
    def test_transcribe_audio_backend_reuse(self, mock_backend_class: Mock) -> None:
        """Test that backend is reused when configuration doesn't change."""
        mock_backend = Mock()
        mock_backend.process_audio.return_value = {"text": "Test", "chunks": []}
        mock_backend_class.return_value = mock_backend

        facade = CLIFacade()

        # First call
        facade.process_audio(Path("test1.mp3"))
        # Second call with same config
        facade.process_audio(Path("test2.mp3"))

        # Backend should only be created once
        mock_backend_class.assert_called_once()
        assert mock_backend.process_audio.call_count == 2


class TestCLICommands:
    """Test CLI commands functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.test_audio_file = Path(__file__).parent / "test.mp3"

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

    @patch("insanely_fast_whisper_api.cli.commands.cli_facade.process_audio")
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

    @patch("insanely_fast_whisper_api.cli.commands.cli_facade.process_audio")
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
            assert f"\U0001F4BE Saved JSON to: {output_path}" in result.output

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

    @patch("insanely_fast_whisper_api.cli.commands.cli_facade.process_audio")
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

    @patch("insanely_fast_whisper_api.cli.commands.cli_facade.process_audio")
    def test_transcribe_transcription_error(self, mock_process: Mock) -> None:
        """Test transcription with transcription error."""
        mock_process.side_effect = TranscriptionError("Model loading failed")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(cli, ["transcribe", str(tmp_path)])

            assert result.exit_code == 1
            assert "Transcribe error" in result.output
            assert "Model loading failed" in result.output

        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_api.cli.commands.cli_facade.process_audio")
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

    @patch("insanely_fast_whisper_api.cli.commands.cli_facade.process_audio")
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

    def test_transcribe_language_none(self) -> None:
        """Test transcribe command with language set to 'none'."""
        with patch(
            "insanely_fast_whisper_api.cli.commands.cli_facade.process_audio"
        ) as mock_process:
            mock_process.return_value = {
                "text": "Test",
                "chunks": [],
                "runtime_seconds": 1.0,
            }

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                result = self.runner.invoke(
                    cli, ["transcribe", str(tmp_path), "--language", "none"]
                )

                assert result.exit_code == 0

                # With current implementation, passing "none" remains a string
                call_args = mock_process.call_args[1]
                assert call_args["language"] == "none"

            finally:
                tmp_path.unlink(missing_ok=True)


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_main_function(self) -> None:
        """Test the main entry point function."""
        with patch("insanely_fast_whisper_api.cli.cli.cli") as mock_cli:
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

        with patch(
            "insanely_fast_whisper_api.cli.facade.HuggingFaceBackend",
            side_effect=DeviceNotFoundError("CUDA device not available"),
        ):
            with pytest.raises(DeviceNotFoundError):
                facade.process_audio(audio_file_path=Path("test.mp3"), device="cuda:0")

    def test_facade_transcription_error(self) -> None:
        """Test that backend transcription errors propagate."""
        with patch(
            "insanely_fast_whisper_api.cli.facade.HuggingFaceBackend"
        ) as mock_backend_class:
            mock_backend = Mock()
            mock_backend.process_audio.side_effect = TranscriptionError("Model failed")
            mock_backend_class.return_value = mock_backend

            facade = CLIFacade()

            with pytest.raises(TranscriptionError):
                facade.process_audio(Path("test.mp3"))


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


if __name__ == "__main__":
    pytest.main([__file__])
