"""Comprehensive tests for the CLI functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from insanely_fast_whisper_api.cli.cli import cli, main
from insanely_fast_whisper_api.cli.facade import CLIFacade, cli_facade
from insanely_fast_whisper_api.core.errors import (
    TranscriptionError,
    DeviceNotFoundError,
)
from insanely_fast_whisper_api.core.asr_backend import HuggingFaceBackendConfig
from insanely_fast_whisper_api import constants


class TestCLIFacade:
    """Test the CLI facade functionality."""

    def test_facade_initialization(self):
        """Test that the facade initializes correctly."""
        facade = CLIFacade()
        assert facade.backend is None
        assert facade._current_config is None

    def test_get_env_config(self):
        """Test environment configuration retrieval."""
        facade = CLIFacade()
        config = facade.get_env_config()

        assert "model" in config
        assert "device" in config
        assert "batch_size" in config
        assert "language" in config
        assert config["model"] == constants.DEFAULT_MODEL
        assert config["batch_size"] == constants.DEFAULT_BATCH_SIZE

    def test_create_backend_config_defaults(self):
        """Test backend configuration creation with defaults."""
        facade = CLIFacade()
        config = facade._create_backend_config()

        assert isinstance(config, HuggingFaceBackendConfig)
        assert config.model_name == constants.DEFAULT_MODEL
        assert config.batch_size == constants.DEFAULT_BATCH_SIZE
        assert config.dtype == "float16"
        assert config.better_transformer is False
        assert config.chunk_length == 30

    def test_create_backend_config_custom(self):
        """Test backend configuration creation with custom parameters."""
        facade = CLIFacade()
        config = facade._create_backend_config(
            model="openai/whisper-large-v3",
            device="cuda:0",
            dtype="float32",
            batch_size=8,
            better_transformer=True,
            chunk_length=15,
        )

        assert config.model_name == "openai/whisper-large-v3"
        assert config.device == "cuda:0"
        assert config.dtype == "float32"
        assert config.batch_size == 8
        assert config.better_transformer is True
        assert config.chunk_length == 15

    def test_create_backend_config_force_cpu(self):
        """Test backend configuration with force_cpu option."""
        facade = CLIFacade()
        config = facade._create_backend_config(
            device="cuda:0", dtype="float16", force_cpu=True
        )

        assert config.device == "cpu"
        assert config.dtype == "float32"  # Should be changed to float32 for CPU

    def test_create_backend_config_cpu_adjustments(self):
        """Test that CPU device gets parameter adjustments."""
        facade = CLIFacade()
        config = facade._create_backend_config(
            device="cpu", chunk_length=30, batch_size=16
        )

        assert config.device == "cpu"
        assert config.chunk_length <= 15  # Should be reduced for CPU
        assert config.batch_size <= 2  # Should be reduced for CPU

    @patch("insanely_fast_whisper_api.cli.facade.HuggingFaceBackend")
    def test_transcribe_audio_success(self, mock_backend_class):
        """Test successful audio transcription."""
        # Mock the backend
        mock_backend = Mock()
        mock_backend.transcribe.return_value = {
            "text": "Test transcription",
            "chunks": [],
            "runtime_seconds": 1.5,
            "config_used": {},
        }
        mock_backend_class.return_value = mock_backend

        facade = CLIFacade()
        result = facade.transcribe_audio(
            audio_file_path=Path("test.mp3"), model="openai/whisper-tiny"
        )

        assert result["text"] == "Test transcription"
        assert "runtime_seconds" in result
        mock_backend.transcribe.assert_called_once()

    @patch("insanely_fast_whisper_api.cli.facade.HuggingFaceBackend")
    def test_transcribe_audio_backend_reuse(self, mock_backend_class):
        """Test that backend is reused when configuration doesn't change."""
        mock_backend = Mock()
        mock_backend.transcribe.return_value = {"text": "Test", "chunks": []}
        mock_backend_class.return_value = mock_backend

        facade = CLIFacade()

        # First call
        facade.transcribe_audio(Path("test1.mp3"))
        # Second call with same config
        facade.transcribe_audio(Path("test2.mp3"))

        # Backend should only be created once
        mock_backend_class.assert_called_once()
        assert mock_backend.transcribe.call_count == 2


class TestCLICommands:
    """Test CLI commands functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.test_audio_file = Path(__file__).parent / "test.mp3"

    def test_cli_group_help(self):
        """Test that CLI group shows help correctly."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Insanely Fast Whisper API" in result.output
        assert "transcribe" in result.output

    def test_cli_version(self):
        """Test CLI version display."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert constants.API_VERSION in result.output

    def test_transcribe_help(self):
        """Test transcribe command help."""
        result = self.runner.invoke(cli, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "Transcribe an audio file" in result.output
        assert "--model" in result.output
        assert "--device" in result.output
        assert "--output" in result.output

    def test_transcribe_missing_file(self):
        """Test transcribe command with missing audio file."""
        result = self.runner.invoke(cli, ["transcribe", "nonexistent.mp3"])
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    @patch("insanely_fast_whisper_api.cli.commands.cli_facade.transcribe_audio")
    def test_transcribe_success(self, mock_transcribe):
        """Test successful transcription command."""
        # Mock successful transcription
        mock_transcribe.return_value = {
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
            assert "Transcription completed!" in result.output
            assert "This is a test transcription." in result.output
            assert "Processing time: 1.23s" in result.output

            # Verify facade was called with correct parameters
            mock_transcribe.assert_called_once()
            call_args = mock_transcribe.call_args
            assert call_args[1]["model"] == "openai/whisper-tiny"
            assert call_args[1]["device"] == "cpu"

        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_api.cli.commands.cli_facade.transcribe_audio")
    def test_transcribe_with_output_file(self, mock_transcribe):
        """Test transcription with output file."""
        mock_transcribe.return_value = {
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
                cli, ["transcribe", str(audio_path), "--output", str(output_path)]
            )

            assert result.exit_code == 0
            assert f"Results saved to: {output_path}" in result.output

            # Verify output file was created and contains expected data
            assert output_path.exists()
            with open(output_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            assert saved_data["transcription"] == "Test output"
            assert "metadata" in saved_data
            assert "timestamp" in saved_data["metadata"]

        finally:
            audio_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_api.cli.commands.cli_facade.transcribe_audio")
    def test_transcribe_device_error(self, mock_transcribe):
        """Test transcription with device error."""
        mock_transcribe.side_effect = DeviceNotFoundError("CUDA device not available")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(
                cli, ["transcribe", str(tmp_path), "--device", "cuda:0"]
            )

            assert result.exit_code == 1
            assert "Device Error" in result.output
            assert "CUDA device not available" in result.output
            assert "Try using --device cpu" in result.output

        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_api.cli.commands.cli_facade.transcribe_audio")
    def test_transcribe_transcription_error(self, mock_transcribe):
        """Test transcription with transcription error."""
        mock_transcribe.side_effect = TranscriptionError("Model loading failed")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(cli, ["transcribe", str(tmp_path)])

            assert result.exit_code == 1
            assert "Transcription Error" in result.output
            assert "Model loading failed" in result.output

        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_api.cli.commands.cli_facade.transcribe_audio")
    def test_transcribe_unexpected_error(self, mock_transcribe):
        """Test transcription with unexpected error."""
        mock_transcribe.side_effect = RuntimeError("Unexpected error")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result = self.runner.invoke(cli, ["transcribe", str(tmp_path)])

            assert result.exit_code == 1
            assert "Unexpected error" in result.output

        finally:
            tmp_path.unlink(missing_ok=True)

    @patch("insanely_fast_whisper_api.cli.commands.cli_facade.transcribe_audio")
    def test_transcribe_all_options(self, mock_transcribe):
        """Test transcribe command with all options."""
        mock_transcribe.return_value = {
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
                    "--better-transformer",
                    "--chunk-length",
                    "20",
                    "--language",
                    "en",
                ],
            )

            assert result.exit_code == 0

            # Verify all parameters were passed correctly
            call_args = mock_transcribe.call_args[1]
            assert call_args["model"] == "openai/whisper-large-v3"
            assert call_args["device"] == "cpu"
            assert call_args["dtype"] == "float32"
            assert call_args["batch_size"] == 4
            assert call_args["better_transformer"] is True
            assert call_args["chunk_length"] == 20
            assert call_args["language"] == "en"

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_transcribe_language_none(self):
        """Test transcribe command with language set to 'none'."""
        with patch(
            "insanely_fast_whisper_api.cli.commands.cli_facade.transcribe_audio"
        ) as mock_transcribe:
            mock_transcribe.return_value = {
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

                # Verify language was set to None
                call_args = mock_transcribe.call_args[1]
                assert call_args["language"] is None

            finally:
                tmp_path.unlink(missing_ok=True)


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_main_function(self):
        """Test the main entry point function."""
        with patch("insanely_fast_whisper_api.cli.cli.cli") as mock_cli:
            main()
            mock_cli.assert_called_once()

    def test_global_facade_instance(self):
        """Test that the global facade instance is properly initialized."""
        assert isinstance(cli_facade, CLIFacade)
        assert cli_facade.backend is None
        assert cli_facade._current_config is None

    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path(__file__).parent.joinpath("test.mp3").exists(),
        reason="Test audio file not available",
    )
    def test_real_transcription(self):
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

    def test_facade_device_validation(self):
        """Test device validation in facade."""
        facade = CLIFacade()

        # Test with invalid device should raise error when backend is created
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(DeviceNotFoundError):
                facade.transcribe_audio(
                    audio_file_path=Path("test.mp3"), device="cuda:0"
                )

    def test_facade_transcription_error(self):
        """Test transcription error handling in facade."""
        with patch(
            "insanely_fast_whisper_api.cli.facade.HuggingFaceBackend"
        ) as mock_backend_class:
            mock_backend = Mock()
            mock_backend.transcribe.side_effect = TranscriptionError("Model failed")
            mock_backend_class.return_value = mock_backend

            facade = CLIFacade()

            with pytest.raises(TranscriptionError):
                facade.transcribe_audio(Path("test.mp3"))


class TestBackwardCompatibility:
    """Test backward compatibility of CLI interface."""

    def test_command_structure_preserved(self):
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
            "--better-transformer",
            "--chunk-length",
            "--language",
            "--output",
        ]

        for option in expected_options:
            assert option in help_output, f"Option {option} missing from help"

    def test_option_defaults_preserved(self):
        """Test that option defaults match the original implementation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])

        # Check that defaults are shown and match constants
        assert str(constants.DEFAULT_MODEL) in result.output
        assert str(constants.DEFAULT_DEVICE) in result.output
        assert str(constants.DEFAULT_BATCH_SIZE) in result.output

    def test_error_messages_consistent(self):
        """Test that error messages are consistent with original."""
        runner = CliRunner()

        # Test missing file error
        result = runner.invoke(cli, ["transcribe", "nonexistent.mp3"])
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()


if __name__ == "__main__":
    pytest.main([__file__])
