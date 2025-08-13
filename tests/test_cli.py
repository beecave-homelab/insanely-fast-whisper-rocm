"""Tests for CLI facade and command integration."""

from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from insanely_fast_whisper_api.cli.cli import cli, main
from insanely_fast_whisper_api.cli.facade import CLIFacade, cli_facade
from insanely_fast_whisper_api.core.asr_backend import (
    CTranslate2BackendConfig,
    HuggingFaceBackendConfig,
)


class TestCLIFacade:
    """Unit tests for the CLI facade."""

    def test_create_backend_config(self) -> None:
        """Facade returns correct backend configurations."""
        facade = CLIFacade()

        hf_cfg = facade._create_backend_config(
            model="m",
            device="cpu",
            dtype="float32",
            batch_size=1,
            chunk_length=30,
            backend="huggingface",
        )
        assert isinstance(hf_cfg, HuggingFaceBackendConfig)

        ct2_cfg = facade._create_backend_config(
            model="m",
            device="cpu",
            dtype="float16",
            batch_size=1,
            chunk_length=30,
            backend="ctranslate2",
        )
        assert isinstance(ct2_cfg, CTranslate2BackendConfig)

    @patch("insanely_fast_whisper_api.cli.facade.HuggingFaceBackend")
    def test_process_audio_huggingface(self, mock_backend_cls) -> None:
        """process_audio uses HuggingFace backend when requested."""
        backend = Mock()
        backend.process_audio.return_value = {"text": "ok"}
        mock_backend_cls.return_value = backend

        facade = CLIFacade()
        facade.process_audio(Path("a.wav"), backend="huggingface")

        mock_backend_cls.assert_called_once()
        backend.process_audio.assert_called_once()

    @patch("insanely_fast_whisper_api.cli.facade.CTranslate2Backend")
    def test_process_audio_ctranslate2(self, mock_backend_cls) -> None:
        """process_audio uses CTranslate2 backend when requested."""
        backend = Mock()
        backend.process_audio.return_value = {"text": "ok"}
        mock_backend_cls.return_value = backend

        facade = CLIFacade()
        facade.process_audio(Path("a.wav"), backend="ctranslate2")

        mock_backend_cls.assert_called_once()
        backend.process_audio.assert_called_once()


class TestCLICommands:
    """High level CLI command tests."""

    def setup_method(self) -> None:  # pylint: disable=no-self-use
        self.runner = CliRunner()
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        self.test_audio_file = Path(tmp.name)
        tmp.close()

    def test_cli_help(self) -> None:
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "transcribe" in result.output

    def test_transcribe_invokes_facade(self) -> None:
        with patch(
            "insanely_fast_whisper_api.cli.commands.cli_facade.process_audio"
        ) as mock_process:
            mock_process.return_value = {
                "text": "hello",
                "chunks": [],
                "runtime_seconds": 1.0,
                "config_used": {},
            }

            result = self.runner.invoke(
                cli,
                [
                    "transcribe",
                    str(self.test_audio_file),
                    "--backend",
                    "ctranslate2",
                ],
            )

            assert result.exit_code == 0
            mock_process.assert_called_once()
            assert mock_process.call_args[1]["backend"] == "ctranslate2"


class TestCLIIntegration:
    """Simple integration tests."""

    def test_main_entry(self) -> None:
        with patch("insanely_fast_whisper_api.cli.cli.cli") as mock_cli:
            main()
            mock_cli.assert_called_once()

    def test_global_facade_instance(self) -> None:
        assert isinstance(cli_facade, CLIFacade)

