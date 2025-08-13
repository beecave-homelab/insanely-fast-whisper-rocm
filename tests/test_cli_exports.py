"""Tests for the CLI export functionality."""
# pylint: disable=attribute-defined-outside-init

import shutil
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from insanely_fast_whisper_api.cli.cli import cli


class TestCliExports:
    """Test suite for CLI export functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        self.audio_file = Path(tmp.name)
        tmp.close()
        self.model = "openai/whisper-tiny.en"
        self.batch_size = 6
        self.output_dirs = [
            Path("transcripts"),
            Path("transcripts-txt"),
            Path("transcripts-srt"),
            Path("custom_output"),
        ]
        self._cleanup()  # Ensure clean state before test
        # Create default directories, but not custom_output, which the test creates itself
        for dir_path in self.output_dirs:
            if "custom" not in str(dir_path):
                dir_path.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up after tests."""
        self._cleanup()

    def _cleanup(self):
        """Remove created directories."""
        for dir_path in self.output_dirs:
            shutil.rmtree(dir_path, ignore_errors=True)

    def test_export_json_default(self):
        """Test default export to JSON."""
        with patch(
            "insanely_fast_whisper_api.cli.commands.cli_facade.process_audio"
        ) as mock_process:
            mock_process.return_value = {
                "text": "test",
                "chunks": [
                    {"text": "test", "timestamp": [0.0, 1.0]}
                ],
                "runtime_seconds": 1.0,
                "config_used": {},
            }
            result = self.runner.invoke(
                cli,
                [
                    "transcribe",
                    str(self.audio_file),
                    "--model",
                    self.model,
                    "--batch-size",
                    str(self.batch_size),
                ],
            )
        assert result.exit_code == 0, result.output
        output_dir = Path("transcripts")
        assert output_dir.exists()
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) == 1
        assert json_files[0].stat().st_size > 0

    def test_export_txt(self):
        """Test --export-format txt."""
        with patch(
            "insanely_fast_whisper_api.cli.commands.cli_facade.process_audio"
        ) as mock_process:
            mock_process.return_value = {
                "text": "test",
                "chunks": [
                    {"text": "test", "timestamp": [0.0, 1.0]}
                ],
                "runtime_seconds": 1.0,
                "config_used": {},
            }
            result = self.runner.invoke(
                cli,
                [
                    "transcribe",
                    str(self.audio_file),
                    "--export-format",
                    "txt",
                    "--model",
                    self.model,
                    "--batch-size",
                    str(self.batch_size),
                ],
            )
        assert result.exit_code == 0, result.output
        output_dir = Path("transcripts-txt")
        assert output_dir.exists()
        txt_files = list(output_dir.glob("*.txt"))
        assert len(txt_files) == 1
        assert txt_files[0].stat().st_size > 0

    def test_export_srt(self):
        """Test --export-format srt."""
        with patch(
            "insanely_fast_whisper_api.cli.commands.cli_facade.process_audio"
        ) as mock_process:
            mock_process.return_value = {
                "text": "test",
                "chunks": [
                    {"text": "test", "timestamp": [0.0, 1.0]}
                ],
                "runtime_seconds": 1.0,
                "config_used": {},
            }
            result = self.runner.invoke(
                cli,
                [
                    "transcribe",
                    str(self.audio_file),
                    "--export-format",
                    "srt",
                    "--model",
                    self.model,
                    "--batch-size",
                    str(self.batch_size),
                ],
            )
        assert result.exit_code == 0, result.output
        output_dir = Path("transcripts-srt")
        assert output_dir.exists()
        srt_files = list(output_dir.glob("*.srt"))
        assert len(srt_files) == 1
        assert srt_files[0].stat().st_size > 0

    def test_export_all(self):
        """Test --export-format all."""
        with patch(
            "insanely_fast_whisper_api.cli.commands.cli_facade.process_audio"
        ) as mock_process:
            mock_process.return_value = {
                "text": "test",
                "chunks": [
                    {"text": "test", "timestamp": [0.0, 1.0]}
                ],
                "runtime_seconds": 1.0,
                "config_used": {},
            }
            result = self.runner.invoke(
                cli,
                [
                    "transcribe",
                    str(self.audio_file),
                    "--export-format",
                    "all",
                    "--model",
                    self.model,
                    "--batch-size",
                    str(self.batch_size),
                ],
            )
        assert result.exit_code == 0, result.output

        # Check for JSON
        json_dir = Path("transcripts")
        assert json_dir.exists()
        json_files = list(json_dir.glob("*.json"))
        assert len(json_files) == 1
        assert json_files[0].stat().st_size > 0

        # Check for TXT
        txt_dir = Path("transcripts-txt")
        assert txt_dir.exists()
        txt_files = list(txt_dir.glob("*.txt"))
        assert len(txt_files) == 1
        assert txt_files[0].stat().st_size > 0

        # Check for SRT
        srt_dir = Path("transcripts-srt")
        assert srt_dir.exists()
        srt_files = list(srt_dir.glob("*.srt"))
        assert len(srt_files) == 1
        assert srt_files[0].stat().st_size > 0

    def test_custom_output_path(self):
        """Test custom output path with --output."""
        with self.runner.isolated_filesystem():
            output_file = Path("custom_output/result.json")
            with patch(
                "insanely_fast_whisper_api.cli.commands.cli_facade.process_audio"
            ) as mock_process:
                mock_process.return_value = {
                    "text": "test",
                    "chunks": [
                        {"text": "test", "timestamp": [0.0, 1.0]}
                    ],
                    "runtime_seconds": 1.0,
                    "config_used": {},
                }
                result = self.runner.invoke(
                    cli,
                    [
                        "transcribe",
                        str(self.audio_file),
                        "--output",
                        str(output_file),
                        "--export-format",
                        "json",
                        "--model",
                        self.model,
                        "--batch-size",
                        str(self.batch_size),
                    ],
                )
            assert result.exit_code == 0, result.output
            assert output_file.exists()
            assert output_file.stat().st_size > 0
