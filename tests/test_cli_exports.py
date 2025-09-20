"""Tests for the CLI export functionality."""
# pylint: disable=attribute-defined-outside-init

import json
import shutil
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from insanely_fast_whisper_api.cli.cli import cli


class TestCliExports:
    """Test suite for CLI export functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.runner = CliRunner()
        self.audio_file = Path("tests/data/conversion-test-file.mp3")
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
        # Ensure the audio file path exists for click.Path(exists=True)
        self.audio_file.parent.mkdir(parents=True, exist_ok=True)
        self.audio_file.write_bytes(b"\x00\x00")

    def teardown_method(self) -> None:
        """Clean up after tests."""
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove test artifacts while preserving .gitkeep files and directories.

        This avoids deleting repository-tracked placeholder files like `.gitkeep`
        in the following directories:
        - transcripts/
        - transcripts-txt/
        - transcripts-srt/
        - custom_output/
        """
        for dir_path in self.output_dirs:
            if not dir_path.exists():
                continue
            # Remove all files/dirs inside, except `.gitkeep`.
            for entry in dir_path.iterdir():
                if entry.name == ".gitkeep":
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    try:
                        entry.unlink()
                    except FileNotFoundError:
                        # Entry might have been concurrently removed; safe to ignore.
                        pass

    def test_export_json_default(self) -> None:
        """Test default export to JSON."""
        with patch(
            "insanely_fast_whisper_api.cli.commands.cli_facade.process_audio"
        ) as mock_process:
            mock_process.return_value = {
                "text": "Hello",
                "chunks": [
                    {"text": "Hello", "timestamp": [0.0, 1.0]},
                ],
                "runtime_seconds": 0.5,
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

    def test_export_txt(self) -> None:
        """Test --export-format txt."""
        with patch(
            "insanely_fast_whisper_api.cli.commands.cli_facade.process_audio"
        ) as mock_process:
            mock_process.return_value = {
                "text": "Hello",
                "chunks": [],
                "runtime_seconds": 0.5,
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

    def test_export_srt(self) -> None:
        """Test --export-format srt."""
        with patch(
            "insanely_fast_whisper_api.cli.commands.cli_facade.process_audio"
        ) as mock_process:
            mock_process.return_value = {
                "text": "Hello",
                "chunks": [
                    {"text": "Hello", "timestamp": [0.0, 1.0]},
                ],
                "runtime_seconds": 0.5,
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

    def test_export_all(self) -> None:
        """Test --export-format all."""
        with patch(
            "insanely_fast_whisper_api.cli.commands.cli_facade.process_audio"
        ) as mock_process:
            mock_process.return_value = {
                "text": "Hello",
                "chunks": [
                    {"text": "Hello", "timestamp": [0.0, 1.0]},
                ],
                "runtime_seconds": 0.5,
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

    def test_custom_output_path(self) -> None:
        """Test custom output path with --output."""
        output_file = Path("custom_output/result.json")
        with patch(
            "insanely_fast_whisper_api.cli.commands.cli_facade.process_audio"
        ) as mock_process:
            mock_process.return_value = {
                "text": "Hello",
                "chunks": [],
                "runtime_seconds": 0.5,
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
                    "--no-stabilize",
                    "--model",
                    self.model,
                    "--batch-size",
                    str(self.batch_size),
                ],
            )
        assert result.exit_code == 0, result.output
        assert output_file.exists()
        saved = json.loads(output_file.read_text(encoding="utf-8"))
        assert saved["text"] == "Hello"
        assert saved["transcribe"] == "Hello"
