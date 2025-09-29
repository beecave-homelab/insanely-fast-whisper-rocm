"""Tests for the CLI commands."""

from __future__ import annotations

import json
import unittest.mock
from pathlib import Path

from click.testing import CliRunner

from insanely_fast_whisper_api.cli.cli import cli


def test_cli_transcribe_fallback_on_corrupted_stabilization(
    tmp_path: Path,
) -> None:
    """Verify the CLI falls back if stabilization is corrupt."""
    # 1. Define mock data
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()
    output_file = tmp_path / "output.json"

    original_result = {
        "text": "This is a valid transcription.",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "This is"},
            {"start": 1.0, "end": 2.0, "text": " a valid transcription."},
        ],
        "config_used": {},
        "runtime_seconds": 0.0,
        "chunks": [],
    }
    corrupted_result = {
        **original_result,
        "segments": [
            {"start": 5.0, "end": 5.0, "text": "corrupt"},
            {"start": 5.0, "end": 5.0, "text": "data"},
        ],
        "segments_count": 2,
    }

    # 2. Mock dependencies
    with (
        unittest.mock.patch(
            "insanely_fast_whisper_api.cli.commands.cli_facade.process_audio"
        ) as mock_process_audio,
        unittest.mock.patch(
            "insanely_fast_whisper_api.cli.commands.stabilize_timestamps"
        ) as mock_stabilize,
    ):
        mock_process_audio.return_value = original_result
        mock_stabilize.return_value = corrupted_result

        # 3. Run the CLI command
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(audio_file),
                "--stabilize",
                "--output",
                str(output_file),
                "--export-format",
                "json",
            ],
            catch_exceptions=False,
        )

    # 4. Assertions
    assert result.exit_code == 0
    mock_stabilize.assert_called_once()

    # Check that the output file contains the ORIGINAL data
    with open(output_file, encoding="utf-8") as f:
        final_data = json.load(f)

    assert final_data["text"] == "This is a valid transcription."
    assert len(final_data["segments"]) == 2
    assert final_data["segments"][0]["start"] == 0.0
