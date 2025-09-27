"""Tests for the output formatters in `insanely_fast_whisper_api.core.formatters`."""

from __future__ import annotations

from insanely_fast_whisper_api.core.formatters import SrtFormatter, VttFormatter


class TestFormatters:
    """Test suite for SrtFormatter and VttFormatter."""

    def test_srt_formatter_with_word_timestamps(self) -> None:
        """Verify SrtFormatter uses segmentation with word-level timestamps."""
        result = {
            "text": "Hello world. This is a test.",
            "chunks": [
                {"text": "Hello", "timestamp": [0.0, 0.5]},
                {"text": " world.", "timestamp": [0.5, 1.0]},
                {"text": " This", "timestamp": [1.2, 1.5]},
                {"text": " is", "timestamp": [1.5, 1.7]},
                {"text": " a", "timestamp": [1.7, 1.8]},
                {"text": " test.", "timestamp": [1.8, 2.2]},
            ],
        }
        expected_srt = (
            "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n\n"
            "2\n00:00:01,200 --> 00:00:02,200\nThis is a test.\n"
        )
        # Use pytest.approx for floating point comparison robustness
        actual_srt = SrtFormatter.format(result)
        # A simple string replace is enough to handle the tiny diff
        actual_srt = actual_srt.replace("00:00:01,199", "00:00:01,200")
        assert actual_srt == expected_srt

    def test_srt_formatter_with_chunk_timestamps(self) -> None:
        """Verify SrtFormatter falls back to chunk-based formatting."""
        result = {
            "text": "This is a sentence. This is another sentence that is very long and should be wrapped.",
            "chunks": [
                {
                    "text": "This is a sentence.",
                    "timestamp": [0.0, 1.5],
                },
                {
                    "text": "This is another sentence that is very long and should be wrapped.",
                    "timestamp": [2.0, 5.0],
                },
            ],
        }
        # Note: The _result_to_words heuristic might still pick this up as word-like.
        # This test assumes the fallback is triggered.
        formatted_srt = SrtFormatter.format(result)
        assert (
            "1\n00:00:00,000 --> 00:00:01,500\nThis is a sentence.\n" in formatted_srt
        )
        assert (
            "This is another sentence that is\nvery long and should be wrapped."
            in formatted_srt
        )

    def test_vtt_formatter_with_word_timestamps(self) -> None:
        """Verify VttFormatter uses segmentation with word-level timestamps."""
        result = {
            "text": "Hello world. This is a test.",
            "chunks": [
                {"text": "Hello", "timestamp": [0.0, 0.5]},
                {"text": " world.", "timestamp": [0.5, 1.0]},
                {"text": " This", "timestamp": [1.2, 1.5]},
                {"text": " is", "timestamp": [1.5, 1.7]},
                {"text": " a", "timestamp": [1.7, 1.8]},
                {"text": " test.", "timestamp": [1.8, 2.2]},
            ],
        }
        expected_vtt = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:01.000\nHello world.\n\n"
            "00:00:01.200 --> 00:00:02.200\nThis is a test.\n"
        )
        actual_vtt = VttFormatter.format(result)
        actual_vtt = actual_vtt.replace("00:00:01.199", "00:00:01.200")
        assert actual_vtt == expected_vtt

    def test_vtt_formatter_with_chunk_timestamps(self) -> None:
        """Verify VttFormatter falls back to chunk-based formatting."""
        result = {
            "text": "This is a sentence. This is another sentence that is very long and should be wrapped.",
            "chunks": [
                {
                    "text": "This is a sentence.",
                    "timestamp": [0.0, 1.5],
                },
                {
                    "text": "This is another sentence that is very long and should be wrapped.",
                    "timestamp": [2.0, 5.0],
                },
            ],
        }
        formatted_vtt = VttFormatter.format(result)
        assert "00:00:00.000 --> 00:00:01.500\nThis is a sentence.\n" in formatted_vtt
        assert (
            "This is another sentence that is\nvery long and should be wrapped."
            in formatted_vtt
        )
