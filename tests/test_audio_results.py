"""Tests for audio processing results utilities."""

from __future__ import annotations

import pytest

from insanely_fast_whisper_api.audio.results import merge_chunk_results


def test_merge_chunk_results_empty_list() -> None:
    """Test merge_chunk_results with empty chunk results list."""
    result = merge_chunk_results([])

    expected = {
        "text": "",
        "chunks": [],
        "runtime_seconds": 0.0
    }
    assert result == expected


def test_merge_chunk_results_single_chunk() -> None:
    """Test merge_chunk_results with single chunk."""
    chunk_results = [
        ({
            "text": "Hello world",
            "chunks": [
                {
                    "timestamp": [0.0, 2.5],
                    "text": "Hello world"
                }
            ],
            "runtime_seconds": 2.5
        }, 0.0)
    ]

    result = merge_chunk_results(chunk_results)

    assert result["text"] == "Hello world"
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["timestamp"] == [0.0, 2.5]
    assert result["runtime_seconds"] == 2.5


def test_merge_chunk_results_multiple_chunks() -> None:
    """Test merge_chunk_results with multiple chunks."""
    chunk_results = [
        ({
            "text": "Hello world",
            "chunks": [
                {
                    "timestamp": [0.0, 2.5],
                    "text": "Hello world"
                }
            ],
            "runtime_seconds": 2.5
        }, 0.0),
        ({
            "text": "This is a test",
            "chunks": [
                {
                    "timestamp": [0.0, 2.0],
                    "text": "This is a test"
                }
            ],
            "runtime_seconds": 2.0
        }, 2.5)
    ]

    result = merge_chunk_results(chunk_results)

    # Check merged text
    expected_text = "Hello world\n\nThis is a test"
    assert result["text"] == expected_text

    # Check chunks with adjusted timestamps
    assert len(result["chunks"]) == 2
    assert result["chunks"][0]["timestamp"] == [0.0, 2.5]  # First chunk unchanged
    assert result["chunks"][1]["timestamp"] == [2.5, 4.5]  # Second chunk adjusted

    # Check runtime calculation
    assert result["runtime_seconds"] == 4.5


def test_merge_chunk_results_word_timestamps() -> None:
    """Test merge_chunk_results with word-level timestamps."""
    chunk_results = [
        ({
            "text": "Hello world",
            "chunks": [
                {
                    "timestamp": [0.0, 2.5],
                    "text": "Hello world",
                    "words": [
                        {"start": 0.0, "end": 0.5, "text": "Hello"},
                        {"start": 0.6, "end": 1.2, "text": "world"}
                    ]
                }
            ],
            "runtime_seconds": 2.5
        }, 0.0),
        ({
            "text": "This is",
            "chunks": [
                {
                    "timestamp": [0.0, 1.5],
                    "text": "This is",
                    "words": [
                        {"start": 0.0, "end": 0.3, "text": "This"},
                        {"start": 0.4, "end": 0.7, "text": "is"}
                    ]
                }
            ],
            "runtime_seconds": 1.5
        }, 2.5)
    ]

    result = merge_chunk_results(chunk_results)

    # Check word timestamps are adjusted
    assert len(result["chunks"]) == 2
    first_chunk_words = result["chunks"][0]["words"]
    assert first_chunk_words[0]["start"] == 0.0  # Unchanged
    assert first_chunk_words[1]["start"] == 0.6  # Unchanged

    second_chunk_words = result["chunks"][1]["words"]
    assert second_chunk_words[0]["start"] == 2.5  # Adjusted by chunk start time
    assert second_chunk_words[1]["start"] == 2.6  # Adjusted by chunk start time


def test_merge_chunk_results_config_used() -> None:
    """Test merge_chunk_results preserves config_used from first chunk."""
    chunk_results = [
        ({
            "text": "First chunk",
            "chunks": [{"timestamp": [0.0, 1.0], "text": "First"}],
            "runtime_seconds": 1.0,
            "config_used": {"language": "en", "model": "tiny"}
        }, 0.0),
        ({
            "text": "Second chunk",
            "chunks": [{"timestamp": [0.0, 1.0], "text": "Second"}],
            "runtime_seconds": 1.0,
            "config_used": {"language": "fr", "model": "base"}  # Should be ignored
        }, 1.0)
    ]

    result = merge_chunk_results(chunk_results)

    # Config from first chunk should be preserved
    assert result["config_used"]["language"] == "en"
    assert result["config_used"]["model"] == "tiny"

    # Chunking info should be added
    assert result["config_used"]["chunking_used"] is True
    assert result["config_used"]["num_chunks"] == 2


def test_merge_chunk_results_tuple_timestamps() -> None:
    """Test merge_chunk_results with tuple timestamps."""
    chunk_results = [
        ({
            "text": "Test",
            "chunks": [
                {
                    "timestamp": (0.0, 2.5),
                    "text": "Test"
                }
            ],
            "runtime_seconds": 2.5
        }, 0.0),
        ({
            "text": "Chunk",
            "chunks": [
                {
                    "timestamp": (0.0, 2.0),
                    "text": "Chunk"
                }
            ],
            "runtime_seconds": 2.0
        }, 2.5)
    ]

    result = merge_chunk_results(chunk_results)

    # Check tuple timestamps are handled correctly
    assert len(result["chunks"]) == 2
    assert result["chunks"][0]["timestamp"] == (0.0, 2.5)
    assert result["chunks"][1]["timestamp"] == (2.5, 4.5)


def test_merge_chunk_results_missing_text() -> None:
    """Test merge_chunk_results handles missing text gracefully."""
    chunk_results = [
        ({
            "chunks": [{"timestamp": [0.0, 1.0], "text": "No text key"}],
            "runtime_seconds": 1.0
        }, 0.0)
    ]

    result = merge_chunk_results(chunk_results)

    # Should handle missing text gracefully
    assert result["text"] == ""
    assert len(result["chunks"]) == 1


def test_merge_chunk_results_missing_runtime() -> None:
    """Test merge_chunk_results handles missing runtime_seconds gracefully."""
    chunk_results = [
        ({
            "text": "Test",
            "chunks": [{"timestamp": [0.0, 1.0], "text": "Test"}]
            # No runtime_seconds
        }, 0.0)
    ]

    result = merge_chunk_results(chunk_results)

    # Should handle missing runtime gracefully
    assert result["runtime_seconds"] == 0.0


def test_merge_chunk_results_no_segments() -> None:
    """Test merge_chunk_results with chunks having no segments."""
    chunk_results = [
        ({
            "text": "Test without chunks",
            "runtime_seconds": 1.0
            # No chunks key
        }, 0.0)
    ]

    result = merge_chunk_results(chunk_results)

    assert result["text"] == "Test without chunks"
    assert result["chunks"] == []
    assert result["runtime_seconds"] == 1.0


def test_merge_chunk_results_mixed_timestamp_types() -> None:
    """Test merge_chunk_results with mixed timestamp types."""
    chunk_results = [
        ({
            "text": "List timestamps",
            "chunks": [
                {
                    "timestamp": [0.0, 2.5],
                    "text": "List chunk"
                }
            ],
            "runtime_seconds": 2.5
        }, 0.0),
        ({
            "text": "Tuple timestamps",
            "chunks": [
                {
                    "timestamp": (0.0, 2.0),
                    "text": "Tuple chunk"
                }
            ],
            "runtime_seconds": 2.0
        }, 2.5)
    ]

    result = merge_chunk_results(chunk_results)

    # Both types should be handled correctly
    assert len(result["chunks"]) == 2
    assert result["chunks"][0]["timestamp"] == [0.0, 2.5]  # Original
    assert result["chunks"][1]["timestamp"] == (2.5, 4.5)  # Adjusted tuple
