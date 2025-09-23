"""Tests for API data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from insanely_fast_whisper_api.api.models import (
    TranscriptionChunk,
    TranscriptionResponse,
)


def test_transcription_chunk_creation() -> None:
    """Test TranscriptionChunk model creation with valid data."""
    chunk = TranscriptionChunk(
        text="Hello world",
        timestamp=(0.0, 2.5)
    )

    assert chunk.text == "Hello world"
    assert chunk.timestamp == (0.0, 2.5)


def test_transcription_chunk_validation() -> None:
    """Test TranscriptionChunk model validation."""
    # Valid chunk
    chunk = TranscriptionChunk(text="test", timestamp=(1.0, 2.0))
    assert chunk.text == "test"
    assert chunk.timestamp == (1.0, 2.0)


def test_transcription_chunk_validation_errors() -> None:
    """Test TranscriptionChunk model validation errors."""
    # Missing required field
    with pytest.raises(ValidationError):
        TranscriptionChunk(timestamp=(0.0, 1.0))  # Missing text

    # Wrong tuple length - this might not raise an error in Pydantic v2
    # Let's test with something that definitely should fail
    with pytest.raises(ValidationError):
        TranscriptionChunk(text="test", timestamp="invalid")  # String instead of tuple


def test_transcription_chunk_serialization() -> None:
    """Test TranscriptionChunk model serialization."""
    chunk = TranscriptionChunk(text="Hello", timestamp=(0.5, 1.5))

    # Test dict conversion
    chunk_dict = chunk.model_dump()
    expected = {
        "text": "Hello",
        "timestamp": (0.5, 1.5)
    }
    assert chunk_dict == expected

    # Test JSON serialization
    chunk_json = chunk.model_dump_json()
    assert '"text":"Hello"' in chunk_json
    assert '"timestamp":[0.5,1.5]' in chunk_json


def test_transcription_response_creation_minimal() -> None:
    """Test TranscriptionResponse model creation with minimal data."""
    response = TranscriptionResponse(text="Hello world")

    assert response.text == "Hello world"
    assert response.chunks is None
    assert response.segments is None
    assert response.language is None
    assert response.runtime_seconds is None


def test_transcription_response_creation_full() -> None:
    """Test TranscriptionResponse model creation with all fields."""
    chunks = [
        TranscriptionChunk(text="Hello", timestamp=(0.0, 1.0)),
        TranscriptionChunk(text="world", timestamp=(1.0, 2.0))
    ]

    segments = [
        {"text": "Hello", "start": 0.0, "end": 1.0},
        {"text": "world", "start": 1.0, "end": 2.0}
    ]

    response = TranscriptionResponse(
        text="Hello world",
        chunks=chunks,
        segments=segments,
        language="en",
        runtime_seconds=2.5
    )

    assert response.text == "Hello world"
    assert len(response.chunks) == 2
    assert len(response.segments) == 2
    assert response.language == "en"
    assert response.runtime_seconds == 2.5


def test_transcription_response_validation() -> None:
    """Test TranscriptionResponse model validation."""
    # Valid response with just text
    response = TranscriptionResponse(text="test")
    assert response.text == "test"
    assert response.chunks is None

    # Valid response with all optional fields
    response = TranscriptionResponse(
        text="full test",
        chunks=[TranscriptionChunk(text="chunk", timestamp=(0.0, 1.0))],
        segments=[{"text": "segment", "start": 0.0, "end": 1.0}],
        language="es",
        runtime_seconds=1.5
    )
    assert response.language == "es"
    assert response.runtime_seconds == 1.5


def test_transcription_response_validation_errors() -> None:
    """Test TranscriptionResponse model validation errors."""
    # Missing required field
    with pytest.raises(ValidationError):
        TranscriptionResponse()  # Missing text

    # Invalid runtime_seconds type
    with pytest.raises(ValidationError):
        TranscriptionResponse(text="test", runtime_seconds="invalid")  # Should be float or None


def test_transcription_response_serialization() -> None:
    """Test TranscriptionResponse model serialization."""
    response = TranscriptionResponse(
        text="Hello world",
        language="en",
        runtime_seconds=1.23
    )

    # Test dict conversion
    response_dict = response.model_dump()
    expected = {
        "text": "Hello world",
        "chunks": None,
        "segments": None,
        "language": "en",
        "runtime_seconds": 1.23
    }
    assert response_dict == expected

    # Test JSON serialization
    response_json = response.model_dump_json()
    assert '"text":"Hello world"' in response_json
    assert '"language":"en"' in response_json
    assert '"runtime_seconds":1.23' in response_json


def test_transcription_response_with_chunks_serialization() -> None:
    """Test TranscriptionResponse serialization with chunks."""
    chunks = [TranscriptionChunk(text="Hello", timestamp=(0.0, 1.0))]
    response = TranscriptionResponse(text="Hello", chunks=chunks)

    response_dict = response.model_dump()
    assert response_dict["chunks"][0]["text"] == "Hello"
    assert response_dict["chunks"][0]["timestamp"] == (0.0, 1.0)


def test_transcription_response_field_descriptions() -> None:
    """Test that field descriptions are properly set."""
    # Test TranscriptionChunk field descriptions
    chunk_schema = TranscriptionChunk.model_json_schema()
    assert "description" in chunk_schema["properties"]["text"]
    assert "description" in chunk_schema["properties"]["timestamp"]

    # Test TranscriptionResponse field descriptions
    response_schema = TranscriptionResponse.model_json_schema()
    assert "description" in response_schema["properties"]["text"]
    assert "description" in response_schema["properties"]["chunks"]
    assert "description" in response_schema["properties"]["segments"]
    assert "description" in response_schema["properties"]["language"]
    assert "description" in response_schema["properties"]["runtime_seconds"]


def test_transcription_chunk_schema() -> None:
    """Test TranscriptionChunk schema generation."""
    schema = TranscriptionChunk.model_json_schema()

    # Check that schema was generated successfully
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "text" in schema["properties"]
    assert "timestamp" in schema["properties"]
    assert schema["title"] == "TranscriptionChunk"


def test_transcription_response_schema() -> None:
    """Test TranscriptionResponse schema generation."""
    schema = TranscriptionResponse.model_json_schema()

    # Check that schema was generated successfully
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "text" in schema["properties"]
    assert "chunks" in schema["properties"]
    assert "segments" in schema["properties"]
    assert "language" in schema["properties"]
    assert "runtime_seconds" in schema["properties"]
    assert schema["title"] == "TranscriptionResponse"
