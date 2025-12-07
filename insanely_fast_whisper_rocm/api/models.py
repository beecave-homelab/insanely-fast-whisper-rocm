"""Data models for the Insanely Fast Whisper API.

This module contains Pydantic models for request and response handling.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptionChunk(BaseModel):
    """A chunk of transcribed text with timestamps."""

    text: str = Field(..., description="The transcribed text for this chunk")
    timestamp: tuple[float, float] = Field(
        ..., description="Start and end timestamps in seconds"
    )


class TranscriptionResponse(BaseModel):
    """Response model for transcription and translation endpoints."""

    text: str = Field(..., description="The complete transcribed/translated text")
    chunks: list[TranscriptionChunk] | None = Field(
        None, description="Individual chunks with timestamps (legacy minimal format)"
    )
    segments: list[dict] | None = Field(
        None,
        description=(
            "Detailed segments following OpenAI Whisper specification (verbose_json)"
        ),
    )
    language: str | None = Field(
        None, description="Detected language for the transcription/translation"
    )
    runtime_seconds: float | None = Field(
        None, description="Processing time in seconds"
    )
