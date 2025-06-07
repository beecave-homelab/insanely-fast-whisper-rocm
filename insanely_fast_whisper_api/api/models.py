"""Data models for the Insanely Fast Whisper API.

This module contains Pydantic models for request and response handling.
"""

from typing import List, Optional
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
    chunks: Optional[List[TranscriptionChunk]] = Field(
        None, description="Individual chunks with timestamps"
    )
    runtime_seconds: Optional[float] = Field(
        None, description="Processing time in seconds"
    )
