"""Response formatting utilities for the API.

This module implements the Strategy pattern for formatting different
types of responses (JSON, text) from ASR processing results.
"""

from typing import Any, Dict, Union

from fastapi.responses import JSONResponse, PlainTextResponse

from insanely_fast_whisper_api.utils import (
    RESPONSE_FORMAT_JSON,
    RESPONSE_FORMAT_TEXT,
)


class ResponseFormatter:
    """Strategy pattern implementation for response formatting."""

    @staticmethod
    def format_transcription(
        result: Dict[str, Any], response_format: str = RESPONSE_FORMAT_JSON
    ) -> Union[JSONResponse, PlainTextResponse]:
        """Format transcription result based on requested format.

        Args:
            result: The transcription result dictionary from ASR pipeline
            response_format: Desired response format ("json" or "text")

        Returns:
            Union[JSONResponse, PlainTextResponse]: Formatted response
        """
        if response_format == RESPONSE_FORMAT_TEXT:
            # Extract text from result for plain text response
            text_output = result.get("text", "")
            return PlainTextResponse(text_output)

        # Default to JSON response
        return JSONResponse(content=result)

    @staticmethod
    def format_translation(
        result: Dict[str, Any], response_format: str = RESPONSE_FORMAT_JSON
    ) -> Union[JSONResponse, PlainTextResponse]:
        """Format translation result based on requested format.

        Args:
            result: The translation result dictionary from ASR pipeline
            response_format: Desired response format ("json" or "text")

        Returns:
            Union[JSONResponse, PlainTextResponse]: Formatted response
        """
        if response_format == RESPONSE_FORMAT_TEXT:
            # Extract text from transcription within result for plain text response
            transcription_output = result.get("transcription", {})
            text_output = transcription_output.get("text", "")
            return PlainTextResponse(text_output)

        # Default to JSON response
        return JSONResponse(content=result)
