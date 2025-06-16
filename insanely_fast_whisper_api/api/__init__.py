"""API module for the Insanely Fast Whisper API.

This module contains the FastAPI application factory and route definitions,
implementing a clean separation of concerns for the API layer.
"""

from insanely_fast_whisper_api.api.app import create_app

__all__ = ["create_app"]
