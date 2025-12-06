"""Main FastAPI application module for the Insanely Fast Whisper API.

This module serves as the entry point for the FastAPI application,
using the app factory pattern for clean separation of concerns.
"""

# Standard library imports
import logging

# Local imports
from insanely_fast_whisper_rocm.api import create_app

# Configure basic logging to ensure output to console
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create the FastAPI application using the factory pattern
app = create_app()
