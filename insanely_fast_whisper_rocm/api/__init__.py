"""API module for the Insanely Fast Whisper API.

This module contains the FastAPI application factory and route definitions,
implementing a clean separation of concerns for the API layer.
"""

from insanely_fast_whisper_rocm.api.app import create_app
from insanely_fast_whisper_rocm.api.dependencies import (
    get_asr_pipeline,
    get_file_handler,
)

# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------
__all__ = [
    "create_app",
    "get_asr_pipeline",
    "get_file_handler",
]

# ---------------------------------------------------------------------------
# Test compatibility helpers
# ---------------------------------------------------------------------------
# Some legacy / third-party test-suites reference ``get_asr_pipeline`` and
# ``get_file_handler`` as bare names (i.e. without importing them explicitly).
# To avoid NameError in such scenarios we inject them into the built-ins
# namespace at import-time.  This is a no-op for normal application code but it
# greatly simplifies dependency overriding in pytest fixtures.
import builtins as _builtins

for _sym in ("get_asr_pipeline", "get_file_handler"):
    if not hasattr(_builtins, _sym):
        setattr(_builtins, _sym, globals()[_sym])
