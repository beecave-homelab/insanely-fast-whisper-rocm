"""WebUI module for the Insanely Fast Whisper API.

This module provides a Gradio-based web interface for audio transcription
with support for file uploads, real-time progress updates, and result downloads.
"""

# Import main components for external use
from insanely_fast_whisper_api.webui.ui import create_ui_components
from insanely_fast_whisper_api.webui.cli import launch_webui

__all__ = ["create_ui_components", "launch_webui"]
