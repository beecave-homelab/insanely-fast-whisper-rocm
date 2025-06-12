"""WebUI module for the Insanely Fast Whisper API.

This module provides a Gradio-based web interface for audio transcription
with support for file uploads, real-time progress updates, and result downloads.
"""

from insanely_fast_whisper_api.webui.cli import launch_webui
from insanely_fast_whisper_api.webui.merge_handler import (
    MergeConfiguration,
    MergeHandler,
    MergeResult,
    SrtMerger,
    TxtMerger,
    VttMerger,
    get_merge_handler,
    merge_files,
)

# Import main components for external use
from insanely_fast_whisper_api.webui.ui import create_ui_components

# Import ZIP and merge utilities
from insanely_fast_whisper_api.webui.zip_creator import (
    BatchZipBuilder,
    ZipConfiguration,
    ZipStats,
    create_batch_zip,
)

__all__ = [
    "create_ui_components",
    "launch_webui",
    # ZIP creation
    "BatchZipBuilder",
    "ZipConfiguration",
    "ZipStats",
    "create_batch_zip",
    # File merging
    "MergeHandler",
    "TxtMerger",
    "SrtMerger",
    "VttMerger",
    "MergeConfiguration",
    "MergeResult",
    "get_merge_handler",
    "merge_files",
]
