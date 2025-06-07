"""Utility functions for Insanely Fast Whisper API WebUI.

This module contains utility functions for file handling, formatting,
and device string conversion used by the WebUI components.
"""

import logging
import os
import tempfile
from datetime import datetime
from typing import Optional
from pathlib import Path

import torch

# Import from core.utils
from insanely_fast_whisper_api.core.utils import (
    convert_device_string as core_convert_device_string,
)

# Configure logger
logger = logging.getLogger("insanely_fast_whisper_api.webui.utils")


def save_temp_file(
    content: str, extension: str = "txt", desired_filename: Optional[str] = None
) -> str:
    """
    Save content to a temporary file and return the file path.

    Args:
        content: The content to save.
        extension: The file extension (e.g., 'txt', 'srt', 'json').
        desired_filename: Optional. If provided, use this as the filename within a temp directory.
    Returns:
        The full path to the temporary file.
    """
    try:
        if desired_filename:
            # Ensure the desired filename has the correct extension
            if not desired_filename.endswith(f".{extension}"):
                desired_filename = f"{Path(desired_filename).stem}.{extension}"

            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, desired_filename)

            # Create the file and write content
            with open(temp_path, "w", encoding="utf-8") as tmp:
                tmp.write(content)
        else:
            # Original behavior: generate a random filename
            suffix = f".{extension}" if extension else ""
            fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="ifw_export_")
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                tmp.write(content)

        logger.debug("[save_temp_file] Temp file created at: %s", temp_path)
        return temp_path
    except (IOError, OSError) as e:
        logger.error("[save_temp_file] Failed to save temp file: %s", e)
        raise


def format_seconds(seconds: Optional[float]) -> str:
    """
    Format seconds to a human-readable time string (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds (can be None).

    Returns:
        Formatted time string.
    """
    if seconds is None:
        return "00:00:00.000"

    whole_seconds = int(seconds)
    milliseconds = int((seconds - whole_seconds) * 1000)
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds = whole_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def convert_device_string(device_id: str) -> str:
    """Wrapper for core.utils.convert_device_string for webui internal use."""
    return core_convert_device_string(device_id)


def generate_timestamped_filename(base_name: str, extension: str) -> str:
    """
    Generate a filename with a timestamp to ensure uniqueness.

    Args:
        base_name: Base name for the file
        extension: File extension (without the dot)

    Returns:
        A filename in the format: base_name_YYYYMMDD_HHMMSS.extension
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


def is_cuda_available() -> bool:
    """
    Check if CUDA is available for PyTorch.

    Returns:
        True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available for PyTorch.

    Returns:
        True if MPS is available, False otherwise
    """
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        return torch.mps.is_available()
    return False
