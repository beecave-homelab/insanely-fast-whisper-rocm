"""Utilities for detecting and classifying Out of Memory (OOM) errors.

This module provides functions to identify whether a RuntimeError is caused by
GPU memory exhaustion (either CUDA or HIP/ROCm) and to extract relevant details
from the error message.
"""

from __future__ import annotations

import re

from insanely_fast_whisper_rocm.core.errors import OutOfMemoryError


def classify_oom_error(exception: Exception) -> OutOfMemoryError | None:
    """Classify an exception as an OutOfMemoryError if applicable.

    Checks if the exception is a RuntimeError containing memory exhaustion
    messages specific to CUDA or HIP/ROCm.

    Args:
        exception: The exception to classify.

    Returns:
        An instance of OutOfMemoryError if an OOM was detected, None otherwise.
    """
    if not isinstance(exception, RuntimeError):
        return None

    msg = str(exception)
    msg_lower = msg.lower()
    # Check for common CUDA and HIP OOM signatures
    is_oom = any(
        pattern in msg_lower for pattern in ("hip out of memory", "cuda out of memory")
    )

    if not is_oom:
        return None

    # Parse memory allocation details if available
    # Example: "HIP out of memory. Tried to allocate 1.00 GiB
    # (GPU 0; 7.93 GiB total capacity; ...)"
    device = None
    device_match = re.search(r"\(GPU (\d+);", msg)
    if device_match:
        device = device_match.group(1)

    return OutOfMemoryError(msg, device=device)
