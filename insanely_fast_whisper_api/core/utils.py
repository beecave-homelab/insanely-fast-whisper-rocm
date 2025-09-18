"""Core utility functions for the ASR pipeline."""

# import torch # Removed unused import

# TODO: Implement _GlobalConfig singleton
# TODO: Add timestamp helpers


def convert_device_string(device_id: str) -> str:
    """Convert a simplified device identifier to a PyTorch-compatible device string.

    Args:
        device_id: A string representing the device. Can be:
                 - A number (e.g., "0") -> converted to "cuda:0"
                 - "mps" -> returned as-is for Apple Silicon
                 - "cpu" -> returned as-is for CPU
                 - Already formatted string (e.g., "cuda:1") -> returned as-is

    Returns:
        str: A PyTorch-compatible device string

    Example:
        >>> convert_device_string("0")
        'cuda:0'
        >>> convert_device_string("cpu")
        'cpu'
    """
    if device_id.isdigit():
        return f"cuda:{device_id}"
    if device_id == "mps":
        return "mps"
    if device_id == "cpu":
        return "cpu"
    # Removed else, de-indented the following line
    # Assume it's already a properly formatted device string
    return device_id
