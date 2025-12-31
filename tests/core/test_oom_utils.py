"""Tests for OOM classification utilities."""

from __future__ import annotations

from insanely_fast_whisper_rocm.core.oom_utils import classify_oom_error


def test_classify_oom_error_non_runtime_error_returns_none() -> None:
    """Return None when exception is not a RuntimeError."""
    assert classify_oom_error(ValueError("nope")) is None


def test_classify_oom_error_non_oom_runtime_error_returns_none() -> None:
    """Return None when RuntimeError is not an OOM signature."""
    assert classify_oom_error(RuntimeError("some other runtime error")) is None


def test_classify_oom_error_parses_hip_device_index() -> None:
    """Extract device index from HIP OOM message."""
    err = RuntimeError(
        "HIP out of memory. Tried to allocate 1.00 GiB (GPU 0; 7.93 GiB total capacity;)"
    )

    oom = classify_oom_error(err)

    assert oom is not None
    assert oom.device == "0"
    assert "HIP out of memory" in str(oom)


def test_classify_oom_error_cuda_without_device_has_none_device() -> None:
    """Return OOM error with device unset when not present in message."""
    oom = classify_oom_error(RuntimeError("CUDA out of memory. Tried to allocate"))

    assert oom is not None
    assert oom.device is None
