"""Tests for core error types."""

from __future__ import annotations

import pytest

from insanely_fast_whisper_rocm.core.errors import (
    InferenceOOMError,
    ModelLoadingOOMError,
    OutOfMemoryError,
    TranscriptionError,
)


def test_out_of_memory_error_stores_metadata() -> None:
    """Store and expose device/config metadata on OOM errors."""
    err = OutOfMemoryError("boom", device="0", config={"batch_size": 4})

    assert isinstance(err, TranscriptionError)
    assert str(err) == "boom"
    assert err.device == "0"
    assert err.config == {"batch_size": 4}


def test_oom_error_subclasses_are_distinct() -> None:
    """Expose distinct error types for model load vs inference OOM."""
    model_err = ModelLoadingOOMError("model")
    infer_err = InferenceOOMError("infer")

    assert isinstance(model_err, OutOfMemoryError)
    assert isinstance(infer_err, OutOfMemoryError)
    assert type(model_err) is not type(infer_err)


def test_oom_error_is_raisable_and_catchable() -> None:
    """Raise/catch OOM errors via base exception types."""

    def _raise() -> None:
        raise InferenceOOMError("infer", device="cuda:0")

    with pytest.raises(OutOfMemoryError, match="infer"):
        _raise()
