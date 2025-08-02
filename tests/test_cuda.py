"""Basic sanity check for CUDA availability.

This replaces the previous infinite loop implementation which caused
significant delays in CI environments lacking a GPU.

If CUDA is unavailable the test will be **skipped** rather than hanging.
"""

import sys

import pytest
import torch


@pytest.mark.timeout(5)
def test_cuda_available_or_skip():
    """Pass if CUDA is present, otherwise skip gracefully."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available on this runner")
    # Print diagnostic and ensure exit code 0
    print("GPU is available (device count:", torch.cuda.device_count(), ")")
    assert torch.cuda.device_count() > 0
    sys.stdout.flush()
