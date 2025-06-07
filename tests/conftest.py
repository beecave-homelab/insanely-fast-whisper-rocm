"""Pytest configuration and fixtures."""

import os
import pytest


@pytest.fixture(scope="session")
def test_data_dir():
    """Create and return a directory for test data files."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def temp_upload_dir(tmp_path_factory):
    """Create and return a temporary directory for file uploads."""
    return tmp_path_factory.mktemp("uploads")
