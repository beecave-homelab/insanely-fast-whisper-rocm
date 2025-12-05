"""Pytest configuration and fixtures."""

import os
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path

import pytest
import requests


@pytest.fixture(scope="session")
def test_data_dir() -> str:
    """Create and return a directory for test data files.

    Returns:
        str: Absolute path to the tests/data directory, created if missing.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def temp_upload_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create and return a temporary directory for file uploads.

    Args:
        tmp_path_factory: Pytest factory to create temporary paths.

    Returns:
        Path: Newly created temporary directory path for uploads.
    """
    return tmp_path_factory.mktemp("uploads")


@pytest.fixture(scope="session")
def webui_server(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """Spin up the WebUI once per test session and tear it down afterwards.

    Args:
        request: Pytest fixture request object.

    Yields:
        str: Base URL that can be passed to ``gradio_client.Client``.

    Raises:
        RuntimeError: If the WebUI fails to start within the timeout.

    Notes:
        Uses the lightweight ``openai/whisper-tiny`` model for faster startup.
    """
    if os.getenv("RUN_WEBUI_TESTS", "0") != "1":
        pytest.skip("WebUI server tests are disabled in this environment.")

    base_url = "http://localhost:7861"

    # Launch the WebUI as a subprocess
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "insanely_fast_whisper_api.webui",
            "--model",
            "openai/whisper-tiny",
            "--port",
            "7861",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait until the server responds or timeout after 60s
    timeout = 60
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(base_url, timeout=3).status_code == 200:
                break
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(1)
    else:
        # Server failed to start; dump logs for debugging and abort tests
        output, _ = process.communicate(timeout=5)
        raise RuntimeError(f"WebUI failed to start within {timeout}s. Logs:\n{output}")

    yield base_url

    # Teardown: terminate the process gracefully
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
