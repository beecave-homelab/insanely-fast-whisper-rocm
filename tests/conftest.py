"""Pytest configuration and fixtures."""

import os
import subprocess
import sys
import time

import pytest
import requests


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


@pytest.fixture(scope="session")
def webui_server(request):
    """
    Start the WebUI server for the test session and provide its base URL.
    
    This pytest fixture launches the WebUI as a subprocess, waits until it responds on http://localhost:7861, yields the base URL for use in tests (for example with gradio_client.Client()), and terminates the subprocess when the session ends.
    
    Returns:
        base_url (str): The server base URL (e.g., "http://localhost:7861").
    """
    base_url = "http://localhost:7861"

    # Launch the WebUI as a subprocess
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "insanely_fast_whisper_rocm.webui",
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