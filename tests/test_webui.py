"""Test script for the WebUI functionality.

This script tests the WebUI functionality by:
1. Starting the WebUI in a separate process
2. Testing the transcription endpoint with sample audio
3. Testing functionality
4. Verifying the output formats
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
TEST_AUDIO_FILE = "uploads/conversion-test-file.mp3"  # Path to the test audio file
LONG_TEST_AUDIO_FILE = "uploads/test.mp3"  # Path to the longer test audio file
WEBUI_URL = "http://localhost:7860"
API_TRANSCRIBE_EP = f"{WEBUI_URL}/transcribe_audio_v2"
API_URL = "http://localhost:8888"

# Global variable to store the WebUI process
webui_process = None


def setup_module():
    """Set up test environment."""
    global webui_process

    # Check if the test audio file exists
    if not os.path.exists(TEST_AUDIO_FILE):
        pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")

    # Start the WebUI in a separate process
    webui_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "insanely_fast_whisper_api.webui",
            "--model",
            "openai/whisper-tiny",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for the WebUI to start
    max_attempts = 10
    for _ in range(max_attempts):
        try:
            response = requests.get(f"{WEBUI_URL}", timeout=5)
            if response.status_code == 200:
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(1)
    else:
        pytest.fail("Failed to start WebUI")


def teardown_module():
    """Clean up test environment."""
    global webui_process

    if webui_process:
        # Send SIGTERM to the WebUI process
        webui_process.terminate()
        try:
            webui_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            webui_process.kill()


def test_webui_ui_elements():
    """Sanity-check that the WebUI root URL responds with valid HTML and the Gradio app container."""
    response = requests.get(WEBUI_URL, timeout=10)
    assert response.status_code == 200, "WebUI did not load successfully"

    # Verify the HTML skeleton and Gradio root element are present
    assert "<!doctype html" in response.text.lower(), "HTML doctype missing"
    assert "<gradio-app" in response.text.lower(), "gradio-app container not found"


@pytest.mark.skip(reason="Direct REST endpoint not yet exposed by Gradio 4 API")
def test_webui_transcription():
    """Test the transcription functionality through the WebUI with a short audio file."""
    # Upload the test audio file
    with open(TEST_AUDIO_FILE, "rb") as f:
        files = {"audio_file": (os.path.basename(TEST_AUDIO_FILE), f, "audio/mp3")}

        # Submit the form with default parameters
        response = requests.post(
            API_TRANSCRIBE_EP,
            files=files,
            data={
                "data": json.dumps([
                    [TEST_AUDIO_FILE],  # audio_file_paths as list
                    "openai/whisper-tiny",  # model
                    "cpu",  # device
                    16,  # batch_size
                    "chunk",  # timestamp_type
                    "en",  # language
                    "transcribe",  # task
                    "float16",  # dtype
                    30,  # chunk_length
                    True,  # save_transcriptions
                    "transcripts",  # temp_uploads_dir
                ])
            },
            timeout=60,
        )

    assert response.status_code == 200, f"Transcription request failed: {response.text}"

    # Parse the response
    result = response.json()
    assert "data" in result, "Response missing 'data' field"
    assert len(result["data"]) >= 3, "Unexpected response format"

    # Check the transcription result
    transcription = result["data"][0]
    assert isinstance(transcription, str), "Transcription should be a string"
    assert len(transcription) > 0, "Transcription is empty"

    # Check the processing time
    processing_time = result["data"][1]
    assert "second" in processing_time.lower(), "Unexpected processing time format"

    # Check the config used
    config_used = result["data"][2]
    assert isinstance(config_used, dict), "Config should be a dictionary"
    assert "model" in config_used, "Config missing 'model' field"


import pytest


def test_long_audio_transcription():
    """Test the transcription functionality with a longer audio file."""
    # Upload the longer test audio file
    if not os.path.exists(LONG_TEST_AUDIO_FILE):
        pytest.skip(f"Long test audio file not found: {LONG_TEST_AUDIO_FILE}")

    with open(LONG_TEST_AUDIO_FILE, "rb") as f:
        files = {"audio_file": (os.path.basename(LONG_TEST_AUDIO_FILE), f, "audio/mp3")}

        # Submit the form with default parameters
        response = requests.post(
            API_TRANSCRIBE_EP,
            files=files,
            data={
                "data": json.dumps([
                    [LONG_TEST_AUDIO_FILE],  # audio_file_paths as list
                    "openai/whisper-tiny",  # model
                    "cpu",  # device
                    16,  # batch_size
                    "chunk",  # timestamp_type
                    "en",  # language
                    "transcribe",  # task
                    "float16",  # dtype
                    30,  # chunk_length
                    True,  # save_transcriptions
                    "transcripts",  # temp_uploads_dir
                ])
            },
            timeout=300,  # Longer timeout for the longer audio file
        )

    assert response.status_code == 200, (
        f"Long audio transcription request failed: {response.text}"
    )

    # Parse the response
    result = response.json()
    assert "data" in result, "Response missing 'data' field"
    assert len(result["data"]) >= 3, "Unexpected response format"

    # Check the transcription result
    transcription = result["data"][0]
    assert isinstance(transcription, str), "Transcription should be a string"
    assert len(transcription) > 0, "Transcription is empty"

    # Check the processing time
    processing_time = result["data"][1]
    assert "second" in processing_time.lower(), "Unexpected processing time format"


@pytest.mark.skip(
    reason="Export endpoints not available in current WebUI implementation"
)
def test_export_formats():
    """Test the export functionality for different formats."""
    # First, perform a transcription to get a result
    with open(TEST_AUDIO_FILE, "rb") as f:
        files = {"audio_file": (os.path.basename(TEST_AUDIO_FILE), f, "audio/mp3")}
        response = requests.post(
            API_TRANSCRIBE_EP,
            files=files,
            data={
                "data": json.dumps([
                    [TEST_AUDIO_FILE],  # audio_file_paths as list
                    "openai/whisper-tiny",  # model
                    "cpu",  # device
                    16,  # batch_size
                    "chunk",  # timestamp_type
                    "en",  # language
                    "transcribe",  # task
                    "float16",  # dtype
                    30,  # chunk_length
                    True,  # save_transcriptions
                    "transcripts",  # temp_uploads_dir
                ])
            },
            timeout=60,
        )

    assert response.status_code == 200, "Transcription request failed"

    # Get the result data
    result_data = response.json()["data"]

    # Test TXT export
    txt_response = requests.post(
        API_TRANSCRIBE_EP,
        data={
            "data": json.dumps(
                ["txt", result_data[3]]  # format_type  # result data from transcription
            )
        },
        timeout=30,
    )
    assert txt_response.status_code == 200, "TXT export failed"
    assert "transcription_" in txt_response.json()["data"][0], (
        "Unexpected TXT filename format"
    )

    # Test SRT export
    srt_response = requests.post(
        API_TRANSCRIBE_EP,
        data={
            "data": json.dumps(
                ["srt", result_data[3]]  # format_type  # result data from transcription
            )
        },
        timeout=30,
    )
    assert srt_response.status_code == 200, "SRT export failed"
    assert "transcription_" in srt_response.json()["data"][0], (
        "Unexpected SRT filename format"
    )

    # Test JSON export
    json_response = requests.post(
        API_TRANSCRIBE_EP,
        data={
            "data": json.dumps([
                "json",  # format_type
                result_data[3],  # result data from transcription
            ])
        },
        timeout=30,
    )
    assert json_response.status_code == 200, "JSON export failed"
    assert "transcription_" in json_response.json()["data"][0], (
        "Unexpected JSON filename format"
    )


if __name__ == "__main__":
    # Run the tests
    setup_module()
    try:
        test_webui_ui_elements()
        print("✓ test_webui_ui_elements passed")

        test_webui_transcription()
        print("✓ test_webui_transcription (short audio) passed")

        test_long_audio_transcription()
        print("✓ test_long_audio_transcription passed")

        test_export_formats()
        print("✓ test_export_formats passed")

        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        raise
    finally:
        teardown_module()
