"""Utility functions for the Insanely Fast Whisper API."""

import logging
import os
import shutil
import tempfile
import uuid

from fastapi import HTTPException, UploadFile

from insanely_fast_whisper_api.utils.constants import (
    SUPPORTED_AUDIO_FORMATS,
    UPLOAD_DIR,
)

logger = logging.getLogger(__name__)


def validate_audio_file(file: UploadFile) -> None:
    """Validate that the uploaded file is a supported audio format.

    Args:
        file: The uploaded file to validate

    Raises:
        HTTPException: If the file format is not supported
    """
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}",
        )


def save_upload_file(file: UploadFile) -> str:
    """Save an uploaded file to disk with a unique filename.

    Args:
        file: The uploaded file to save

    Returns:
        str: Path to the saved file

    Raises:
        HTTPException: If there's an error saving the file
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    temp_filename = f"{str(uuid.uuid4())}_{file.filename}"
    temp_filepath = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return temp_filepath
    except (OSError, IOError) as e:
        raise HTTPException(
            status_code=500, detail=f"Error saving uploaded file: {str(e)}"
        ) from e


def cleanup_temp_files(file_paths: list[str]) -> None:
    """
    Clean up temporary files.

    Args:
        file_paths: List of file paths to delete.
    """
    for file_path in file_paths:
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # Try to remove parent directory if it's empty
            dir_path = os.path.dirname(file_path)
            # Ensure we only attempt to remove directories from the designated UPLOAD_DIR or a tempfile.gettempdir()
            # and that the directory is indeed empty.
            if dir_path and os.path.isdir(dir_path) and not os.listdir(dir_path):
                if dir_path.startswith(UPLOAD_DIR) or dir_path.startswith(
                    tempfile.gettempdir()
                ):
                    try:
                        os.rmdir(dir_path)
                    except OSError:  # Catch potential race conditions or other errors
                        pass  # Don't raise if cleanup fails
        except (OSError, IOError) as e:
            # Log cleanup failures but don't raise exceptions
            logger.warning("Failed to clean up %s: %s", file_path, e, exc_info=True)


class FileHandler:
    """Centralized file handling operations for the API.

    This class provides a clean interface for file validation, saving,
    and cleanup operations used by the API endpoints.
    """

    def __init__(self, upload_dir: str = UPLOAD_DIR):
        """Initialize FileHandler with upload directory.

        Args:
            upload_dir: Directory for temporary file uploads
        """
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    def validate_audio_file(self, file: UploadFile) -> None:
        """Validate that the uploaded file is a supported audio format.

        Args:
            file: The uploaded file to validate

        Raises:
            HTTPException: If the file format is not supported
        """
        validate_audio_file(file)  # Use existing function

    def save_upload(self, file: UploadFile) -> str:
        """Save an uploaded file to disk with a unique filename.

        Args:
            file: The uploaded file to save

        Returns:
            str: Path to the saved file

        Raises:
            HTTPException: If there's an error saving the file
        """
        temp_filename = f"{str(uuid.uuid4())}_{file.filename}"
        temp_filepath = os.path.join(self.upload_dir, temp_filename)

        try:
            with open(temp_filepath, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info("File saved temporarily as: %s", temp_filepath)
            return temp_filepath
        except (OSError, IOError) as e:
            logger.error("Error saving uploaded file: %s", str(e))
            raise HTTPException(
                status_code=500, detail=f"Error saving uploaded file: {str(e)}"
            ) from e

    def cleanup(self, file_path: str) -> None:
        """Clean up a temporary file.

        Args:
            file_path: Path to the file to delete
        """
        try:
            if os.path.exists(file_path):
                logger.debug("Cleaning up temporary file: %s", file_path)
                os.remove(file_path)
        except (OSError, IOError) as e:
            logger.warning("Failed to cleanup file %s: %s", file_path, e)
            # Don't raise - cleanup failures shouldn't break the API
