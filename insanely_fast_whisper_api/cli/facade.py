"""CLI facade for simplified access to core ASR functionality."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from insanely_fast_whisper_api.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)
from insanely_fast_whisper_api.core.utils import convert_device_string
from insanely_fast_whisper_api.utils import constants

logger = logging.getLogger(__name__)


class CLIFacade:
    """Facade for CLI access to ASR functionality."""

    def __init__(self):
        self.backend = None
        self._current_config = None

    def get_env_config(self) -> Dict[str, Any]:
        """Get configuration from environment variables with fallbacks to defaults."""
        device_id = constants.DEFAULT_DEVICE
        return {
            "model": constants.DEFAULT_MODEL,
            "device": convert_device_string(device_id),
            "batch_size": constants.DEFAULT_BATCH_SIZE,
            "timestamp_type": constants.DEFAULT_TIMESTAMP_TYPE,
            "language": (
                None
                if constants.DEFAULT_LANGUAGE.lower() == "none"
                else constants.DEFAULT_LANGUAGE
            ),
        }

    def _create_backend_config(
        self,
        model: str,
        device: str,
        dtype: str,
        batch_size: int,
        chunk_length: int,
    ) -> HuggingFaceBackendConfig:
        """Create a HuggingFaceBackendConfig from CLI args."""
        return HuggingFaceBackendConfig(
            model_name=model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            chunk_length=chunk_length,
        )

    def transcribe_audio(
        self,
        audio_file_path: Path,
        model: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "float16",
        batch_size: Optional[int] = None,
        chunk_length: int = 30,
        language: Optional[str] = None,
        task: str = "transcribe",
        return_timestamps: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using the core ASR backend.

        Args:
            audio_file_path: Path to the audio file
            model: Model name to use
            device: Device for inference
            dtype: Data type for inference
            batch_size: Batch size for processing
            chunk_length: Audio chunk length in seconds
            language: Language code for transcription
            task: Task to perform (transcribe/translate)
            return_timestamps: Whether to return timestamps

        Returns:
            Dictionary containing transcription results

        Raises:
            TranscriptionError: If transcription fails
            DeviceNotFoundError: If device is not available
        """
        # Get config from environment with defaults
        config = self.get_env_config()

        # Use provided parameters or fall back to config values
        model = model or config["model"]
        device = convert_device_string(device) if device else config["device"]

        batch_size = min(
            max(batch_size or config["batch_size"], constants.MIN_BATCH_SIZE),
            constants.MAX_BATCH_SIZE,
        )

        # For CPU, adjust parameters for better stability
        if device == "cpu":
            logger.info("Running on CPU - adjusting parameters for better stability")
            chunk_length = min(chunk_length, 15)  # Use smaller chunks on CPU
            batch_size = min(batch_size, 2)  # Reduce batch size for CPU

        # Create backend configuration
        backend_config = self._create_backend_config(
            model=model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            chunk_length=chunk_length,
        )

        # Log final configuration
        logger.info(
            "Final configuration - Model: %s, Device: %s, Dtype: %s, "
            "Batch size: %s, Chunk length: %s, Language: %s",
            backend_config.model_name,
            backend_config.device,
            backend_config.dtype,
            backend_config.batch_size,
            backend_config.chunk_length,
            language,
        )

        # Create or reuse backend if configuration hasn't changed
        if self.backend is None or self._current_config != backend_config:
            self.backend = HuggingFaceBackend(backend_config)
            self._current_config = backend_config

        # Get language from config if not provided
        if language is None:
            language = config["language"]

        # Determine timestamp format
        return_timestamps_value = "word" if return_timestamps else False

        # Perform transcription
        return self.backend.process_audio(
            audio_file_path=str(audio_file_path),
            language=language,
            task=task,
            return_timestamps_value=return_timestamps_value,
        )


# Global facade instance for CLI use
cli_facade = CLIFacade()
