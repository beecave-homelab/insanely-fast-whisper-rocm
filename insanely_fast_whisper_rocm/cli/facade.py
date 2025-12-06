"""CLI facade for simplified access to core ASR functionality."""

import logging
from pathlib import Path
from typing import Any

from insanely_fast_whisper_rocm.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)
from insanely_fast_whisper_rocm.core.utils import convert_device_string
from insanely_fast_whisper_rocm.utils import constants

logger = logging.getLogger(__name__)


class CLIFacade:
    """Facade for CLI access to ASR functionality."""

    def __init__(self) -> None:
        """Initialize the CLI facade.

        Initializes internal backend cache and the last-used configuration so
        repeated calls with the same configuration can reuse the backend.
        """
        self.backend = None
        self._current_config = None

    def get_env_config(self) -> dict[str, Any]:
        """Get configuration from environment variables with safe defaults.

        Returns:
            dict[str, Any]: Mapping containing defaults for:
                - model (str): Default model identifier.
                - device (str): Device string resolved from environment.
                - batch_size (int): Default batch size.
                - timestamp_type (str): Default timestamp type.
                - language (str | None): Default language code or None.
        """
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

    def _create_backend_config(  # pylint: disable=too-many-arguments
        self,
        model: str,
        device: str,
        dtype: str,
        batch_size: int,
        chunk_length: int,
    ) -> HuggingFaceBackendConfig:
        """Create a backend configuration from CLI-supplied arguments.

        Args:
            model (str): Model identifier (e.g., a Hugging Face repo id).
            device (str): Target device (e.g., "cpu", "cuda:0").
            dtype (str): Data type for inference (e.g., "float16").
            batch_size (int): Number of samples to batch per inference step.
            chunk_length (int): Audio chunk size in seconds.

        Returns:
            HuggingFaceBackendConfig: The constructed backend configuration.
        """
        return HuggingFaceBackendConfig(
            model_name=model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            chunk_length=chunk_length,
        )

    def process_audio(  # pylint: disable=too-many-arguments
        self,
        audio_file_path: Path,
        model: str | None = None,
        device: str | None = None,
        dtype: str = "float16",
        batch_size: int | None = None,
        chunk_length: int = 30,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps_value: bool | str = True,
    ) -> dict[str, Any]:
        """Process an audio file via the ASR backend.

        Supports transcription and translation depending on the ``task``.

        Args:
            audio_file_path (Path): Path to the audio file.
            model (str | None): Optional model name to use.
            device (str | None): Optional device for inference.
            dtype (str): Data type for inference.
            batch_size (int | None): Optional batch size for processing.
            chunk_length (int): Audio chunk length in seconds.
            language (str | None): Optional language code for processing.
            task (str): Task to perform ("transcribe" or "translate").
            return_timestamps_value (bool | str): Whether/how to return
                timestamps. Pass True/False or a string mode supported by the
                backend.

        Returns:
            dict[str, Any]: Results payload with transcription/translation
            outputs as provided by the backend.
        """
        # Get config from environment with defaults
        config = self.get_env_config()

        # Use provided parameters or fall back to config values
        model_name = model or config["model"]
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
            model=model_name,
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

        # Perform transcription
        return self.backend.process_audio(
            audio_file_path=str(audio_file_path),
            language=language,
            task=task,
            return_timestamps_value=return_timestamps_value,
        )


# Global facade instance for CLI use (exposes process_audio for
# both transcription and translation)
cli_facade = CLIFacade()
