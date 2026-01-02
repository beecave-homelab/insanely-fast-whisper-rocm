"""CLI facade for simplified access to core ASR functionality."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from insanely_fast_whisper_rocm.core.asr_backend import (
    HuggingFaceBackendConfig,
)
from insanely_fast_whisper_rocm.core.cancellation import CancellationToken
from insanely_fast_whisper_rocm.core.errors import (
    OutOfMemoryError,
    TranscriptionError,
)
from insanely_fast_whisper_rocm.core.orchestrator import (
    Orchestrator,
    create_orchestrator,
)
from insanely_fast_whisper_rocm.core.progress import ProgressCallback
from insanely_fast_whisper_rocm.core.utils import convert_device_string
from insanely_fast_whisper_rocm.utils import constants

logger = logging.getLogger(__name__)


class CLIFacade:
    """Facade for CLI access to ASR functionality."""

    orchestrator_factory: Callable[[], Orchestrator]

    def __init__(
        self,
        *,
        orchestrator_factory: Callable[[], Orchestrator] | None = None,
        check_file_exists: bool = False,
    ) -> None:
        """Initialize the CLI facade.

        Initializes internal backend cache and the last-used configuration so
        repeated calls with the same configuration can reuse the backend.

        Args:
            orchestrator_factory: Factory used to create the orchestrator.
                Defaults to :func:`create_orchestrator`.
            check_file_exists: Whether to verify input audio paths exist. Disabled
                by default so tests and programmatic callers can provide synthetic
                paths. Production entry points should pass ``True`` to retain
                strict validation.

        """
        self.backend: Any | None = None
        self._current_config: HuggingFaceBackendConfig | None = None
        self.orchestrator_factory = orchestrator_factory or create_orchestrator
        self.check_file_exists = check_file_exists

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

    def _create_backend_config(
        self,
        model: str,
        device: str,
        dtype: str,
        batch_size: int,
        chunk_length: int,
        progress_group_size: int,
    ) -> HuggingFaceBackendConfig:
        """Create a backend configuration from CLI-supplied arguments.

        Args:
            model (str): Model identifier (e.g., a Hugging Face repo id).
            device (str): Target device (e.g., "cpu", "cuda:0").
            dtype (str): Data type for inference (e.g., "float16").
            batch_size (int): Number of samples to batch per inference step.
            chunk_length (int): Audio chunk size in seconds.
            progress_group_size (int): Chunks per progress update group.

        Returns:
            HuggingFaceBackendConfig: The constructed backend configuration.

        """
        return HuggingFaceBackendConfig(
            model_name=model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            chunk_length=chunk_length,
            progress_group_size=progress_group_size,
        )

    def process_audio(
        self,
        audio_file_path: Path,
        model: str | None = None,
        device: str | None = None,
        dtype: str = "float16",
        batch_size: int | None = None,
        chunk_length: int = 30,
        progress_group_size: int | None = None,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps_value: bool | str = True,
        progress_cb: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Process an audio file via the ASR backend.

        Supports transcription and translation depending on ``task``.

        Args:
            audio_file_path: Path to the audio file on disk.
            model: Optional model name to use.
            device: Optional device for inference.
            dtype: Data type for inference.
            batch_size: Optional batch size for processing.
            chunk_length: Audio chunk length in seconds.
            progress_group_size: Chunks per progress update group. Defaults to
                :data:`~insanely_fast_whisper_rocm.utils.constants.DEFAULT_PROGRESS_GROUP_SIZE`
                when ``None``.
            language: Optional language code for processing.
            task: Task to perform ("transcribe" or "translate").
            return_timestamps_value: Whether/how to return timestamps.
            progress_cb: Optional progress callback for granular updates.
            cancellation_token: Cooperative cancellation token.

        Returns:
            dict[str, Any]: Transcription or translation payload.

        Raises:
            TranscriptionError: When the pipeline fails and fallback processing
                is unavailable or also fails.

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
        eff_progress_group_size = (
            progress_group_size
            if progress_group_size and progress_group_size > 0
            else constants.DEFAULT_PROGRESS_GROUP_SIZE
        )
        backend_config = self._create_backend_config(
            model=model_name,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            chunk_length=chunk_length,
            progress_group_size=eff_progress_group_size,
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

        # Create orchestrator using factory
        orchestrator = self.orchestrator_factory()

        def _warning_callback(message: str) -> None:
            """Log orchestrator recovery warnings.

            Args:
                message: The warning message from the orchestrator.
            """
            logger.warning("Orchestrator recovery action: %s", message)

        try:
            # Use orchestrator for robust transcription with OOM recovery.
            # It internally handles pipeline acquisition and retries.
            return orchestrator.run_transcription(
                audio_path=str(audio_file_path),
                backend_config=backend_config,
                task=task,
                language=language,
                timestamp_type=return_timestamps_value,
                progress_callback=progress_cb,
                warning_callback=_warning_callback,
                save_transcriptions=False,  # CLI usually doesn't need auto-save
                # to transcripts/
            )
        except OutOfMemoryError as oom:
            logger.error("CLI processing failed: Out of GPU memory even after retries.")
            err_msg = f"Insufficient memory for CLI processing: {str(oom)}"
            raise TranscriptionError(err_msg) from oom
        except TranscriptionError:
            raise


# Global facade instance for CLI use (exposes process_audio for
# both transcription and translation)
cli_facade = CLIFacade(check_file_exists=True)
