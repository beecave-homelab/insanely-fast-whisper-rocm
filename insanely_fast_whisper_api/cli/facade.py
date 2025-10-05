"""CLI facade for simplified access to core ASR functionality."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from insanely_fast_whisper_api.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)
from insanely_fast_whisper_api.core.cancellation import CancellationToken
from insanely_fast_whisper_api.core.errors import TranscriptionError
from insanely_fast_whisper_api.core.pipeline import WhisperPipeline
from insanely_fast_whisper_api.core.progress import ProgressCallback
from insanely_fast_whisper_api.core.utils import convert_device_string
from insanely_fast_whisper_api.utils import constants

logger = logging.getLogger(__name__)


class CLIFacade:
    """Facade for CLI access to ASR functionality."""

    def __init__(
        self,
        *,
        backend_factory: type[HuggingFaceBackend] | None = None,
        pipeline_factory: type[WhisperPipeline] | None = None,
        check_file_exists: bool = False,
    ) -> None:
        """Initialize the CLI facade.

        Initializes internal backend cache and the last-used configuration so
        repeated calls with the same configuration can reuse the backend.

        Args:
            backend_factory: Optional factory used to create the ASR backend.
                Defaults to :class:`HuggingFaceBackend` (used in production).
            pipeline_factory: Optional factory for the pipeline class. Defaults
                to :class:`WhisperPipeline`. Tests can inject a stub to avoid
                touching the filesystem.
            check_file_exists: Whether to verify input audio paths exist. Disabled
                by default so tests and programmatic callers can provide synthetic
                paths. Production entry points should pass ``True`` to retain
                strict validation.
        """
        self.backend_factory = backend_factory
        self.pipeline_factory = pipeline_factory
        self.check_file_exists = check_file_exists

        self.backend: HuggingFaceBackend | None = None
        self.pipeline: WhisperPipeline | None = None
        self._current_config: HuggingFaceBackendConfig | None = None

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

    def process_audio(  # pylint: disable=too-many-arguments
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
                :data:`~insanely_fast_whisper_api.utils.constants.DEFAULT_PROGRESS_GROUP_SIZE`
                when ``None``.
            language: Optional language code for processing.
            task: Task to perform ("transcribe" or "translate").
            return_timestamps_value: Whether/how to return timestamps.
            progress_cb: Optional progress callback for granular updates.
            cancellation_token: Cooperative cancellation token.

        Returns:
            dict[str, Any]: Transcription or translation payload.

        Raises:
            RuntimeError: If the ASR backend cannot be initialized.
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

        backend_cls = self.backend_factory or HuggingFaceBackend
        pipeline_cls = self.pipeline_factory or WhisperPipeline

        backend_changed = self.backend is None or self._current_config != backend_config
        if backend_changed:
            self.backend = backend_cls(backend_config)
            if self.check_file_exists:
                self.pipeline = pipeline_cls(
                    asr_backend=self.backend,
                    storage_backend=None,
                    save_transcriptions=False,
                )
            else:
                self.pipeline = None
            self._current_config = backend_config
        elif self.pipeline is None and self.backend is not None:
            if self.check_file_exists:
                self.pipeline = pipeline_cls(
                    asr_backend=self.backend,
                    storage_backend=None,
                    save_transcriptions=False,
                )

        # Get language from config if not provided
        if language is None:
            language = config["language"]

        if self.backend is None:
            raise RuntimeError("ASR backend failed to initialize for CLI pipeline.")

        if self.pipeline is None or not self.check_file_exists:
            return self.backend.process_audio(
                audio_file_path=str(audio_file_path),
                language=language,
                task=task,
                return_timestamps_value=return_timestamps_value,
                progress_cb=progress_cb,
                cancellation_token=cancellation_token,
            )

        if return_timestamps_value == "word":
            timestamp_type = "word"
        elif return_timestamps_value:
            timestamp_type = "chunk"
        else:
            timestamp_type = "none"

        try:
            # CRITICAL FIX: Do NOT pass original_filename for CLI usage.
            # The pipeline will use the absolute path from audio_file_path,
            # which is required for stabilization to locate the audio file.
            # original_filename should only be used for uploaded files (API)
            # where we want to preserve the upload's original name.
            return self.pipeline.process(
                audio_file_path=str(audio_file_path),
                language=language,
                task=task,
                timestamp_type=timestamp_type,  # type: ignore[arg-type]
                original_filename=None,  # Let pipeline use absolute path
                progress_callback=progress_cb,
                cancellation_token=cancellation_token,
            )
        except TranscriptionError as exc:
            missing_file = "audio file not found" in str(exc).lower()
            if self.check_file_exists or not missing_file:
                raise
            logger.debug(
                "Falling back to backend-only processing for %s due to missing"
                " file: %s",
                audio_file_path,
                exc,
            )
            return self.backend.process_audio(
                audio_file_path=str(audio_file_path),
                language=language,
                task=task,
                return_timestamps_value=return_timestamps_value,
                progress_cb=progress_cb,
                cancellation_token=cancellation_token,
            )


# Global facade instance for CLI use (exposes process_audio for
# both transcription and translation)
cli_facade = CLIFacade(check_file_exists=True)
