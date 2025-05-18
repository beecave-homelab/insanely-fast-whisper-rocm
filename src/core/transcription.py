"""
Core module for handling audio transcription using Whisper models.

This module provides functionality for transcribing audio files using the
Whisper model optimized for AMD GPUs with ROCm support.
"""

import os
import json
import logging
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator


class TranscriptionConfig(BaseModel):
    """Configuration for the transcription process."""

    model_name: str = Field(
        default="openai/whisper-large-v3",
        description="Name of the Whisper model to use for transcription.",
    )
    batch_size: int = Field(
        default=4, gt=0, description="Number of audio chunks to process in parallel."
    )
    device: str = Field(
        default="cuda", description="Device to run the model on (e.g., 'cuda', 'cpu')."
    )
    compute_type: str = Field(
        default="float16",
        description="Compute type for model inference (e.g., 'float16', 'int8').",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language of the audio (if known, helps with transcription accuracy).",
    )
    verbose: bool = Field(
        default=False, description="Enable verbose logging during transcription."
    )
    chunk_length_s: int = Field(
        default=30,
        gt=0,
        description="Length of audio chunks to process at a time (in seconds).",
    )
    no_repeat_ngram_size: int = Field(
        default=1,
        ge=1,
        description="If set to int > 0, all ngrams of that size can only occur once.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The value used to modulate the next token probabilities.",
    )
    better_transformer: bool = Field(
        default=True,
        description="Whether to use BetterTransformer for faster inference.",
    )

    @validator("batch_size")
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError("Batch size must be greater than 0")
        return v


class TranscriptionResult(BaseModel):
    """Result of a transcription operation."""

    text: str = Field(..., description="The full transcribed text.")
    language: str = Field(..., description="Detected language of the audio.")
    duration: float = Field(..., description="Duration of the audio in seconds.")
    chunks: List[Dict] = Field(
        default_factory=list,
        description="List of transcribed chunks with timestamps and confidence scores.",
    )
    model: str = Field(..., description="Name of the model used for transcription.")

    class Config:
        json_encoders = {
            # Custom JSON encoder for numpy.float32 which might be in chunks
            type(lambda: 0.0): lambda v: float(v)
        }


class TranscriptionError(Exception):
    """Base exception for transcription errors."""

    pass


class DeviceNotFoundError(TranscriptionError):
    """Raised when the specified device is not available."""

    pass


class TranscriptionEngine:
    """Core engine for transcribing audio using Whisper models."""

    def __init__(self, config: TranscriptionConfig):
        """Initialize the transcription engine.

        Args:
            config: Configuration for the transcription process.

        Raises:
            DeviceNotFoundError: If the specified device is not available.
            ImportError: If required dependencies are not installed.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._pipeline = None

        # Set environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Validate device
        self._validate_device()

        # Initialize the pipeline
        self._init_pipeline()

    def _validate_device(self) -> None:
        """Validate that the specified device is available."""
        import torch

        device = self.config.device.lower()
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise DeviceNotFoundError(
                    "CUDA is not available. Please check your CUDA installation."
                )
        elif device.startswith("mps"):
            if (
                not getattr(torch.backends, "mps", None)
                or not torch.backends.mps.is_available()
            ):
                raise DeviceNotFoundError(
                    "MPS (Metal Performance Shaders) is not available on this system."
                )
        elif device != "cpu":
            raise DeviceNotFoundError(f"Unsupported device: {device}")

    def _init_pipeline(self) -> None:
        """Initialize the Hugging Face pipeline for speech recognition."""
        from transformers import pipeline
        import torch
        from tqdm.auto import tqdm

        # Show progress for model loading
        with tqdm(
            desc=f"Loading model {self.config.model_name}",
            unit="step",
            disable=not self.config.verbose,
        ) as pbar:

            def progress_callback(step, total, **kwargs):
                if pbar.total != total:
                    pbar.reset(total=total)
                pbar.update(1)

            # Initialize the pipeline
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.config.model_name,
                device=self.config.device,
                torch_dtype=(
                    torch.float16
                    if self.config.compute_type == "float16"
                    else torch.float32
                ),
                model_kwargs={"use_better_transformer": self.config.better_transformer},
            )

            # Set model parameters
            self._pipeline.model.config.forced_decoder_ids = None

            # Set generation config
            if hasattr(self._pipeline.model, "generation_config"):
                self._pipeline.model.generation_config.update(
                    task="transcribe",
                    language=self.config.language,
                    no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                    temperature=self.config.temperature,
                )

    def transcribe_audio(self, audio_file_path: str) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_file_path: Path to the audio file to transcribe.

        Returns:
            TranscriptionResult containing the transcription results.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            TranscriptionError: If there is an error during transcription.
        """
        import time
        from pathlib import Path

        # Check if file exists
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        try:
            # Start timer
            start_time = time.time()

            # Transcribe the audio file
            result = self._pipeline(
                str(audio_path),
                chunk_length_s=self.config.chunk_length_s,
                batch_size=self.config.batch_size,
                return_timestamps=True,
                generate_kwargs={
                    "task": "transcribe",
                    "language": self.config.language,
                    "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
                    "temperature": self.config.temperature,
                },
            )

            # Calculate runtime
            runtime_seconds = time.time() - start_time

            # Format the result
            return TranscriptionResult(
                text=result.get("text", ""),
                language=result.get("language", self.config.language or "en"),
                duration=runtime_seconds,
                chunks=result.get("chunks", []),
                model=self.config.model_name,
            )

        except Exception as e:
            self.logger.error(f"Error during transcription: {str(e)}", exc_info=True)
            raise TranscriptionError(f"Failed to transcribe audio: {str(e)}") from e

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resources if needed
        pass

    def transcribe_file(
        self,
        input_file: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file using the Whisper model.

        Args:
            input_file: Path to the input audio file.
            output_file: Optional path to save the transcription result as JSON.

        Returns:
            TranscriptionResult containing the transcription results.

        Raises:
            FileNotFoundError: If the input file does not exist.
            subprocess.CalledProcessError: If the transcription process fails.
        """
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Starting transcription of {input_file}")
        self.logger.debug(f"Using model: {self.config.model_name}")

        try:
            # Build the command for insanely-fast-whisper
            cmd = [
                "insanely-fast-whisper",
                "--file-name",
                str(input_file),
                "--model",
                self.config.model_name,
                "--batch-size",
                str(self.config.batch_size),
                "--device",
                self.config.device,
                "--compute-type",
                self.config.compute_type,
            ]

            if self.config.language:
                cmd.extend(["--language", self.config.language])

            if output_file:
                cmd.extend(["--transcript-path", str(output_file)])

            self.logger.debug(f"Running command: {' '.join(cmd)}")

            # Run the transcription
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Parse the output
            try:
                output = json.loads(result.stdout)
                transcription_result = TranscriptionResult(
                    text=output.get("text", ""),
                    language=output.get("language", ""),
                    duration=output.get("duration", 0.0),
                    chunks=output.get("chunks", []),
                    model=self.config.model_name,
                )

                # Save to output file if specified
                if output_file:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(transcription_result.json(indent=2))

                self.logger.info("Successfully transcribed %s", input_file)
                return transcription_result

            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse transcription output: {e}"
                self.logger.error(error_msg)
                self.logger.debug("Command output: %s", result.stdout)
                raise RuntimeError(error_msg) from e

        except subprocess.CalledProcessError as e:
            error_msg = f"Transcription failed with return code {e.returncode}"
            self.logger.error(error_msg)
            self.logger.error("Error output: %s", e.stderr)
            raise TranscriptionError(error_msg) from e

        except Exception as e:
            error_msg = f"Error during transcription: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise TranscriptionError(error_msg) from e

    def transcribe_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: Optional[List[str]] = None,
    ) -> Dict[str, Union[TranscriptionResult, Exception]]:
        """Transcribe all audio files in a directory.

        Args:
            input_dir: Directory containing audio files to transcribe.
            output_dir: Directory to save transcription results.
            extensions: List of file extensions to process (e.g., ['.wav', '.mp3']).
                      If None, common audio extensions are used.

        Returns:
            Dictionary mapping input file paths to their transcription results or exceptions.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if extensions is None:
            extensions = [
                ".wav",
                ".mp3",
                ".m4a",
                ".flac",
                ".ogg",
                ".mp4",
                ".m4v",
                ".mkv",
            ]

        results = {}

        for ext in extensions:
            for input_file in input_dir.glob(f"*{ext}"):
                output_file = output_dir / f"{input_file.stem}.json"

                try:
                    result = self.transcribe_file(input_file, output_file)
                    results[str(input_file)] = result
                except Exception as e:
                    results[str(input_file)] = e
                    self.logger.error(f"Failed to transcribe {input_file}: {str(e)}")

        return results


def transcribe_file(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    model_name: str = "openai/whisper-large-v3",
    batch_size: int = 4,
    device: str = "cuda",
    compute_type: str = "float16",
    language: Optional[str] = None,
    verbose: bool = False,
) -> TranscriptionResult:
    """Convenience function for transcribing a single file with default settings.

    Args:
        input_file: Path to the input audio file.
        output_file: Optional path to save the transcription result as JSON.
        model_name: Name of the Whisper model to use.
        batch_size: Number of audio chunks to process in parallel.
        device: Device to run the model on (e.g., 'cuda', 'cpu')..
        compute_type: Compute type for model inference.
        language: Language of the audio (if known).
        verbose: Enable verbose logging.

    Returns:
        TranscriptionResult containing the transcription results.
    """
    config = TranscriptionConfig(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        compute_type=compute_type,
        language=language,
        verbose=verbose,
    )

    processor = TranscriptionProcessor(config)
    return processor.transcribe_file(input_file, output_file)
