"""
Insanely Fast Whisper - High-performance speech-to-text transcription tool.

Use the transcribe command to convert audio files to text using Whisper ASR.
"""

import time
import json
import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import click
from transformers import pipeline, logging as transformers_logging

from insanely_fast_whisper_api import constants

# Suppress all warnings from transformers
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress parallelism warning


# Custom exceptions
class TranscriptionError(click.ClickException):
    """Raised when transcription fails."""

    pass


class DeviceNotFoundError(click.ClickException):
    """Raised when specified device is not available."""

    pass


def convert_device_string(device_id: str) -> str:
    """Convert device ID to proper device string format."""
    if device_id.isdigit():
        return f"cuda:{device_id}"
    elif device_id == "mps":
        return "mps"
    elif device_id == "cpu":
        return "cpu"
    else:
        # If none of the above, assume it's already a proper device string (e.g., "cuda:0")
        return device_id


def get_env_config():
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


def run_asr_pipeline(
    audio_file_path,
    model=None,
    device=None,
    dtype="float16",
    batch_size=None,
    better_transformer=False,
    chunk_length=30,
    language=None,
    task="transcribe",
) -> Dict[str, Any]:
    """
    Runs ASR on an audio file and returns the transcription text and a list of chunks.

    :param audio_file_path: Path to the audio file on disk
    :param model: The model name, e.g. 'openai/whisper-base'
    :param device: The device, e.g. 'cuda:0' or 'cpu'
    :param dtype: 'float32' or 'float16'
    :param batch_size: batch size
    :param better_transformer: whether to use BetterTransformer
    :param chunk_length: length in seconds for chunking
    :param language: Language code for transcription (e.g., 'en', 'fr')
    :param task: The task to perform (e.g., 'transcribe' or 'translate')
    :return: a dict with keys "text" and "chunks", plus optional metadata
    """
    # Get config from environment with defaults
    config = get_env_config()

    # Use provided parameters or fall back to config values
    model = model or config["model"]
    device = convert_device_string(device) if device else config["device"]
    batch_size = min(
        max(batch_size or config["batch_size"], constants.MIN_BATCH_SIZE),
        constants.MAX_BATCH_SIZE,
    )
    language = language or config["language"]

    # Validate device availability
    if "cuda" in device and not torch.cuda.is_available():
        raise DeviceNotFoundError(
            f"CUDA device {device} requested but CUDA is not available. Use 'cpu' instead."
        )
    elif device == "mps" and not torch.backends.mps.is_available():
        raise DeviceNotFoundError(
            "MPS device requested but MPS (Apple Silicon) is not available. Use 'cpu' instead."
        )

    click.secho(f"\nDevice: {device}", fg="blue")

    # Initialize the ASR pipeline with specific generation config
    model_kwargs = {
        "model": model,
        "device": device,
        "torch_dtype": torch.float16 if dtype == "float16" else torch.float32,
    }

    try:
        with click.progressbar(
            length=2,
            label="Loading model",
            show_eta=False,
            fill_char="█",
            empty_char="░",
            width=40,
        ) as bar:
            # Load model
            asr_pipe = pipeline("automatic-speech-recognition", **model_kwargs)
            bar.update(1)

            # Apply BetterTransformer if requested
            if better_transformer:
                asr_pipe.model = asr_pipe.model.to_bettertransformer()
            bar.update(1)

        click.secho("\nStarting transcription...", fg="green")
        start_time = time.perf_counter()

        pipeline_kwargs = {
            "chunk_length_s": chunk_length,
            "batch_size": batch_size,
            "return_timestamps": True,
            "generate_kwargs": {
                "task": task,
                "no_repeat_ngram_size": 3,
                "temperature": 0,
            },
        }

        # Only add language if specifically set
        if language and language.lower() != "none":
            pipeline_kwargs["language"] = language

        outputs = asr_pipe(audio_file_path, **pipeline_kwargs)

    except Exception as e:
        raise TranscriptionError(f"Failed to transcribe audio: {str(e)}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Format the results to mimic OpenAI's style
    result = {
        "text": outputs["text"].strip(),  # Strip any extra whitespace
        "chunks": outputs["chunks"],
        "runtime_seconds": round(elapsed_time, 2),
        "config_used": {
            "model": model,
            "device": device,
            "batch_size": batch_size,
            "language": language or "auto",
            "better_transformer": better_transformer,
            "dtype": dtype,
        },
    }

    return result


@click.group()
@click.version_option(version=constants.API_VERSION, prog_name=constants.API_TITLE)
def cli():
    """Insanely Fast Whisper - High-performance speech-to-text transcription tool.

    Use the transcribe command to convert audio files to text using Whisper ASR.
    """
    pass


@cli.command(
    short_help="""Transcribe audio files
    [--model, --device, --dtype, --batch-size,
    --better-transformer, --chunk-length,
    --language, --output --help]"""
)
@click.argument(
    "audio_file", type=click.Path(exists=True, path_type=Path), metavar="AUDIO_FILE"
)
@click.option(
    "--model",
    "-m",
    help="Model name to use for transcription (e.g., distil-whisper/distil-large-v3)",
    show_default=True,
    default=constants.DEFAULT_MODEL,
)
@click.option(
    "--device",
    "-d",
    help="Device for inference (cuda:0, cpu, mps)",
    show_default=True,
    default=constants.DEFAULT_DEVICE,
)
@click.option(
    "--dtype",
    type=click.Choice(["float16", "float32"]),
    default="float16",
    help="Data type for model inference",
    show_default=True,
)
@click.option(
    "--batch-size",
    "-b",
    type=click.IntRange(constants.MIN_BATCH_SIZE, constants.MAX_BATCH_SIZE),
    default=constants.DEFAULT_BATCH_SIZE,
    help=f"Batch size for processing ({constants.MIN_BATCH_SIZE}-{constants.MAX_BATCH_SIZE})",
    show_default=True,
)
@click.option(
    "--better-transformer/--no-better-transformer",
    default=False,
    help="Use BetterTransformer for faster inference",
    show_default=True,
)
@click.option(
    "--chunk-length",
    "-c",
    default=30,
    type=int,
    help="Audio chunk length in seconds",
    show_default=True,
)
@click.option(
    "--language",
    "-l",
    help="Language code (en, fr, de, None=auto)",
    show_default=True,
    default=constants.DEFAULT_LANGUAGE,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Save detailed results to JSON file",
)
def transcribe(
    audio_file: Path,
    model: str,
    device: str,
    dtype: str,
    batch_size: int,
    better_transformer: bool,
    chunk_length: int,
    language: str,
    output: Optional[Path],
) -> None:
    """Transcribe audio files to text using Whisper ASR.
    
    AUDIO_FILE: Path to the audio file to transcribe
    
    Supported formats: {formats}
    
    Environment Variables:
      WHISPER_MODEL      Default model to use
      WHISPER_DEVICE     Default device (0, 1, mps)
      WHISPER_BATCH_SIZE Default batch size
      WHISPER_LANGUAGE   Default language code
    
    Examples:
      # Basic usage
      transcribe audio.mp3
      
      # Specify model and device
      transcribe --model openai/whisper-large-v3 --device cuda:0 audio.mp3
      
      # Set language and save results
      transcribe --language en --output results.json audio.mp3
    """.format(
        formats=", ".join(sorted(constants.SUPPORTED_AUDIO_FORMATS))
    )

    # Validate audio file format
    if audio_file.suffix.lower() not in constants.SUPPORTED_AUDIO_FORMATS:
        supported_formats = ", ".join(constants.SUPPORTED_AUDIO_FORMATS)
        raise click.BadParameter(
            f"Unsupported audio format: {audio_file.suffix}. Supported formats: {supported_formats}"
        )

    try:
        result = run_asr_pipeline(
            str(audio_file),
            model=model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            better_transformer=better_transformer,
            chunk_length=chunk_length,
            language=language,
        )

        if output:
            # Write full results to file
            with open(output, "w") as f:
                json.dump(result, f, indent=2)
            click.secho(f"Full results written to {output}", fg="green")

        # Always print the transcription to stdout
        click.secho("\nTranscription:", bold=True)
        click.echo(result["text"])
        click.secho(
            f"\nProcessing time: {result['runtime_seconds']} seconds", fg="blue"
        )

        # Print configuration used
        click.secho("\nConfiguration used:", fg="yellow")
        for key, value in result["config_used"].items():
            click.echo(f"  {key}: {value}")

    except (TranscriptionError, DeviceNotFoundError) as e:
        raise e
    except Exception as e:
        raise click.ClickException(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    cli()
