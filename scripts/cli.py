"""
Command-line interface for the Insanely Fast Whisper ROCm project.

This module provides a user-friendly CLI for transcribing audio files using the
Whisper model optimized for AMD GPUs with ROCm support.
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.core.transcription import (
    TranscriptionConfig,
    TranscriptionEngine,
    TranscriptionError,
    DeviceNotFoundError,
)
from src.core.file_handlers import FileValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize console for rich output
console = Console()

# File validator for input validation
file_validator = FileValidator()


def validate_audio_file(ctx, param, value):
    """Validate that the input file exists and is a supported audio format."""
    if value is None:
        return None

    path = Path(value)

    # Check if file exists
    if not path.exists():
        raise click.BadParameter(f"File does not exist: {value}")

    # Check if it's a file
    if not path.is_file():
        raise click.BadParameter(f"Not a file: {value}")

    # Check if it's a supported audio format
    is_valid, msg = file_validator.validate_file(
        path,
        check_exists=True,
        check_type=True,
    )

    if not is_valid:
        raise click.BadParameter(f"Unsupported audio format: {msg}")

    return path


def validate_output_file(ctx, param, value):
    """Validate the output file path and ensure the directory exists."""
    if value is None:
        return None

    path = Path(value)

    # Ensure the parent directory exists
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise click.BadParameter(f"Invalid output directory: {e}")

    return path


@click.group()
def cli():
    """Insanely Fast Whisper ROCm - Fast and efficient speech recognition."""
    pass


@cli.command()
@click.argument(
    "audio_file",
    type=click.Path(exists=True, path_type=Path, dir_okay=False, resolve_path=True),
    callback=validate_audio_file,
)
@click.option(
    "--model",
    default="openai/whisper-large-v3",
    help="Name of the Whisper model to use for transcription.",
    show_default=True,
)
@click.option(
    "--device",
    default="cuda",
    help="Device to run the model on (e.g., 'cuda', 'cpu', 'mps').",
    show_default=True,
)
@click.option(
    "--batch-size",
    type=int,
    default=4,
    help="Number of audio chunks to process in parallel.",
    show_default=True,
)
@click.option(
    "--chunk-length",
    type=int,
    default=30,
    help="Length of audio chunks to process at a time (in seconds).",
    show_default=True,
)
@click.option(
    "--compute-type",
    type=click.Choice(["float16", "float32"], case_sensitive=False),
    default="float16",
    help="Compute type for model inference.",
    show_default=True,
)
@click.option(
    "--language",
    type=str,
    default=None,
    help="Language of the audio (if known, helps with transcription accuracy).",
)
@click.option(
    "--no-better-transformer",
    is_flag=True,
    default=False,
    help="Disable BetterTransformer for inference.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    help="Temperature for sampling. Lower values make the output more deterministic.",
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False, writable=True, resolve_path=True),
    default=None,
    help="Output file path for the transcription results (JSON format).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging.",
)
def transcribe(
    audio_file: Path,
    model: str,
    device: str,
    batch_size: int,
    chunk_length: int,
    compute_type: str,
    language: Optional[str],
    no_better_transformer: bool,
    temperature: float,
    output: Optional[Path],
    verbose: bool,
):
    """Transcribe an audio file using the Whisper model."""
    # Configure logging level
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(log_level)

    # Initialize progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Initializing...", total=None)

        try:
            # Create configuration
            config = TranscriptionConfig(
                model_name=model,
                device=device,
                batch_size=batch_size,
                chunk_length_s=chunk_length,
                compute_type=compute_type,
                language=language,
                better_transformer=not no_better_transformer,
                temperature=temperature,
                verbose=verbose,
            )

            # Initialize transcription engine
            engine = TranscriptionEngine(config)

            # Update progress
            progress.update(task, description="Transcribing audio...")

            # Perform transcription
            result = engine.transcribe_audio(str(audio_file))

            # Prepare output data
            output_data = result.dict()

            # Save to file if output path is provided
            if output:
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                console.print(f"\n[green]✓ Transcription saved to: {output}")

            # Print results
            console.print("\n[bold]Transcription Complete[/bold]")
            console.print(f"\n[bold]Language:[/bold] {result.language}")
            console.print(f"[bold]Duration:[/bold] {result.duration:.2f} seconds")
            console.print(f"[bold]Model:[/bold] {result.model}")
            console.print("\n[bold]Transcription:[/bold]")
            console.print(result.text)

            # Print segments if verbose
            if verbose and result.chunks:
                console.print("\n[bold]Segments:[/bold]")
                for i, segment in enumerate(result.chunks, 1):
                    console.print(f"\n[bold]Segment {i}:[/bold]")
                    console.print(f"  [dim]Start:[/dim] {segment['start']:.2f}s")
                    console.print(f"  [dim]End:[/dim] {segment['end']:.2f}s")
                    if 'avg_logprob' in segment:
                        console.print(f"  [dim]Confidence:[/dim] {segment['avg_logprob']:.2f}")
                    console.print(f"  [dim]Text:[/dim] {segment['text']}")

            return 0

        except DeviceNotFoundError as e:
            console.print(f"[bold red]Device error:[/bold red] {str(e)}")
            if verbose:
                console.print(f"\n[bold]Traceback:[/bold]\n{traceback.format_exc()}")
            return 1
            
        except TranscriptionError as e:
            console.print(f"[bold red]Transcription error:[/bold red] {str(e)}")
            if verbose:
                console.print(f"\n[bold]Traceback:[/bold]\n{traceback.format_exc()}")
            return 1
            
        except Exception as e:
            console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")
            if verbose:
                console.print(f"\n[bold]Traceback:[/bold]\n{traceback.format_exc()}")
            return 1


def main() -> int:
    """Entry point for the CLI application.
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    try:
        cli(prog_name="insanely-fast-whisper-rocm")
        return 0
    except Exception as e:
        console.print(f"[red]Error: {str(e)}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            console.print(f"\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
