"""Main CLI entry point for insanely-fast-whisper.

This module provides the main CLI group and coordinates all CLI functionality.
"""

import logging
import os
import sys
import warnings

import click
from transformers import logging as transformers_logging

from insanely_fast_whisper_api.cli.commands import transcribe
from insanely_fast_whisper_api.utils import constants

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Suppress all warnings from transformers
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress parallelism warning


@click.group()
@click.version_option(version=constants.API_VERSION, prog_name=constants.API_TITLE)
def cli():
    """
    🎵 Insanely Fast Whisper API - CLI Tool

    A high-performance CLI tool for audio transcription using Whisper models.
    """


# Add commands to the CLI group
cli.add_command(transcribe)


def main():
    """Main entry point for the CLI application."""
    cli()


if __name__ == "__main__":
    main()
