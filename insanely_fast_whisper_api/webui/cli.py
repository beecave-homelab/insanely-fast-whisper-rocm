"""CLI entrypoint for Insanely Fast Whisper API WebUI.

This module provides a simple command-line interface for launching the WebUI.
"""

import logging

# Standard library imports
import sys
import click

# import gradio as gr # This was unused

from insanely_fast_whisper_api.utils.download_hf_model import download_model_if_needed
from insanely_fast_whisper_api.webui.ui import create_ui_components

# Configure logger
logger = logging.getLogger("insanely_fast_whisper_api.webui.cli")


@click.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind the server to. Default is 0.0.0.0 (all interfaces).",
)
@click.option(
    "--port",
    default=7860,
    type=int,
    help="Port to bind the server to. Default is 7860.",
)
@click.option(
    "--share",
    is_flag=True,
    help="Create a public URL. This is useful for running on Colab or sharing with others.",
)
@click.option(
    "--model",
    help="Whisper model to use. If not specified, uses the WHISPER_MODEL env var or default.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging.",
)
def launch_webui(host: str, port: int, share: bool, model: str, debug: bool):
    """Launch the Insanely Fast Whisper WebUI."""
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    logger.info("Starting Insanely Fast Whisper WebUI...")

    # Ensure model is downloaded/verified before UI setup
    logger.info("Attempting to download/verify Whisper model for WebUI...")
    # download_model_if_needed will use WHISPER_MODEL and HUGGINGFACE_TOKEN env vars by default
    download_model_if_needed(model_name=model, custom_logger=logger)
    logger.info("Model download/verification process for WebUI complete.")

    # Create the interface
    iface = create_ui_components()

    # Launch the interface
    logger.info("Launching WebUI on %s:%s", host, port)
    iface.launch(
        server_name=host,
        server_port=port,
        share=share,
    )


if __name__ == "__main__":
    # Call with default parameters when run directly
    # launch_webui(host="0.0.0.0", port=7860, share=False, model=None, debug=False)
    launch_webui()  # pylint: disable=no-value-for-parameter
