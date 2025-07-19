"""WebUI application logic for Insanely Fast Whisper API.

This module contains the core function for launching the Gradio WebUI.
"""

import logging
import sys

import click
from insanely_fast_whisper_api.utils.constants import DEFAULT_MODEL

from insanely_fast_whisper_api.utils.download_hf_model import download_model_if_needed
from insanely_fast_whisper_api.webui.ui import create_ui_components

# Configure logger
logger = logging.getLogger("insanely_fast_whisper_api.webui.app")


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
    download_model_if_needed(model_name=model, custom_logger=logger)
    logger.info("Model download/verification process for WebUI complete.")

    # Determine which model value should be shown in the UI
    ui_default_model = model if model else DEFAULT_MODEL

    # Create the interface with the chosen default model
    iface = create_ui_components(default_model=ui_default_model)

    # Launch the interface
    logger.info("Launching WebUI on %s:%s", host, port)
    iface.launch(
        server_name=host,
        server_port=port,
        share=share,
    )
