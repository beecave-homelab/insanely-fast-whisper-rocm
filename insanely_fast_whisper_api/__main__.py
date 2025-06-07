"""Entry point for running the package as a module.

This module allows running the FastAPI application using:
python -m insanely_fast_whisper_api
"""

import argparse
import logging
import logging.config
import os
import time
from pathlib import Path

try:
    import uvicorn
except ImportError:
    # uvicorn is only needed when running the server, not for all imports
    pass

import yaml


def setup_timezone():
    """Set the timezone for the application."""
    # Set timezone to Europe/Amsterdam
    os.environ["TZ"] = "Europe/Amsterdam"
    time.tzset()


def load_logging_config(verbose: bool = False) -> dict:
    """Load and configure logging from YAML file.

    Args:
        verbose: Whether to enable verbose logging

    Returns:
        dict: The configured logging dictionary
    """
    # Ensure timezone is set before loading config
    setup_timezone()

    # Load the YAML config
    config_path = Path(__file__).parent / "logging_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Adjust log levels based on verbosity
    if verbose:
        config["root"]["level"] = "DEBUG"
        for logger in config["loggers"].values():
            logger["level"] = "DEBUG"

    return config


def main():
    """Run the FastAPI application using uvicorn."""
    parser = argparse.ArgumentParser(
        description="Run the Insanely Fast Whisper API server"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    # Load and apply logging configuration
    log_config = load_logging_config(verbose=args.verbose)
    logging.config.dictConfig(log_config)

    # Get logger after configuration
    logger = logging.getLogger("insanely_fast_whisper_api")

    # Start the server
    logger.info("Starting Uvicorn server...")
    uvicorn.run(
        "insanely_fast_whisper_api.main:app",
        host="0.0.0.0",
        port=8888,
        reload=True,
        log_level="debug" if args.verbose else "info",
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
