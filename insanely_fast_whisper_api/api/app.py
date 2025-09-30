"""FastAPI application factory for the Insanely Fast Whisper API.

This module provides the main app factory function that creates and configures
the FastAPI application with all necessary routes and middleware.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.routing import APIRoute

from insanely_fast_whisper_api import __version__
from insanely_fast_whisper_api.api.middleware import add_middleware
from insanely_fast_whisper_api.api.routes import router as api_router
from insanely_fast_whisper_api.utils.constants import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    DEFAULT_MODEL,
    HF_TOKEN,
)
from insanely_fast_whisper_api.utils.download_hf_model import download_model_if_needed

logger = logging.getLogger(__name__)


async def run_startup_sequence(app: FastAPI) -> None:
    """Initialize application state and log configuration on startup.

    Args:
        app: FastAPI instance that is being started.
    """
    logger.info(
        "Attempting to download/verify Whisper model specified by WHISPER_MODEL "
        "or default..."
    )
    download_model_if_needed(
        model_name=DEFAULT_MODEL,
        hf_token=HF_TOKEN,
        custom_logger=logger,
    )
    logger.info("Model download/verification process for API startup complete.")

    logger.info("=" * 50)
    logger.info(f"Starting {API_TITLE} v{__version__}")
    logger.info(f"API Description: {API_DESCRIPTION}")
    logger.info("-" * 50)

    logger.info("Available endpoints:")
    for route in app.routes:
        if isinstance(route, APIRoute):
            methods = ", ".join(sorted(route.methods))
            logger.info("  %s %s", f"{methods:<10}", route.path)
            if route.description and logger.isEnabledFor(logging.DEBUG):
                logger.debug("    Description: %s", route.description)
    logger.info("=" * 50)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup sequence using FastAPI's lifespan support."""
    await run_startup_sequence(app)
    yield


def create_app() -> FastAPI:
    """Factory function to create and configure FastAPI application.

    This function implements the Factory pattern to create a properly configured
    FastAPI application with all necessary middleware, routes, and lifespan events.

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add middleware
    add_middleware(app)

    # Include routes
    app.include_router(api_router)

    return app
