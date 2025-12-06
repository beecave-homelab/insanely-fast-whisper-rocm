"""Middleware for the FastAPI application.

This module contains middleware functions for cross-cutting concerns
such as request timing and logging.
"""

import logging
import time

from fastapi import FastAPI, Request

logger = logging.getLogger(__name__)


async def log_request_timing(request: Request, call_next):
    """Log timing information for each request.

    Args:
        request: The incoming HTTP request
        call_next: The next middleware or route handler

    Returns:
        The HTTP response
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    end_time = time.perf_counter()

    duration = end_time - start_time
    logger.info(
        "Request %s %s completed in %.2fs with status %s",
        request.method,
        request.url.path,
        duration,
        response.status_code,
    )
    return response


def add_middleware(app: FastAPI) -> None:
    """Add all middleware to the FastAPI application.

    Args:
        app: The FastAPI application instance
    """
    app.middleware("http")(log_request_timing)
