"""Main entrypoint for running the FastAPI application.

This script allows the API to be started directly using `python -m insanely_fast_whisper_api.api`.
It uses `uvicorn` to run the FastAPI application created by the `create_app` factory.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "insanely_fast_whisper_api.api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8888,
        reload=True,
    )
