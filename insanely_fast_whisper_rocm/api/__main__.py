"""Main entrypoint for running the FastAPI application.

This script allows the API to be started directly using
`python -m insanely_fast_whisper_rocm.api`. It uses `uvicorn` to run the FastAPI
application created by the `create_app` factory.
"""

import uvicorn

from insanely_fast_whisper_rocm.utils import constants

if __name__ == "__main__":
    uvicorn.run(
        "insanely_fast_whisper_rocm.api.app:create_app",
        factory=True,
        host=constants.API_HOST,
        port=constants.API_PORT,
        reload=True,
    )
