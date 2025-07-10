# Use an official Python runtime as a parent image
FROM python:3.10-slim

LABEL org.opencontainers.image.source https://github.com/beecave-homelab/insanely-fast-whisper-rocm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off
ENV TZ=Europe/Amsterdam
ENV ROCM_PATH=/opt/rocm
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0

# Install specific packages using pip
RUN apt-get update -y && apt-get upgrade -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \ 
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the working directory in the container
WORKDIR /app

# Copy the project config file for installing dependencies
COPY pyproject.toml /app/

# Install pdm (Python Development Master) via pip
RUN pip install --no-cache-dir pdm

# Use pdm to install all project dependencies (prod only)
RUN pdm install --prod

# Copy the OpenAPI spec file
COPY openapi.yaml /app/

# Copy the application source code
# This is needed for `pdm install` to build and install the local package.
# It assumes your main package source is in the 'insanely_fast_whisper_api' directory.
COPY ./insanely_fast_whisper_api /app/insanely_fast_whisper_api/

# Now, install the local package itself (handled by pdm above)
# RUN pip install -U pip
# RUN pip install --no-cache-dir .
# (Handled by pdm install above)

# After `pip install .`, the package `insanely_fast_whisper_api` and its CLI/modules
# should be available in the Python environment.

# Added in case Gradio is used and needs to be accessible; remove if not needed.
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Make port 8888 available to the world outside this container (standard for uvicorn)
EXPOSE 8888
EXPOSE 7860

# Define the command to run the application using uvicorn.
# This points to the `app` instance in your `main.py` inside the installed package.
CMD ["insanely-fast-whisper-api", "--host", "0.0.0.0", "--port", "8888"]
