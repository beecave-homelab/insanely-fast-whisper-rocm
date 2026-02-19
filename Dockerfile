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

# Copy requirements file for installing dependencies
# COPY requirements-rocm-v6-4-1.txt .
COPY requirements-rocm-v7-0.txt .
COPY .python-version .

# Install project dependencies using pip
RUN pip install --no-cache-dir -r requirements-rocm-v7-0.txt

# Copy the OpenAPI spec file
COPY openapi.yaml /app/

# Copy the application source code
# This is needed for `pdm install` to build and install the local package.
# It assumes your main package source is in the 'insanely_fast_whisper_rocm' directory.
# Copy the application source code and project metadata
COPY pyproject.toml /app/
COPY ./insanely_fast_whisper_rocm /app/insanely_fast_whisper_rocm/

# Install the local package itself
RUN pip install --no-cache-dir .

# After `pip install .`, the package `insanely_fast_whisper_rocm` and its CLI/modules
# should be available in the Python environment.

# Added in case Gradio is used and needs to be accessible; remove if not needed.
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Expose default internal ports (API/WebUI). Actual bindings are controlled by Compose.
EXPOSE 8888
EXPOSE 7860

# Use the package entrypoint so host/port are controlled by env vars (API_HOST/API_PORT).
CMD ["insanely-fast-whisper-rocm"]
