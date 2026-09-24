# syntax=docker/dockerfile:1

ARG ROCM_PYTORCH_BASE=rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0
FROM ${ROCM_PYTORCH_BASE}

LABEL org.opencontainers.image.source https://github.com/beecave-homelab/insanely-fast-whisper-rocm

ARG HSA_OVERRIDE_GFX_VERSION=

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off
ENV TZ=Europe/Amsterdam
ENV ROCM_PATH=/opt/rocm
ENV HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}
# Skip MIOpen JIT kernel compilation — the slim image lacks rocrand headers
# needed by MIOpenDropoutHIP.cpp.  Mode 2 = database-only (no JIT).
ENV MIOPEN_FIND_MODE=2

# Install specific packages using pip
RUN apt-get update -y && apt-get upgrade -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the working directory in the container
WORKDIR /app

# Copy the fully resolved runtime export first for Docker layer caching.
COPY requirements-container.txt /app/

# Install the resolved runtime package set without dependency resolution. This
# preserves the full ROCm PyTorch wheel and excludes development-only tooling.
RUN pip install --no-cache-dir --no-deps -r /app/requirements-container.txt

# Copy the OpenAPI spec file
COPY openapi.yaml /app/

# Copy the application source code
# This is needed for `pdm install` to build and install the local package.
# It assumes your main package source is in the 'insanely_fast_whisper_rocm' directory.
# Copy the application source code and project metadata
COPY pyproject.toml /app/
COPY ./insanely_fast_whisper_rocm /app/insanely_fast_whisper_rocm/

# Install the local package itself (no-deps: all dependencies are already
# installed from the PDM-generated requirements file which excludes torchcodec
# and other ROCm-incompatible packages).
RUN pip install --no-cache-dir --no-deps .

# After `pip install .`, the package `insanely_fast_whisper_rocm` and its CLI/modules
# should be available in the Python environment.

# Added in case Gradio is used and needs to be accessible; remove if not needed.
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV TORCHAUDIO_USE_SOUNDFILE=1

# Expose default internal ports (API/WebUI). Actual bindings are controlled by Compose.
EXPOSE 8888
EXPOSE 7860

# Use the package entrypoint so host/port are controlled by env vars (API_HOST/API_PORT).
CMD ["insanely-fast-whisper-rocm"]
