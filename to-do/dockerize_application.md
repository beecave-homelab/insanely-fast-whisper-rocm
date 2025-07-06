# To-Do: Dockerize Insanely Fast Whisper API

This plan outlines the steps to containerize the application using Docker and Docker Compose. This setup will allow testing the application in a Docker container without requiring local package installations, while ensuring AMD GPU (ROCm) support is correctly configured.

## Tasks

- [x] **Create/Update `.dockerignore`**
  - Path: `/home/elvee/Local-AI/insanely-fast-whisper-api/.dockerignore`
  - Action: Add entries to exclude unnecessary files and directories from the Docker build context to optimize build time and image size.
    - `.git`
    - `.gitignore`
    - `.idea/`
    - `.vscode/`
    - `.venv/`
    - `__pycache__/`
    - `*.pyc`
    - `*.log`
    - `*.lock`
    - `dist/`
    - `build/`
    - `*.egg-info/`
    - `.DS_Store`
    - `to-do/`
    - `*.md` # Excluding markdown files unless explicitly needed in the image

- [x] **Create/Update `Dockerfile`**
  - Path: `/home/elvee/Local-AI/insanely-fast-whisper-api/Dockerfile`
  - Action: Define the Docker image for the "insanely-fast-whisper-api" application.
    - Base Image: Use an official Python 3.10 image (e.g., `python:3.10-slim`).
    - Environment Variables: Set `PYTHONUNBUFFERED=1` for direct output of Python print statements and `PIP_NO_CACHE_DIR=off`.
    - Working Directory: Create and set `/app` as the working directory.
    - Install Dependencies:
      - Copy `requirements.txt`, `requirements-onnxruntime-rocm.txt`, and `requirements-rocm.txt` to the `/app/` directory.
      - Install project dependencies using `pip install --no-cache-dir -r requirements.txt`, then `pip install --no-cache-dir -r requirements-onnxruntime-rocm.txt`, and then `pip install --no-cache-dir -r requirements-rocm.txt`.
      - After installing dependencies from requirements files, copy `pyproject.toml` to `/app/pyproject.toml`.
      - Install the application itself using `pip install --no-cache-dir .`. This step required the following modifications to `pyproject.toml` to ensure a successful build:
        - Resolved Hatchling build errors by adding `[tool.hatch.metadata]` and setting `allow-direct-references = true`.
        - Corrected the Python version specifier by initially setting `project.requires-python = "==3.10"`, and then refining it to `project.requires-python = "~=3.10"` to resolve a pip version matching issue.
        - Ensured that complex dependencies like `torch` (with direct URLs) were removed from `pyproject.toml`'s main dependency list if they are handled by `requirements*.txt` files, to prevent re-download attempts and conflicts.
    - Copy Application Code: Copy the application source code (e.g., `insanely_fast_whisper_api` directory and other necessary files like `openapi.yaml` if needed by the app at runtime) into the `/app` directory.
    - Expose Port: Expose the port the FastAPI application will run on (e.g., 8000).
    - Run Command: Set the default command to start the application using Uvicorn (e.g., `CMD ["uvicorn", "insanely_fast_whisper_api.main:app", "--host", "0.0.0.0", "--port", "8000"]`, assuming the main FastAPI app instance is `app` in `insanely_fast_whisper_api/main.py`).

- [x] **Create/Update `docker-compose.yaml`**
  - Path: `/home/elvee/Local-AI/insanely-fast-whisper-api/docker-compose.yaml`
  - Action: Define the Docker Compose services to build and run the application.
    - Version: Use `version: '3.8'` or a compatible newer version.
    - Service Definition (`api`):
      - `build: .` (to use the `Dockerfile` in the current project directory).
      - `ports`: Map host port to container port (e.g., `8000:8000`).
      - `volumes`: (Optional, for development convenience) Consider mapping `./insanely_fast_whisper_api:/app/insanely_fast_whisper_api` for live code reloading. Consider `./models:/app/models` if models are stored/downloaded.
      - `environment`: Define any necessary runtime environment variables (e.g., from `.env.example`).
      - **AMD GPU (ROCm) Support**:
        - Ensure the container has access to AMD GPU resources using the following specific configuration:

          ```yaml
          # To allow AMD GPU access
          devices:
          - "/dev/kfd:/dev/kfd"
          - "/dev/dri:/dev/dri"
          stdin_open: true
          tty: true
          cap_add:
          - SYS_PTRACE
          security_opt:
          - seccomp=unconfined
          group_add:
          - video
          ipc: host
          shm_size: 8G
          ```

- [x] **Build and Test Dockerized API**
  - [x] Build the Docker container using Docker Compose
    - Command: `docker-compose up --build -d api`
    - Action: Container built and started successfully.
  - [x] Verify `/openapi.json` reflects the correct API structure after changes per Dockerfile and docker-compose.yaml.
  - [x] Run API test script
    - Path: `/home/elvee/Local-AI/insanely-fast-whisper-api/tests/test_api.sh`
    - Action: Execute the test script. The script previously connected to the API on port 8001, but most endpoints returned 'Internal Server Error (500)'. Need to check Docker container logs to diagnose the root cause. Test 6 (unsupported file format) failed due to a client-side curl error (file not found or unreadable). Ensure the test script targets the correct port (8000 as per Dockerfile and docker-compose.yaml).

- [x] **Update `project-overview.md`**

  - Path: `/home/elvee/Local-AI/insanely-fast-whisper-api/project-overview.md`
  - Action: Add a new section detailing how to build and run the application using the Docker and Docker Compose setup. Include commands and any prerequisites (like Docker and Docker Compose installation, AMD GPU drivers on the host).
