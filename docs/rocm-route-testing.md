# ROCm Route Testing

Integration tests that verify transcription works end-to-end (CLI, API, WebUI) across different ROCm Docker configurations.

> **Branch**: `test/rocm-routes` contains all testing files. Switch to this branch to run the tests.

## Overview

The test suite validates that the application can transcribe audio with all feature flags enabled (`--vad --demucs --stabilize --diarize`) across two ROCm delivery routes:

| Route | Base image | ROCm source | Image size |
| -- | -- | -- | -- |
| **rocm-pytorch** | `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0` | Pre-installed in base image | ~60 GB |
| **slim-rocm-apt** | `python:3.10-slim` | Bundled in PyTorch ROCm wheels from AMD's manylinux repo | ~20 GB |

Both routes access the GPU via host device passthrough (`/dev/kfd`, `/dev/dri`). No ROCm apt packages are needed in the slim route — the PyTorch ROCm wheels bundle the necessary userspace libraries.

## Prerequisites

- AMD GPU with ROCm host drivers installed
- Docker with Compose v2
- At least 25 GB free disk space per route image
- `temp_uploads/1-minute-test-audio.wav` test audio file (mounted from host)
- HuggingFace cache at `~/.cache/huggingface/hub` (mounted for model reuse)

## File reference

### Dockerfiles

| File | Purpose |
| -- | -- |
| `Dockerfile.test` | Single-route test container (used by `docker-compose.test.yaml`) |
| `Dockerfile.test.rocm-pytorch` | ROCm PyTorch route — builds on `rocm/pytorch` base image |
| `Dockerfile.test.slim-rocm-apt` | Slim route — builds on `python:3.10-slim` with ROCm wheels |

### Docker Compose files

| File | Purpose |
| -- | -- |
| `docker-compose.test.yaml` | Single-route test (one service, default ports 8891/7864) |
| `docker-compose.test.rocm-routes.yaml` | Multi-route comparison (two services with separate ports) |

### Scripts

| File | Purpose |
| -- | -- |
| `scripts/entrypoint-test.sh` | Main test entrypoint — runs CLI, API, and WebUI tests sequentially |
| `scripts/entrypoint-test-rocm-diagnostics.sh` | Collects ROCm/GPU diagnostics (rocm-smi, PyTorch info, MIOpen, env vars) |
| `scripts/run-rocm-route-comparison.sh` | Orchestrator — builds, runs, and collects comparison artifacts |
| `scripts/webui_integration_test.py` | WebUI test client — submits transcription via Gradio's client API |

## Quick start

### Single route (simplest)

```bash
# Build and run the default test container
docker compose -f docker-compose.test.yaml build
docker compose -f docker-compose.test.yaml run --rm insanely-fast-whisper-rocm-test
```

### Multi-route comparison

```bash
# Run a single route
scripts/run-rocm-route-comparison.sh --service insanely-fast-whisper-rocm-test-rocm-pytorch

# Run the slim route
scripts/run-rocm-route-comparison.sh --service insanely-fast-whisper-rocm-test-slim-rocm-apt

# Run all routes (default)
scripts/run-rocm-route-comparison.sh

# Verbose output (streams build + run logs to console)
scripts/run-rocm-route-comparison.sh --verbose --service insanely-fast-whisper-rocm-test-rocm-pytorch

# Force rebuild even if image exists
scripts/run-rocm-route-comparison.sh --force-build --service insanely-fast-whisper-rocm-test-rocm-pytorch

# Dry run (prints planned actions without running Docker)
scripts/run-rocm-route-comparison.sh --dry-run

# Drop to interactive shell inside container after tests
scripts/run-rocm-route-comparison.sh --drop-to-shell --service insanely-fast-whisper-rocm-test-rocm-pytorch
```

### `run-rocm-route-comparison.sh` options

| Flag | Description |
| -- | -- |
| `-f, --compose-file FILE` | Override compose file (default: `docker-compose.test.rocm-routes.yaml`) |
| `-o, --output-dir DIR` | Override output directory (default: `test-results/rocm-route-comparison`) |
| `-s, --service SERVICE` | Run specific service; repeat for multiple |
| `--drop-to-shell` | Keep container open with interactive bash after tests |
| `--dry-run` | Print planned actions without running Docker |
| `--verbose` | Stream build and run output to console while saving logs |
| `--no-header` | Skip ASCII art header |
| `--force-build` | Rebuild images even if they already exist |
| `-h, --help` | Show help message |

### Environment variables

| Variable | Default | Description |
| -- | -- | -- |
| `TEST_ROCM_PYTORCH_API_PORT` | 8891 | API port for rocm-pytorch route |
| `TEST_ROCM_PYTORCH_WEBUI_PORT` | 7864 | WebUI port for rocm-pytorch route |
| `TEST_SLIM_ROCM_APT_API_PORT` | 8892 | API port for slim-rocm-apt route |
| `TEST_SLIM_ROCM_APT_WEBUI_PORT` | 7865 | WebUI port for slim-rocm-apt route |
| `MIOPEN_FIND_MODE` | 2 | MIOpen find mode (2 = database-only, no JIT) |
| `ROCM_PYTORCH_BASE` | `rocm/pytorch:latest` | Override base image for rocm-pytorch route |
| `TEST_DROP_TO_SHELL` | 1 (in containers) | Set to `0` to exit immediately after tests |

## Test phases

`entrypoint-test.sh` runs four phases sequentially:

### Phase 0 — Prerequisites

Verifies the test audio file exists at `temp_uploads/1-minute-test-audio.wav`. If missing, drops to bash for manual inspection.

### Phase 1 — CLI transcription tests

Runs `insanely-fast-whisper-cli transcribe` with `--vad --demucs --stabilize --diarize` for both `chunk` and `word` timestamp types. Results are validated with `validate_json_diarized` which checks for non-empty text, `diarized=true`, and speaker labels in segments.

### Phase 2 — API transcription tests

Starts the API server (`python -m insanely_fast_whisper_rocm`), waits for it to be ready, then sends `POST /v1/audio/transcriptions` requests with `diarize=true`, `diarization_device=cuda`, and `num_speakers=2`. Validates responses with `validate_json_diarized`.

### Phase 3 — WebUI transcription tests

Starts the WebUI server (`python -m insanely_fast_whisper_rocm.webui --diarize`), waits for Gradio to be ready, then runs `scripts/webui_integration_test.py` which submits transcription jobs via the Gradio client API. The test script validates the response contains diarized output with speaker labels.

### Phase 4 — Summary

Prints a results table and drops to an interactive bash shell (unless `TEST_DROP_TO_SHELL=0`).

## Test validation

Two validators are used:

- **`validate_json_text`** — checks for a non-empty top-level `text` field in the JSON output
- **`validate_json_diarized`** — checks for non-empty text, `diarized=true`, and at least one segment with a `speaker` label

All tests use `validate_json_diarized` since diarization is enabled by default.

## Output artifacts

Each route produces artifacts under `test-results/rocm-route-comparison/<service-name>/`:

| File | Description |
| -- | -- |
| `build.log` | Docker build output |
| `run.log` | Container run output (test results) |
| `image.txt` | Docker image info |
| `summary.txt` | Exit codes and artifact paths |
| `rocm-container-diagnostics.txt` | GPU/ROCm diagnostics (if available) |

Per-route test results are also written to `test-results/<route-name>/` inside the container (mounted from host).

## ROCm diagnostics

`entrypoint-test-rocm-diagnostics.sh` collects:

- OS info and Python version
- ROCm version (from `/opt/rocm/.info/version` or `rocm-smi`)
- `rocm-smi` output
- PyTorch CUDA/HIP availability and device info
- MIOpen version
- Key Python package versions (torch, torchaudio, transformers, gradio, etc.)
- Environment variables (`HSA_OVERRIDE_GFX_VERSION`, `MIOPEN_FIND_MODE`, `ROCM_PATH`)
- Device file listing (`/dev/kfd`, `/dev/dri/`)

## GPU access

Both routes use the same device passthrough configuration:

```yaml
devices:
  - "/dev/kfd:/dev/kfd"
  - "/dev/dri:/dev/dri"
group_add:
  - video
ipc: host
shm_size: 8G
```

The `HSA_OVERRIDE_GFX_VERSION=10.3.0` environment variable is set for RDNA2 GPUs (e.g., RX 6600). Adjust this for other GPU architectures.

## Troubleshooting

### Build fails with "no space left on device"

Prune Docker build cache and unused images:

```bash
docker builder prune -f
docker image prune -a -f
```

### WebUI tests fail with "No such file or directory: webui_integration_test.py"

The compose file must mount `scripts/webui_integration_test.py` into the container. Verify the volume mount exists in your compose file.

### WebUI tests fail with "empty or missing 'text' field"

The `webui_integration_test.py` script writes a JSON file with a top-level `text` field. If this field is missing, ensure the script is up to date.

### Permission denied on entrypoint-test.sh

The host file must be executable:

```bash
chmod +x scripts/entrypoint-test.sh scripts/entrypoint-test-rocm-diagnostics.sh
```

### MIOpen errors during diarization

If pyannote LSTM inference fails with `miopenStatusUnknownError`, the diarization module will automatically retry on CPU. Set `MIOPEN_FIND_MODE=2` to skip JIT kernel compilation.

### Container appears stuck after tests

By default, `TEST_DROP_TO_SHELL=1` keeps the container running with an interactive bash prompt after tests complete. Set `TEST_DROP_TO_SHELL=0` or use `--drop-to-shell` flag on the orchestrator script to control this behavior.
