#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# entrypoint-test-rocm-diagnostics.sh — Collect ROCm/GPU diagnostics
#
# Writes rocm-container-diagnostics.txt to the test-results directory
# (mounted at /tmp/test-results/ inside the container).
#
# Called by entrypoint-test.sh or can be run standalone.
# ---------------------------------------------------------------------------

set -euo pipefail

DIAGNOSTICS_OUTPUT="/tmp/test-results/rocm-container-diagnostics.txt"

mkdir -p "$(dirname "$DIAGNOSTICS_OUTPUT")"

{
    echo "=== ROCm Container Diagnostics ==="
    echo "Timestamp: $(date --iso-8601=seconds)"
    echo "Container route: ${TEST_CONTAINER_ROUTE:-unknown}"
    echo ""

    echo "--- OS Info ---"
    cat /etc/os-release 2>/dev/null || echo "(not available)"
    echo ""

    echo "--- Python Version ---"
    python3 --version 2>/dev/null || echo "(python3 not found)"
    echo ""

    echo "--- ROCm Version ---"
    if [ -f /opt/rocm/.info/version ]; then
        cat /opt/rocm/.info/version
    elif command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showproductname 2>/dev/null | head -20
    else
        echo "(ROCm not found at /opt/rocm)"
    fi
    echo ""

    echo "--- rocm-smi ---"
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi 2>/dev/null || echo "(rocm-smi failed)"
    else
        echo "(rocm-smi not available)"
    fi
    echo ""

    echo "--- PyTorch / CUDA (ROCm) Info ---"
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
    print(f'HIP version: {torch.version.hip}')
    print(f'CUDA version (runtime): {torch.version.cuda}')
" 2>/dev/null || echo "(PyTorch CUDA check failed)"
    echo ""

    echo "--- MIOpen Info ---"
    python3 -c "
try:
    import miopen
    print(f'MIOpen version: {miopen.__version__}')
except ImportError:
    print('(MIOpen not importable)')
except Exception as e:
    print(f'MIOpen error: {e}')
" 2>/dev/null || echo "(MIOpen check failed)"
    echo ""

    echo "--- Key Python Packages ---"
    pip list 2>/dev/null | grep -iE '^(torch|torchaudio|transformers|gradio|fastapi|stable-ts|demucs|pyannote|soundfile|numpy|onnxruntime)' || echo "(pip list failed)"
    echo ""

    echo "--- Environment Variables ---"
    echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-not set}"
    echo "MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-not set}"
    echo "ROCM_PATH=${ROCM_PATH:-not set}"
    echo "PYTORCH_TRITON_ROCM=${PYTORCH_TRITON_ROCM:-not set}"
    echo ""

    echo "--- /dev/kfd and /dev/dri ---"
    ls -la /dev/kfd 2>/dev/null || echo "(/dev/kfd not found)"
    ls -la /dev/dri/ 2>/dev/null || echo "(/dev/dri not found)"
    echo ""

    echo "=== End Diagnostics ==="
} > "$DIAGNOSTICS_OUTPUT"

echo "[+] Diagnostics written to ${DIAGNOSTICS_OUTPUT}"
