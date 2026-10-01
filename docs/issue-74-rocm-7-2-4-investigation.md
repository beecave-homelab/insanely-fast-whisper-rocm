# Issue #74: ROCm 7.2.4 image investigation

Status: **experimental; do not change the production image yet**. Measurements below
were taken on 2026-09-29. The restricted development shell does not expose
`/dev/kfd` or `/dev/dri`, but the Docker daemon does. Containers started with
the Compose GPU device settings can use the host's AMD Radeon RX 6600.

## Candidate and package boundary

AMD [validates the Python 3.10 / Ubuntu 22.04 PyTorch 2.9.1 image](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.4/install/3rd-party/pytorch-install.html)
for ROCm 7.2.4. The experimental Dockerfile pins the locally pulled image by digest
and retains its preinstalled framework stack:

| Component | Installed version |
| --- | --- |
| Torch distribution | `2.9.1+rocm7.2.4.lw.git39497456` |
| `torch.__version__` | `2.9.1+rocm7.2.4.git39497456` |
| `torch.version.hip` | `7.2.53211-97f5574fe2` |
| Torchaudio | `2.9.0+rocm7.2.4.gite3c6ee2b` |
| Torchvision | `0.24.0+rocm7.2.4.gitb919bd0c` |
| Triton | `3.5.1+rocm7.2.4.gita272dfa8` |

These versions came from `importlib.metadata` and `torch` inside the pulled image.
The `lw` suffix is the installed Torch distribution's local version. Its native
dependencies are supplied by the matching ROCm base: `ldd` on `libtorch_hip.so`
resolved MIOpen, HIP runtime/RTC, rocBLAS, RCCL, rocFFT, rocRAND, rocSPARSE,
and related libraries from `/opt/rocm/lib`. AMD recommends its prebuilt image as
an integrated, tested PyTorch/ROCm environment. This inspection establishes that
the lightweight package works with this base at import time; it does **not**
establish that overlaying arbitrary `.lw` wheels on a different base is supported.

The production `requirements-container.txt` export pins ROCm 7.0 because the
production Dockerfile still uses the ROCm 7.0 base. The experiment instead
uses `requirements-container-rocm-7-2-4.txt`, generated from
`pdm.rocm-7-2-4.lock`. It pins Torch 2.9.1, Torchaudio 2.9.0, Torchvision
0.24.0, and Triton 3.5.1 to the exact builds published in
[AMD's ROCm 7.2.4 wheel index](https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/).
The candidate base already contains those identical wheels, so pip leaves
them installed. The new lock reuses the existing runtime dependency versions
where possible; it omits the old `pytorch-triton-rocm` package, which is not
in AMD's 7.2.4 index, and the unused `onnxruntime-rocm` extra. Production and
development exports should be aligned together before changing their bases.

## Reproduce the packaging check

```bash
pdm lock -L pdm.rocm-7-2-4.lock -G rocm-7-2-4 -G diarization \
  --prod --python '==3.10.*' --platform manylinux_2_17_x86_64 \
  --strategy inherit_metadata --update-reuse
pdm export -L pdm.rocm-7-2-4.lock -G rocm-7-2-4 -G diarization \
  --prod --without-hashes -o requirements-container-rocm-7-2-4.txt
docker build --progress=plain \
  -f Dockerfile.rocm-7-2-4-experiment \
  -t insanely-fast-whisper-rocm:issue-74-rocm-724-pinned .
docker run --rm --pull=never insanely-fast-whisper-rocm:issue-74-rocm-724-pinned \
  python -c 'import torch, torchaudio, torchvision, triton, transformers, pyannote.audio, stable_whisper; import insanely_fast_whisper_rocm.core.asr_backend; print(torch.__version__, torch.version.hip)'
docker run --rm --pull=never insanely-fast-whisper-rocm:issue-74-rocm-724-pinned pip check
docker image inspect insanely-fast-whisper-rocm:issue-74-rocm-724-pinned \
  --format '{{.Size}}'
```

Build and import checks passed. With `/dev/kfd`, `/dev/dri`, and
`HSA_OVERRIDE_GFX_VERSION=10.3.0`, the candidate container detected the RX 6600
and successfully computed a GPU tensor sum. The ROCm 7.0 image passed the same
check. `pip check` reports that `pyannote-audio 4.0.4`
requires `torchcodec`, which is intentionally excluded by the existing project
dependency setup. The same error occurs in the existing ROCm 7.0 development
image. Pyannote import succeeds with a warning that built-in audio decoding
will fail. A SoundFile preload fallback was added and validated below.
The revised build using the explicit 7.2.4 export passed the framework version
assertion and import check. Its GPU CLI transcription produced the same text and
all 168 timestamps as the first candidate build. The pinned image's API also
returned HTTP 200 with a JSON payload identical to the earlier candidate;
the empty `segments` behavior remains tracked in issue #77. Refreshing the
existing `pdm.lock` after adding the new group left all 172 package versions
unchanged; its diff includes newly published artifact hashes.

## CLI and API transcription check

The candidate transcribed the repository's one-minute speech sample,
`temp_uploads/1-minute-test-audio.wav`, on the RX 6600 using cached
`openai/whisper-tiny`, float16, word timestamps, and stable-ts stabilization.
The CLI reported 168 stabilized word segments and completed successfully.
The candidate API returned HTTP 200, 949 characters of transcription text,
and `stabilized: true` in 13.82 seconds. Its log also reported 168 stabilized
segments. The same request to the existing ROCm 7.0 API returned HTTP 200 in
17.19 seconds; the complete JSON response bodies were identical. These are
single warm-cache requests, not a controlled performance benchmark.

Both API responses contain an empty `segments` array despite the processing
log reporting 168 segments. This is an existing API formatter bug, not a new
7.2.4 regression. `stable_ts.stabilize_timestamps` returns refined `segments`
and removes `chunks` when the refined timestamps are usable. The CLI exporter
prefers `segments`, but the API `verbose_json` formatter reads only `chunks`.
The same formatter mistake exists for translation. Feeding the real result
with 168 `segments` and no `chunks` directly to both API formatters reproduced
an empty response array in each case. The API tests mainly provide `chunks`,
and one response-format test allows an empty `segments` list, so they miss the
stabilized-result case. This is tracked in
[API issue #77](https://github.com/beecave-homelab/insanely-fast-whisper-rocm/issues/77)
with regression coverage requested for both endpoints. The 10-second
`tests/audio/fixtures/test_clip.mp3`
also completed through the candidate CLI, but yielded zero segments, so it
was not used as the speech validation sample.

### Direct comparison with the current development image

On September 29, 2026, both images transcribed the same one-minute WAV on the
RX 6600 with `openai/whisper-tiny`, `cuda:0`, float16, batch size 4, 30-second
chunks, word timestamps, stabilization enabled, and Demucs, VAD, and
diarization disabled. CLI used JSON export; API used `verbose_json`.

| Entry point | Current ROCm 7.0 dev image | ROCm 7.2.4 candidate |
| --- | --- | --- |
| CLI | 949 text characters; 168 stabilized segments | Identical text and all 168 segments, including timestamps |
| API | HTTP 200; 949 text characters; `stabilized=true`; 0 response segments | Identical complete JSON response |
| CLI-reported total time | 15.10 s | 14.69 s |
| API HTTP request time | 12.98 s | 13.80 s |

The text was identical across both images and both entry points. CLI JSON
matched on every field except run-specific timing and creation timestamp
metadata. These are single sequential runs with a shared warm model cache,
so the timings do not establish a performance difference. The API's empty
`segments` array occurs in both images even though each CLI export contains
168 segments.

## GPU diarization check

Before the SoundFile fallback, pyannote loaded on the GPU but audio preloading
failed: the candidate's Torchaudio 2.9 build requires TorchCodec even for WAV
input. The existing FFmpeg fallback converted to WAV and then called
Torchaudio again, reproducing the same error. The current `main` and `dev`
branches have that Torchaudio-only loader, but their pinned Torchaudio 2.8
successfully loaded this WAV in the existing development image. In the
candidate image, `soundfile.read` decoded the same WAV successfully. The
unmerged change on this investigation branch tries SoundFile when Torchaudio
fails and after FFmpeg conversion if needed. Regression tests cover a real WAV
and an FFmpeg-converted WAV with `torchaudio.load` raising the TorchCodec error.

With the fix, GPU diarization completed for the one-minute sample. It found
13 speaker turns and assigned 168 word chunks across three speaker labels.
With `MIOPEN_FIND_MODE=2`, inference took 12.76 seconds. With the variable
unset, inference also completed but took 62.93 seconds. These are cold,
single runs; the slower unset run may include JIT compilation. Keep mode 2
until repeated warm runs show a clear reason to change it. Demucs and VAD
were disabled for these isolated diarization checks.

## Repeated loading and tests

A temporary candidate API with `IFW_EAGER_MODEL_RELEASE=1` processed two
sequential one-minute speech requests. Both returned HTTP 200 and identical
949-character transcriptions. Logs show the ASR model loading on `cuda:0`
for each request, consistent with eager release and reload. The requests took
13.06 and 5.49 seconds, respectively; they are single observations and do
not establish a performance gain. A forced hardware OOM recovery was not
attempted on the host while other GPU services were running.

`pdm run pytest tests/core/integrations/test_diarization.py
tests/core/test_oom_utils.py tests/core/test_backend_cache.py
tests/core/test_backend_cache_timeout.py -q --maxfail=1` passed: 90 tests.
Ruff lint and format checks passed for the changed Python files. The full
`pdm run pytest --maxfail=1 -q` run stalled at its first API TestClient test,
`tests/api/test_api.py::test_transcription_with_stabilization_options`, and
was interrupted. The same stall occurred when that test ran alone. The full
suite still needs a separate diagnosis and completed run before promotion.
On 2026-10-01, the same full-suite command produced no progress and reached a
120-second timeout. The targeted 90-test run, Ruff checks, and both PDM lock
checks passed again.

## Measurements and limits

| Image | Reported size | Framework / HIP |
| --- | ---: | --- |
| Pulled ROCm 7.2.4 base | 39.50 GB | Torch 2.9.1 / HIP 7.2 |
| Experimental app, initial filtered export | 42.85 GB | Torch 2.9.1 / HIP 7.2 |
| Experimental app, pinned 7.2.4 export | 40.63 GB | Torch 2.9.1 / HIP 7.2 |
| Existing development app | 59.55 GB | Torch 2.8.0 / HIP 7.0 |

The revised experimental image adds 1.13 GB over its base and is 18.92 GB smaller
than the existing development image. The existing image uses another base and
is not a controlled comparison of framework wheels alone. The experimental
build took roughly three minutes, including package download and image export;
there is no comparable clean-build timing for the existing image. Startup,
first/warm transcription latency, and VRAM still need controlled measurements.
The build reduced available root filesystem space from 32 GB to 26 GB; no
Docker cache, image, or volume cleanup was performed.

## GPU-host acceptance work

1. Exercise the OOM cleanup path on a controlled GPU test host. Record first
   and warm latency, startup time, and peak VRAM for current and candidate
   images under identical settings. Diagnose the stalled API TestClient test
   and run the full suite to completion.
2. Test the ROCm 7.2.4 / PyTorch 2.8.0 validated image as a controlled
   migration baseline if disk space permits. AMD lists it alongside 2.9.1 in
   the [validated image inventory](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.4/install/3rd-party/pytorch-install.html).
3. If the remaining GPU checks pass, regenerate runtime and development dependency exports,
   update production and development bases together, and repeat the comparison.

**Recommendation:** keep ROCm 7.0 as the production default for now. The
7.2.4 / 2.9.1 image is a promising candidate because its bundled lightweight
framework imports with the app, produces a smaller image locally, and passes
GPU ASR and stabilization through both CLI and API. GPU diarization completed
with and without the MIOpen workaround after the SoundFile preload fix, and
two eager-release API requests completed with a model reload. Hardware OOM
recovery and controlled performance remain unverified, so the issue's
promotion criteria are not yet met.

Security updates to other pinned Python dependencies are tracked separately in
[web/API/media issue #79](https://github.com/beecave-homelab/insanely-fast-whisper-rocm/issues/79),
[model-loading issue #80](https://github.com/beecave-homelab/insanely-fast-whisper-rocm/issues/80),
and [remaining-dependencies issue #81](https://github.com/beecave-homelab/insanely-fast-whisper-rocm/issues/81).
Rebase this experiment and regenerate its lock/export after those changes merge.
