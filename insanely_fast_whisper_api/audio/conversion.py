"""Audio format conversion utilities.

Currently used to convert unsupported input formats (e.g. `.m4a`) into a
Whisper-compatible `.wav` file before passing it to the ASR backend.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import ffmpeg

# Default parameters aligned with Whisper feature extractor defaults
DEFAULT_SAMPLE_RATE = 16000  # Hz
DEFAULT_CHANNELS = 1  # mono
DEFAULT_CODEC = "pcm_s16le"  # 16-bit PCM WAV


def ensure_wav(
    input_path: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> str:
    """Return a path to a WAV file for *input_path*.

    If *input_path* already ends with ".wav" (case-insensitive), it is returned
    unchanged. Otherwise the file is transcoded with FFmpeg to an uncompressed
    16-bit PCM WAV with the desired *sample_rate* and *channels* and a path to
    the temporary WAV file is returned.
    """

    input_path = str(input_path)

    if input_path.lower().endswith(".wav"):
        return input_path

    tmp_dir = tempfile.mkdtemp(prefix="converted_audio_")
    output_path = os.path.join(
        tmp_dir,
        f"{Path(input_path).stem}.wav",
    )

    (
        ffmpeg.input(input_path)
        .output(
            output_path,
            acodec=DEFAULT_CODEC,
            ac=channels,
            ar=sample_rate,
        )
        .overwrite_output()
        .run(quiet=True)
    )

    return output_path
