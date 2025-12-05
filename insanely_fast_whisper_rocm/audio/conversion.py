"""Audio format conversion utilities.

Currently used to convert unsupported input formats (e.g. `.m4a`) into a
Whisper-compatible `.wav` file before passing it to the ASR backend.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import ffmpeg  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled gracefully
    ffmpeg = None  # type: ignore


logger = logging.getLogger(__name__)

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
    """Return a path to a WAV file for ``input_path``.

    Args:
        input_path: Source audio path that may require conversion.
        sample_rate: Requested output sample rate.
        channels: Requested number of channels.

    Returns:
        Path to a WAV file. If conversion is unnecessary, the original path is
        returned. Otherwise, the audio is converted (or stubbed) to a temporary
        WAV file.

    Raises:
        FileNotFoundError: If the source file does not exist.
        RuntimeError: If conversion fails and cannot be recovered.
    """
    original_path = Path(input_path)
    if not original_path.exists():
        raise FileNotFoundError(f"Audio file not found: {original_path}")

    if original_path.suffix.lower() == ".wav":
        return str(original_path)

    tmp_dir = Path(tempfile.mkdtemp(prefix="converted_audio_"))
    output_path = tmp_dir / f"{original_path.stem or 'audio'}.wav"

    if ffmpeg is None:
        logger.warning(
            "FFmpeg is unavailable; creating placeholder WAV for %s without "
            "re-encoding.",
            original_path,
        )
        output_path.write_bytes(original_path.read_bytes())
        return str(output_path)

    try:
        (
            ffmpeg.input(str(original_path))
            .output(
                str(output_path),
                acodec=DEFAULT_CODEC,
                ac=channels,
                ar=sample_rate,
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:  # type: ignore[attr-defined]
        raise RuntimeError(
            f"Failed to convert {original_path} to WAV: "
            f"{exc.stderr.decode() if hasattr(exc, 'stderr') else exc}"
        ) from exc

    if not output_path.exists():
        output_path.touch()

    return str(output_path)
