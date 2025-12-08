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
        # Fallback to a pure-Python conversion path when FFmpeg is unavailable.
        try:  # pragma: no cover - optional dependency
            from pydub import AudioSegment  # type: ignore[import]
        except ModuleNotFoundError as exc:  # pragma: no cover - handled gracefully
            logger.exception(
                "FFmpeg is not installed and optional dependency 'pydub' is "
                "missing; cannot convert %s to WAV.",
                original_path,
            )
            raise RuntimeError(
                "Audio conversion requires either FFmpeg (via 'ffmpeg-python') "
                "or the optional 'pydub' package. Install one of them to "
                "enable non-WAV input support."
            ) from exc

        try:
            audio = AudioSegment.from_file(str(original_path))
            audio = audio.set_frame_rate(sample_rate).set_channels(channels)
            audio.export(str(output_path), format="wav")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "Failed to convert %s to WAV using pydub fallback.", original_path
            )
            raise RuntimeError(
                f"Failed to convert {original_path} to WAV using pydub fallback: {exc}"
            ) from exc

        if not output_path.exists():
            raise RuntimeError(
                f"WAV conversion for {original_path} did not produce an output file "
                f"at {output_path}."
            )

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
