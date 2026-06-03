"""Run WebUI transcription integration checks through Gradio's client API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gradio_client import Client, handle_file

WEBUI_API_NAMES = ("/transcribe_audio_v2", "transcribe_audio_v2")


class WebUITestError(Exception):
    """Raised when the WebUI integration check fails."""


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run diarized transcription checks through the Gradio WebUI."
    )
    parser.add_argument("--base-url", required=True, help="Base URL of the WebUI.")
    parser.add_argument("--audio-file", required=True, help="Audio file to transcribe.")
    parser.add_argument("--model", required=True, help="Whisper model name.")
    parser.add_argument("--timestamp-type", choices=["chunk", "word"], required=True)
    parser.add_argument("--output-file", required=True, help="Path for result JSON.")
    parser.add_argument("--device", default="cuda", help="Whisper device.")
    parser.add_argument("--diarization-device", default="cuda")
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--language", default="")
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--chunk-length", type=int, default=30)
    parser.add_argument("--vad-threshold", type=float, default=0.35)
    parser.add_argument("--save-dir", default="temp_uploads")
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def _client_endpoint_names(client: Client) -> set[str]:
    """Return endpoint names advertised by the running Gradio app."""
    try:
        api_info = client.view_api(return_format="dict")
    except TypeError:
        api_info = getattr(client, "config", {})

    if isinstance(api_info, dict):
        named_endpoints = api_info.get("named_endpoints", {})
        if isinstance(named_endpoints, dict):
            return set(named_endpoints)

    endpoints = getattr(client, "endpoints", [])
    return {
        str(getattr(endpoint, "api_name", ""))
        for endpoint in endpoints
        if getattr(endpoint, "api_name", "")
    }


def _pick_api_name(client: Client) -> str:
    """Select the supported Gradio endpoint name for transcription.

    Args:
        client: Connected Gradio client.

    Returns:
        API name accepted by the running WebUI.

    Raises:
        WebUITestError: If the WebUI does not advertise the transcription endpoint.
    """
    endpoint_names = _client_endpoint_names(client)
    for api_name in WEBUI_API_NAMES:
        if api_name in endpoint_names:
            return api_name
    expected = ", ".join(WEBUI_API_NAMES)
    available = ", ".join(sorted(endpoint_names)) or "<none>"
    raise WebUITestError(
        f"WebUI transcription endpoint not found. Expected one of: {expected}. "
        f"Available endpoints: {available}"
    )


def _extract_json_preview(result: object) -> dict[str, object]:
    """Extract the JSON preview object from a Gradio result tuple.

    Args:
        result: Final result returned by the Gradio client.

    Returns:
        Parsed transcription JSON object.

    Raises:
        WebUITestError: If the result does not contain transcription JSON.
    """
    if not isinstance(result, tuple | list):
        raise WebUITestError(f"Unexpected WebUI result type: {type(result).__name__}")

    for item in result:
        if isinstance(item, dict) and "text" in item:
            return item
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "text" in parsed:
                return parsed

    raise WebUITestError("Could not find transcription JSON in WebUI result.")


def _extract_transcript(result: object) -> str:
    """Extract human-readable transcript text from a Gradio result tuple.

    Args:
        result: Final result returned by the Gradio client.

    Returns:
        Transcript text, or an empty string when no transcript field is present.
    """
    if not isinstance(result, tuple | list):
        return ""

    for item in result:
        if isinstance(item, str) and item.strip() and not item.lstrip().startswith("{"):
            if "transcription" in item.lower() and "complete" in item.lower():
                continue
            return item.strip()
    return ""


def _has_speaker_labels(data: dict[str, object]) -> bool:
    """Return whether diarization produced speaker labels in result segments."""
    for key in ("chunks", "segments"):
        entries = data.get(key, [])
        if isinstance(entries, list) and any(
            isinstance(entry, dict) and entry.get("speaker") for entry in entries
        ):
            return True
    return False


def _validate_result(data: dict[str, object], transcript: str) -> None:
    """Validate transcription and diarization output from the WebUI.

    Args:
        data: Parsed transcription JSON object.
        transcript: Transcript text returned by the UI.

    Raises:
        WebUITestError: If transcription or diarization output is incomplete.
    """
    text = str(data.get("text", "")).strip()
    if not text and not transcript:
        raise WebUITestError("WebUI returned an empty transcription.")

    if data.get("diarized") is not True:
        raise WebUITestError(
            "WebUI transcription completed, but diarized=true is missing."
        )

    if not _has_speaker_labels(data):
        raise WebUITestError(
            "WebUI transcription completed, but no speaker labels were returned."
        )


def _run_prediction(client: Client, args: argparse.Namespace, api_name: str) -> object:
    """Submit a transcription job to the WebUI and return its final result.

    Args:
        client: Connected Gradio client.
        args: Parsed command-line arguments.
        api_name: Named Gradio API endpoint to call.

    Returns:
        Final Gradio result for the submitted transcription job.
    """
    job = client.submit(
        [handle_file(args.audio_file)],
        args.model,
        args.device,
        args.batch_size,
        args.timestamp_type,
        args.language,
        args.task,
        args.dtype,
        args.chunk_length,
        True,
        True,
        True,
        args.vad_threshold,
        True,
        args.num_speakers,
        args.min_speakers,
        args.max_speakers,
        args.diarization_device,
        True,
        args.save_dir,
        api_name=api_name,
    )
    return job.result(timeout=args.timeout)


def main() -> int:
    """Run the WebUI integration test.

    Returns:
        Process exit code.

    Raises:
        WebUITestError: If the audio input or WebUI result is invalid.
    """
    args = _parse_args()
    audio_path = Path(args.audio_file)
    if not audio_path.is_file():
        raise WebUITestError(f"Audio file not found: {audio_path}")

    client = Client(args.base_url)
    api_name = _pick_api_name(client)
    result = _run_prediction(client, args, api_name)
    data = _extract_json_preview(result)
    transcript = _extract_transcript(result)
    _validate_result(data, transcript)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "api_name": api_name,
                "timestamp_type": args.timestamp_type,
                "transcript": transcript or data.get("text", ""),
                "data": data,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        "WebUI transcription passed "
        f"(timestamp_type={args.timestamp_type}, api_name={api_name})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WebUITestError as exc:
        print(f"WebUI integration test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
