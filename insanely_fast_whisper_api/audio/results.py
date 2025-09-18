"""Utilities for processing and merging transcription results."""


from typing import Any


def merge_chunk_results(chunk_results: list[tuple[dict, float]]) -> dict[str, Any]:
    """Merge results from multiple chunked transcriptions, adjusting timestamps.

    Args:
        chunk_results: A list of tuples, where each tuple contains the
            transcription result from an individual chunk and the chunk's start
            time in seconds.

    Returns:
        dict: Combined transcription result with adjusted timestamps.
    """
    if not chunk_results:
        return {"text": "", "chunks": [], "runtime_seconds": 0.0}

    all_chunks = []
    full_text = []
    total_runtime = 0.0

    for result, start_time in chunk_results:
        full_text.append(result.get("text", "").strip())
        total_runtime += result.get("runtime_seconds", 0.0)

        if "chunks" in result:
            for segment in result["chunks"]:
                # Adjust segment timestamps
                if "timestamp" in segment and isinstance(
                    segment["timestamp"], (list, tuple)
                ):
                    segment["timestamp"] = (
                        segment["timestamp"][0] + start_time,
                        segment["timestamp"][1] + start_time,
                    )

                # Adjust word timestamps if they exist
                if "words" in segment and isinstance(segment["words"], list):
                    for word in segment["words"]:
                        if "start" in word and "end" in word:
                            word["start"] += start_time
                            word["end"] += start_time
                all_chunks.append(segment)

    combined = {
        "text": "\n\n".join(filter(None, full_text)),
        "chunks": all_chunks,
        "runtime_seconds": round(total_runtime, 2),
    }

    # Preserve config_used from the first chunk and add chunking info
    first_chunk_config = chunk_results[0][0].get("config_used", {})
    combined["config_used"] = {
        **first_chunk_config,
        "chunking_used": True,
        "num_chunks": len(chunk_results),
    }

    return combined
