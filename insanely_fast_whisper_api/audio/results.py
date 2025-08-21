"""Utilities for processing and merging transcription results."""


def merge_chunk_results(chunk_results: list[dict]) -> dict:
    """Merge results from multiple chunked transcriptions.

    Args:
        chunk_results: List of transcription results from individual chunks.

    Returns:
        dict: Combined transcription result.
    """
    if not chunk_results:
        return {"text": "", "chunks": [], "runtime_seconds": 0.0}

    combined = {
        "text": "\n\n".join(
            r.get("text", "").strip() for r in chunk_results if r.get("text")
        ),
        "chunks": [chunk for r in chunk_results for chunk in r.get("chunks", [])],
        "runtime_seconds": round(
            sum(r.get("runtime_seconds", 0.0) for r in chunk_results), 2
        ),
    }
    # Preserve config_used from the first chunk if available, and add chunking info
    # Ensure config_used is always present
    first_chunk_config = chunk_results[0].get("config_used", {})
    combined["config_used"] = {
        **first_chunk_config,
        "chunking_used": True,
        "num_chunks": len(chunk_results),
    }

    return combined
