"""Utilities to compute a simple SRT formatting quality score.

The public contract:

    compute_srt_quality(segments, srt_text) -> {
        "score": float in [0, 1],
        "details": {...}
    }

Notes:
    The metric is intentionally lightweight and dependency-free so it can be
    easily used without external dependencies. It focuses on a few intuitive
    checks (overlaps, hyphen spacing, line lengths, CPS bounds) that correlate
    well with readable subtitles.
"""

from __future__ import annotations

import statistics
from typing import Any

from insanely_fast_whisper_rocm.utils import constants


def compute_srt_quality(
    segments: list[dict[str, Any]],
    srt_text: str,
) -> dict[str, Any]:
    """Compute a simple quality score for SRT output.

    Args:
        segments: List of segment dictionaries with at least ``start``, ``end``,
            and ``text`` keys.
        srt_text: The rendered SRT file contents as a single string.

    Returns:
        A mapping containing a float ``score`` in [0, 1] and a ``details``
        dictionary with diagnostic sub-metrics. The details include overlap
        counts, hyphen spacing checks, line length statistics, CPS ratios,
        duration summaries, histograms, and representative sample offenders.
    """
    # --- Sub-metric 1: overlaps ---
    overlap_violations = 0
    prev_end = None
    for seg in segments or []:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
        except (TypeError, ValueError):
            # Defensive: ignore malformed entries
            continue
        if prev_end is not None and start < prev_end:  # overlap
            overlap_violations += 1
        prev_end = max(prev_end or end, end)

    # --- Sub-metric 2: hyphen spacing normalization issues (e.g., "co -pilot") ---
    # A simple heuristic that flags patterns with spaces around a single hyphen
    # where both sides are alphabetic tokens.
    hyphen_bad = _has_bad_hyphen_spacing(srt_text)

    # --- Sub-metric 3: line length violations ---
    lines = [ln for ln in (srt_text.splitlines() if srt_text else [])]
    text_lines = [
        ln
        for ln in lines
        if (
            ln
            and not ln[0].isdigit()
            and " --> " not in ln
            and not ln.strip().isdigit()
        )
    ]
    line_length_violations = sum(
        1 for ln in text_lines if len(ln) > constants.MAX_LINE_CHARS
    )
    total_text_lines = max(1, len(text_lines))
    line_length_violation_ratio = line_length_violations / total_text_lines

    # --- Sub-metric 4: CPS within range ratio ---
    cps_ok_count = 0
    cps_total = 0
    durations: list[float] = []
    boundary_counts = {
        "within_range": 0,
        "too_short": 0,
        "too_long": 0,
    }
    cps_histogram = {
        "below_min": 0,
        "within_range": 0,
        "above_max": 0,
        "total": 0,
    }
    cps_offenders: list[dict[str, Any]] = []
    for index, seg in enumerate(segments or []):
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = str(seg.get("text", ""))
            dur = max(1e-6, end - start)
            cps = len(text) / dur
            cps_total += 1
            cps_histogram["total"] += 1
            durations.append(dur)
            if dur < constants.MIN_SEGMENT_DURATION_SEC:
                boundary_counts["too_short"] += 1
            elif dur > constants.MAX_SEGMENT_DURATION_SEC:
                boundary_counts["too_long"] += 1
            else:
                boundary_counts["within_range"] += 1
            if constants.MIN_CPS <= cps <= constants.MAX_CPS:
                cps_ok_count += 1
                cps_histogram["within_range"] += 1
            elif cps < constants.MIN_CPS:
                cps_histogram["below_min"] += 1
                cps_offenders.append({
                    "segment_index": index,
                    "start": start,
                    "end": end,
                    "duration_seconds": float(dur),
                    "cps": float(cps),
                    "category": "below_min",
                    "text": text,
                })
            else:
                cps_histogram["above_max"] += 1
                cps_offenders.append({
                    "segment_index": index,
                    "start": start,
                    "end": end,
                    "duration_seconds": float(dur),
                    "cps": float(cps),
                    "category": "above_max",
                    "text": text,
                })
        except Exception:
            # Ignore malformed segments
            continue
    cps_within_range_ratio = (cps_ok_count / cps_total) if cps_total else 1.0

    duration_stats = _summarize_durations(durations)
    line_length_offenders = _collect_line_length_offenders(text_lines)
    sample_offenders = {
        "line_length": line_length_offenders,
        "cps": cps_offenders[:5],
    }

    # --- Compose score ---
    # Start from perfect score and subtract proportional penalties.
    score = 1.0
    # Overlaps are severe readability issues.
    if overlap_violations > 0:
        score -= 0.3
    # Bad hyphen spacing slightly penalizes.
    if hyphen_bad:
        score -= 0.2
    # Penalize proportionally to line length violations (up to 0.3)
    score -= min(0.3, line_length_violation_ratio * 0.3)
    # Penalize lack of CPS compliance (up to 0.2)
    score -= min(0.2, (1.0 - cps_within_range_ratio) * 0.2)
    # Penalize duration boundary violations (baseline 0.1 plus proportional up to 0.4)
    violating_segments = boundary_counts["too_short"] + boundary_counts["too_long"]
    if violating_segments > 0:
        total_segments = sum(boundary_counts.values()) or 1
        duration_penalty_ratio = violating_segments / total_segments
        score -= 0.1
        score -= min(0.4, duration_penalty_ratio * 0.4)

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    return {
        "score": float(score),
        "details": {
            "overlap_violations": int(overlap_violations),
            "hyphen_normalization_ok": not hyphen_bad,
            "line_length_violations": int(line_length_violations),
            "line_length_violation_ratio": float(line_length_violation_ratio),
            "cps_within_range_ratio": float(cps_within_range_ratio),
            "duration_stats": duration_stats,
            "cps_histogram": {key: int(value) for key, value in cps_histogram.items()},
            "boundary_counts": {
                key: int(value) for key, value in boundary_counts.items()
            },
            "sample_offenders": sample_offenders,
        },
    }


def _summarize_durations(durations: list[float]) -> dict[str, float]:
    """Summarize segment durations in seconds.

    Args:
        durations: Durations for valid segments.

    Returns:
        Summary statistics including min, max, average, and median durations.
    """
    if not durations:
        return {
            "min_seconds": 0.0,
            "max_seconds": 0.0,
            "average_seconds": 0.0,
            "median_seconds": 0.0,
        }
    return {
        "min_seconds": float(min(durations)),
        "max_seconds": float(max(durations)),
        "average_seconds": float(statistics.mean(durations)),
        "median_seconds": float(statistics.median(durations)),
    }


def _collect_line_length_offenders(text_lines: list[str]) -> list[dict[str, Any]]:
    """Return SRT text lines exceeding configured length limits.

    Args:
        text_lines: Rendered SRT text lines without timing or numbering.

    Returns:
        A list of sample offenders with line content and metadata.
    """
    offenders: list[dict[str, Any]] = []
    max_chars = constants.MAX_LINE_CHARS
    for index, line in enumerate(text_lines, start=1):
        if len(line) > max_chars:
            offenders.append({
                "line_index": index,
                "line": line,
                "length": len(line),
                "limit": max_chars,
            })
            if len(offenders) >= 5:
                break
    return offenders


def _has_bad_hyphen_spacing(srt_text: str) -> bool:
    """Return True if suspicious hyphen spacing appears in SRT text.

    Pattern examples flagged as bad: ``"co -pilot"`` or ``"end - to-end"``.
    The heuristic checks for single hyphen with spaces around it between letters
    while remaining dependency-free.

    Args:
        srt_text: Rendered SRT contents as a single string.

    Returns:
        True if a bad hyphen spacing pattern is found, otherwise False.
    """
    if not srt_text:
        return False
    tokens = srt_text.split()
    punctuation = '.,!?;:"'
    for i, token in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""

        prev_clean = prev.strip(punctuation)
        token_clean = token.strip(punctuation)
        next_clean = nxt.strip(punctuation)

        if token == "-" and prev_clean.isalpha() and next_clean.isalpha():
            return True

        if token_clean.startswith("-") and len(token_clean) > 1:
            trailing = token_clean[1:]
            if prev_clean.isalpha() and trailing.isalpha():
                return True

        if token_clean.endswith("-") and len(token_clean) > 1:
            leading = token_clean[:-1]
            if next_clean.isalpha() and leading.isalpha():
                return True
    return False
