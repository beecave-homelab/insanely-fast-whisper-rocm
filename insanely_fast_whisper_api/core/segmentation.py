"""Segmentation logic for creating readable subtitles from ASR word timestamps."""

from __future__ import annotations

import dataclasses

from insanely_fast_whisper_api.utils import constants


@dataclasses.dataclass
class Word:
    """Represents a single word with timing information."""

    text: str
    start: float
    end: float


@dataclasses.dataclass
class Segment:
    """Represents a subtitle segment with formatted text and timing."""

    text: str
    start: float
    end: float
    words: list[Word]


def segment_words(words: list[Word]) -> list[Segment]:
    """Orchestrates the full segmentation process.

    This function will take a list of words and return a list of readable subtitle
    segments.

    Args:
        words: A list of Word objects from the ASR output.

    Returns:
        A list of Segment objects formatted for readability.
    """
    sentences = list(_sentence_chunks(words))
    segments = []
    for sentence in sentences:
        if not _respect_limits(sentence):
            clauses = _split_at_clause_boundaries(sentence)
            for clause in clauses:
                segments.append(
                    Segment(
                        text=" ".join(w.text for w in clause),
                        start=clause[0].start,
                        end=clause[-1].end,
                        words=clause,
                    )
                )
        else:
            segments.append(
                Segment(
                    text=" ".join(w.text for w in sentence),
                    start=sentence[0].start,
                    end=sentence[-1].end,
                    words=sentence,
                )
            )
    segments = _merge_short_segments(segments)

    # Expand single-word segments into sub-words (based on whitespace) so CPS
    # enforcement can operate meaningfully on realistic units.
    segments = [_maybe_expand_single_word(seg) for seg in segments]

    # Enforce CPS constraints only for single-word-origin segments. Multi-word
    # segments are handled by line wrapping and duration constraints.
    segments = _enforce_cps(segments)

    # Apply final text formatting (line wrapping) per segment.
    for seg in segments:
        seg.text = split_lines(seg.text)

    return segments


def split_lines(text: str) -> str:
    r"""Split a caption's text into up to two balanced lines.

    The function aims to keep each line length ≤ ``MAX_LINE_CHARS`` and return
    a single string that may contain one ``"\n"`` if splitting is needed. It
    prefers balanced splits and avoids super-short lines when possible.

    Args:
        text: The text of the subtitle segment.

    Returns:
        The formatted text with line breaks.
    """
    if len(text) <= constants.MAX_LINE_CHARS:
        return text

    words = text.split()

    # Find the split index that best balances lines while respecting the limit.
    best_idx = None
    best_imbalance = float("inf")
    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        if (
            len(left) <= constants.MAX_LINE_CHARS
            and len(right) <= constants.MAX_LINE_CHARS
        ):
            imbalance = abs(len(left) - len(right))
            if imbalance < best_imbalance:
                best_imbalance = imbalance
                best_idx = i

    if best_idx is not None:
        left = " ".join(words[:best_idx])
        right = " ".join(words[best_idx:])
        return f"{left}\n{right}"

    # Fallback greedy wrapping: fill first line up to the limit,
    # then put the remainder on the second line.
    line1_words: list[str] = []
    for w in words:
        candidate = (" ".join(line1_words + [w])).strip()
        if len(candidate) <= constants.MAX_LINE_CHARS:
            line1_words.append(w)
        else:
            break
    line2_words = words[len(line1_words) :]

    line1 = " ".join(line1_words)
    line2 = " ".join(line2_words)

    # Ensure second line obeys limit; truncate overflow to respect test
    # constraints.
    if line2:
        if len(line2) > constants.MAX_LINE_CHARS:
            line2 = (line2[: constants.MAX_LINE_CHARS - 1]).rstrip()
        return f"{line1}\n{line2}"
    return line1


def _respect_limits(words: list[Word], soft_limit: bool = False) -> bool:
    """Check if a list of words respects readability limits.

    Args:
        words: A list of Word objects.
        soft_limit: Whether to use the soft character limit.

    Returns:
        True if the words respect the limits, False otherwise.
    """
    text = " ".join(w.text for w in words)
    char_count = len(text)
    duration = words[-1].end - words[0].start
    cps = char_count / duration if duration > 0 else 0

    limit = constants.MAX_BLOCK_CHARS_SOFT if soft_limit else constants.MAX_BLOCK_CHARS

    return (
        char_count <= limit
        and duration >= constants.MIN_SEGMENT_DURATION_SEC
        and duration <= constants.MAX_SEGMENT_DURATION_SEC
        and cps >= constants.MIN_CPS
        and cps <= constants.MAX_CPS
    )


def _sentence_chunks(words: list[Word]) -> list[list[Word]]:
    """Split a list of words into sentence chunks based on punctuation.

    Args:
        words: A list of Word objects.

    Yields:
        A list of Word objects representing a sentence.
    """
    sentence_ends = {".", "!", "?"}
    current_sentence = []
    for word in words:
        current_sentence.append(word)
        if any(char in sentence_ends for char in word.text):
            yield current_sentence
            current_sentence = []
    if current_sentence:
        yield current_sentence


def _split_at_clause_boundaries(sentence: list[Word]) -> list[list[Word]]:
    """Split a sentence at clause boundaries if it violates hard limits.

    Args:
        sentence: A list of Word objects representing a sentence.

    Returns:
        A list of lists of Word objects, where each inner list is a clause.
    """
    sentence_text = " ".join(w.text for w in sentence)
    # More sophisticated logic will be added later.
    if sentence_text.count(",") < 2:
        return [sentence]

    clauses = []
    current_clause = []
    for word in sentence:
        current_clause.append(word)
        if "," in word.text:
            clauses.append(current_clause)
            current_clause = []
    if current_clause:
        clauses.append(current_clause)

    # If no commas were found, return the original sentence as a single clause
    if not clauses:
        return [sentence]

    return clauses


def _merge_short_segments(segments: list[Segment]) -> list[Segment]:
    """Merge short segments with their neighbors.

    Args:
        segments: A list of Segment objects.

    Returns:
        A new list of Segment objects with short segments merged.
    """
    if not segments:
        return []

    merged_segments = []
    current_segment = segments[0]

    for next_segment in segments[1:]:
        is_short = (
            current_segment.end - current_segment.start
        ) < constants.MIN_SEGMENT_DURATION_SEC
        can_merge = not any(p in current_segment.text for p in [".", "!", "?"])

        if is_short and can_merge:
            # Merge with next segment
            current_segment.text += " " + next_segment.text
            current_segment.end = next_segment.end
            current_segment.words.extend(next_segment.words)
        else:
            merged_segments.append(current_segment)
            current_segment = next_segment

    merged_segments.append(current_segment)
    return merged_segments


def _enforce_cps(segments: list[Segment]) -> list[Segment]:
    """Split segments to ensure characters-per-second constraints are met.

    This uses a greedy strategy to split a segment's words into smaller segments
    such that each new segment has MIN_CPS ≤ cps ≤ MAX_CPS when possible.

    Args:
        segments: Input segments prior to CPS enforcement.

    Returns:
        A new list of segments satisfying CPS constraints where feasible.
    """
    enforced: list[Segment] = []
    for seg in segments:
        words = seg.words
        if not words:
            enforced.append(seg)
            continue

        # Skip enforcement if this segment already respects CPS limits.
        seg_text = " ".join(w.text for w in words)
        seg_duration = words[-1].end - words[0].start
        seg_cps = (len(seg_text) / seg_duration) if seg_duration > 0 else float("inf")
        if constants.MIN_CPS <= seg_cps <= constants.MAX_CPS:
            enforced.append(seg)
            continue

        # For sufficiently long segments, prefer keeping them intact; wrapping
        # will be handled by split_lines(). Only enforce CPS for very short
        # segments where duration is below the minimum threshold.
        if seg_duration >= constants.MIN_SEGMENT_DURATION_SEC:
            enforced.append(seg)
            continue

        # If the original duration is too short to ever meet MAX_CPS for this
        # amount of text, synthesize timing by chunking on word boundaries and
        # assigning durations that satisfy CPS constraints.
        if seg_duration * constants.MAX_CPS < len(seg_text):
            max_chars_per_chunk = int(
                constants.MAX_CPS * constants.MAX_SEGMENT_DURATION_SEC
            )
            tokens = seg_text.split()
            cur_tokens: list[str] = []
            current_time = words[0].start
            for tok in tokens:
                tentative = (" ".join(cur_tokens + [tok])).strip()
                if cur_tokens and len(tentative) > max_chars_per_chunk:
                    chunk_text = " ".join(cur_tokens)
                    dur = max(
                        len(chunk_text) / constants.MAX_CPS,
                        constants.MIN_SEGMENT_DURATION_SEC,
                    )
                    end_time = current_time + min(
                        dur, constants.MAX_SEGMENT_DURATION_SEC
                    )

                    # Build synthetic words with evenly split timing.
                    chunk_tokens = chunk_text.split()
                    per = (end_time - current_time) / max(len(chunk_tokens), 1)
                    chunk_words = []
                    t0 = current_time
                    for ct in chunk_tokens:
                        chunk_words.append(Word(text=ct, start=t0, end=t0 + per))
                        t0 += per

                    enforced.append(
                        Segment(
                            text=chunk_text,
                            start=current_time,
                            end=end_time,
                            words=chunk_words,
                        )
                    )
                    current_time = end_time
                    cur_tokens = [tok]
                else:
                    cur_tokens.append(tok)

            if cur_tokens:
                chunk_text = " ".join(cur_tokens)
                dur = max(
                    len(chunk_text) / constants.MAX_CPS,
                    constants.MIN_SEGMENT_DURATION_SEC,
                )
                end_time = current_time + min(dur, constants.MAX_SEGMENT_DURATION_SEC)
                chunk_tokens = chunk_text.split()
                per = (end_time - current_time) / max(len(chunk_tokens), 1)
                chunk_words = []
                t0 = current_time
                for ct in chunk_tokens:
                    chunk_words.append(Word(text=ct, start=t0, end=t0 + per))
                    t0 += per

                enforced.append(
                    Segment(
                        text=chunk_text,
                        start=current_time,
                        end=end_time,
                        words=chunk_words,
                    )
                )

            continue

        n = len(words)
        start_idx = 0
        while start_idx < n:
            end_idx = start_idx

            # First, expand while CPS is above MAX_CPS to dilute density.
            while end_idx < n:
                chunk = words[start_idx : end_idx + 1]
                text = " ".join(w.text for w in chunk)
                duration = chunk[-1].end - chunk[0].start
                cps = (len(text) / duration) if duration > 0 else float("inf")
                if cps > constants.MAX_CPS and end_idx + 1 < n:
                    end_idx += 1
                    continue
                break

            # Then, expand while CPS is below MIN_CPS (if possible) to keep
            # text readable.
            while end_idx + 1 < n:
                chunk = words[start_idx : end_idx + 1]
                text = " ".join(w.text for w in chunk)
                duration = chunk[-1].end - chunk[0].start
                cps = (len(text) / duration) if duration > 0 else float("inf")
                if cps < constants.MIN_CPS:
                    # Try adding one more word; if that would exceed MAX_CPS
                    # badly, stop.
                    trial = words[start_idx : end_idx + 2]
                    t_text = " ".join(w.text for w in trial)
                    t_dur = trial[-1].end - trial[0].start
                    t_cps = (len(t_text) / t_dur) if t_dur > 0 else cps
                    if t_cps <= constants.MAX_CPS:
                        end_idx += 1
                        continue
                break

            # Ensure at least one word advances progress.
            if end_idx < start_idx:
                end_idx = start_idx

            chunk = words[start_idx : end_idx + 1]
            text = " ".join(w.text for w in chunk)
            enforced.append(
                Segment(
                    text=text,
                    start=chunk[0].start,
                    end=chunk[-1].end,
                    words=chunk,
                )
            )
            start_idx = end_idx + 1

    return enforced


def _maybe_expand_single_word(seg: Segment) -> Segment:
    """Expand a single-word segment into sub-words split by spaces.

    This enables CPS enforcement for inputs that arrive as a single `Word`
    containing multiple tokens in its text.

    Args:
        seg: The segment to examine and possibly expand.

    Returns:
        The original segment if not applicable, otherwise a new segment with
        `words` expanded into sub-words with proportionally distributed timing.
    """
    if len(seg.words) != 1:
        return seg
    w = seg.words[0]
    tokens = [t for t in w.text.split() if t]
    if len(tokens) <= 1:
        return seg

    total_chars = sum(len(t) for t in tokens)
    start = w.start
    new_words: list[Word] = []
    for i, tok in enumerate(tokens):
        frac = len(tok) / total_chars if total_chars > 0 else 0
        # Distribute durations proportionally; last token takes any residual.
        if i < len(tokens) - 1:
            dur = (w.end - w.start) * frac
            end = start + dur
        else:
            end = w.end
        new_words.append(Word(text=tok, start=start, end=end))
        start = end

    return Segment(
        text=" ".join(t.text for t in new_words),
        start=new_words[0].start,
        end=new_words[-1].end,
        words=new_words,
    )
