"""Segmentation logic for creating readable subtitles from ASR word timestamps."""

from __future__ import annotations

import dataclasses
import logging

from insanely_fast_whisper_rocm.utils import constants

logger = logging.getLogger(__name__)


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


def _expand_multi_token_words(words: list[Word]) -> list[Word]:
    """Expand Word objects containing multiple space-separated tokens.

    This function splits any Word whose text contains multiple tokens
    (separated by spaces) into multiple Word objects with proportionally
    distributed timing.

    Args:
        words: Original word list, potentially with multi-token Word objects.

    Returns:
        A new list of Word objects where each Word contains a single token.
    """
    expanded: list[Word] = []
    for w in words:
        tokens = [t for t in w.text.split() if t]
        if len(tokens) <= 1:
            expanded.append(w)
            continue

        # Distribute timing proportionally across tokens
        total_chars = sum(len(t) for t in tokens)
        start = w.start
        for i, tok in enumerate(tokens):
            frac = len(tok) / total_chars if total_chars > 0 else 0
            if i < len(tokens) - 1:
                dur = (w.end - w.start) * frac
                end = start + dur
            else:
                end = w.end
            expanded.append(Word(text=tok, start=start, end=end))
            start = end

    return expanded


def _sanitize_words_timing(words: list[Word]) -> list[Word]:
    """Return a sanitized copy of ``words`` with stable, monotonic timings.

    - Ensure each word has a positive duration by expanding zero/negative
      durations by a minimal epsilon.
    - Enforce non-decreasing starts across the sequence by nudging starts
      forward when necessary; end is adjusted accordingly.

    Args:
        words: Original word list from ASR output.

    Returns:
        A new list of ``Word`` with sanitized ``start``/``end`` values.
    """
    if not words:
        return []

    eps = constants.MIN_WORD_DURATION_SEC
    sanitized: list[Word] = []
    prev_end = max(0.0, words[0].start)
    for w in words:
        start = max(w.start, prev_end)
        end = w.end
        if end <= start:
            end = start + eps
        sanitized.append(Word(text=w.text, start=start, end=end))
        prev_end = end
    return sanitized


def segment_words(words: list[Word]) -> list[Segment]:
    """Orchestrates the full segmentation process.

    This function will take a list of words and return a list of readable subtitle
    segments.

    Args:
        words: A list of Word objects from the ASR output.

    Returns:
        A list of Segment objects formatted for readability.
    """
    logger.debug("segment_words: processing %d input words", len(words))
    # Expand multi-token Word objects into individual tokens first.
    # This handles test/edge cases where a Word contains multiple tokens.
    words = _expand_multi_token_words(words)
    logger.debug("After expansion: %d words", len(words))
    # Sanitize timings to avoid zero/negative durations and enforce monotonicity.
    words = _sanitize_words_timing(words)
    logger.debug("After sanitization: %d words", len(words))

    sentences = list(_sentence_chunks(words))
    logger.debug("Split into %d sentence chunks", len(sentences))
    segments = []
    for sentence in sentences:
        txt = " ".join(w.text for w in sentence)
        wrapped = split_lines(txt)
        lines = wrapped.split("\n")
        sent_dur = sentence[-1].end - sentence[0].start

        logger.debug(
            "Processing sentence: dur=%.2fs, len=%d chars, text=%r",
            sent_dur,
            len(txt),
            txt[:60],
        )

        # Keep as single segment if it can be wrapped into ≤2 lines
        # with each line ≤ MAX_LINE_CHARS, regardless of total char count
        if (
            len(lines) <= 2
            and all(len(line) <= constants.MAX_LINE_CHARS for line in lines)
            and _respect_limits(sentence)
        ):
            logger.debug("  -> Branch A: wrapping works + respects limits")
            segments.append(
                Segment(
                    text=wrapped,
                    start=sentence[0].start,
                    end=sentence[-1].end,
                    words=sentence,
                )
            )
        elif not _respect_limits(sentence):
            # If wrapping doesn't work and limits aren't respected, split into clauses
            logger.debug(
                "  -> Branch B: doesn't respect limits, splitting into clauses"
            )
            clauses = _split_at_clause_boundaries(sentence, force_line_limit=False)
            logger.debug("  -> Split into %d clauses", len(clauses))
            for clause in clauses:
                clause_dur = clause[-1].end - clause[0].start
                segments.append(
                    Segment(
                        text=" ".join(w.text for w in clause),
                        start=clause[0].start,
                        end=clause[-1].end,
                        words=clause,
                    )
                )
                logger.debug("  -> Clause: dur=%.2fs", clause_dur)
        else:
            logger.debug("  -> Branch C: wrapping doesn't work, but respects limits")
            segments.append(
                Segment(
                    text=" ".join(w.text for w in sentence),
                    start=sentence[0].start,
                    end=sentence[-1].end,
                    words=sentence,
                )
            )
    logger.debug("Before merge_short_segments: %d segments", len(segments))
    segments = _merge_short_segments(segments)
    logger.debug("After merge_short_segments: %d segments", len(segments))

    # Expand single-word segments into sub-words (based on whitespace) so CPS
    # enforcement can operate meaningfully on realistic units.
    segments = [_maybe_expand_single_word(seg) for seg in segments]

    # After word expansion, re-apply character limits to ensure segments
    # are appropriately sized for readability
    segments = _reapply_character_limits(segments)
    # Enforce CPS constraints only for single-word-origin segments. Multi-word
    # segments are handled by line wrapping and duration constraints.
    logger.debug("Before enforce_cps: %d segments", len(segments))
    segments = _enforce_cps(segments)
    logger.debug("After enforce_cps: %d segments", len(segments))
    segments = _merge_short_segments(segments)
    logger.debug("After second merge_short_segments: %d segments", len(segments))
    segments = _enforce_duration_limits(segments)
    segments = _merge_short_segments(segments)
    logger.debug("After third merge_short_segments: %d segments", len(segments))

    # Guarantee monotonic timings after any synthetic duration adjustments.
    segments = _ensure_monotonic_segments(segments)
    logger.debug("After ensure_monotonic: %d segments", len(segments))

    # Apply final text formatting (line wrapping) per segment.
    for seg in segments:
        seg.text = split_lines(seg.text)

    logger.debug("segment_words returning %d final segments", len(segments))
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

    # Find candidate split indices where both sides respect the line length.
    candidates: list[tuple[int, int]] = []  # (index, score)
    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        if (
            len(left) <= constants.MAX_LINE_CHARS
            and len(right) <= constants.MAX_LINE_CHARS
        ):
            # Base score favors balanced lines (smaller imbalance = better)
            imbalance = abs(len(left) - len(right))
            score = 1000 - min(999, imbalance)  # higher is better

            # Prefer a split right at a comma boundary: left endswith ","
            if left.endswith(","):
                score += 10000  # dominate choice when feasible

            # Prefer soft boundary words at the end of the first line
            last_word_left = words[i - 1].strip(",.?!:;\"'()[]{}")
            if last_word_left.lower() in set(constants.SOFT_BOUNDARY_WORDS):
                score += 5000

            candidates.append((i, score))

    if candidates:
        # Take the candidate with the best score; tie-break by minimal imbalance
        def cand_key(item: tuple[int, int]) -> tuple[int, int]:
            """Key function for ranking split candidates by score and balance.

            Args:
                item: A tuple containing (index, score) for a candidate split.

            Returns:
                A tuple of (score, negative imbalance) for sorting.
            """
            idx, sc = item
            left = " ".join(words[:idx])
            right = " ".join(words[idx:])
            return (sc, -abs(len(left) - len(right)))

        best_idx = max(candidates, key=cand_key)[0]
        left = " ".join(words[:best_idx])
        right = " ".join(words[best_idx:])
        return f"{left}\n{right}"

    # Fallback: enforce at most two lines.
    # First, if the text contains a comma, split immediately after the first comma
    # to avoid mid-phrase breaks (test prefers this behavior), but only if both
    # sides respect the per-line character limit.
    for idx_c, tok in enumerate(words):
        if tok.endswith(",") and idx_c + 1 < len(words):
            left = " ".join(words[: idx_c + 1]).strip()
            right = " ".join(words[idx_c + 1 :]).strip()
            if (
                left
                and right
                and (
                    len(left) <= constants.MAX_LINE_CHARS
                    and len(right) <= constants.MAX_LINE_CHARS
                )
            ):
                return f"{left}\n{right}"

    # Prefer to end the first line at the last comma that fits.
    # Otherwise, fill up to the limit and put the remainder on the second line.
    # 1) Find maximum tokens that fit in first line
    first_line_tokens: list[str] = []
    idx = 0
    while idx < len(words):
        nxt = words[idx]
        tentative = (" ".join(first_line_tokens + [nxt])).strip()
        if first_line_tokens and len(tentative) > constants.MAX_LINE_CHARS:
            break
        first_line_tokens.append(nxt)
        idx += 1

    # 2) If any comma-terminated token exists within the first-line window,
    #    split right after the last such token to keep the comma at end of line 1.
    last_comma_pos = -1
    acc: list[str] = []
    for j, tok in enumerate(first_line_tokens, start=1):
        acc.append(tok)
        if tok.endswith(","):
            last_comma_pos = j

    if last_comma_pos != -1:
        first_line = " ".join(first_line_tokens[:last_comma_pos]).strip()
        second_line = " ".join(words[last_comma_pos:]).strip()
        if (
            first_line
            and second_line
            and (
                len(first_line) <= constants.MAX_LINE_CHARS
                and len(second_line) <= constants.MAX_LINE_CHARS
            )
        ):
            return f"{first_line}\n{second_line}"

    # 3) Default: find a balanced split that keeps both lines under the limit
    first_line = " ".join(first_line_tokens)
    if idx >= len(words):
        return first_line
    second_line = " ".join(words[idx:])

    # Try exhaustive split search across all token boundaries to enforce limits.
    # Choose the split that minimizes the max line length, then minimizes imbalance.
    best_pair: tuple[str, str] | None = None
    best_score: tuple[int, int] | None = None
    for split_idx in range(1, len(words)):
        left = " ".join(words[:split_idx])
        right = " ".join(words[split_idx:])
        if (
            len(left) <= constants.MAX_LINE_CHARS
            and len(right) <= constants.MAX_LINE_CHARS
        ):
            score = (max(len(left), len(right)), abs(len(left) - len(right)))
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (left, right)

    if best_pair is not None:
        return f"{best_pair[0]}\n{best_pair[1]}"

    # As a strict fallback, hard-wrap to the per-line cap without breaking words:
    # fill the first line up to the limit, put remaining tokens on the second line.
    wrap_first: list[str] = []
    pos = 0
    while pos < len(words):
        candidate = (" ".join(wrap_first + [words[pos]])).strip()
        if wrap_first and len(candidate) > constants.MAX_LINE_CHARS:
            break
        wrap_first.append(words[pos])
        pos += 1
    wrap_second = words[pos:]
    return f"{' '.join(wrap_first)}\n{' '.join(wrap_second)}"


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

    This function groups words into sentences by looking for sentence-ending
    punctuation (., !, ?). It accumulates words until it finds punctuation,
    ensuring that individual words without punctuation are grouped together
    rather than becoming separate sentences.

    This is a generator function that yields sentence chunks. May return
    early without yielding if input is empty.

    Args:
        words: A list of Word objects.

    Yields:
        Lists of Word objects, each representing a sentence.
    """
    sentence_ends = {".", "!", "?"}
    current_sentence = []

    for word in words:
        current_sentence.append(word)
        # Check if this word ends with sentence-ending punctuation
        # Strip whitespace and check the last character
        text_stripped = word.text.strip()
        if text_stripped and text_stripped[-1] in sentence_ends:
            yield current_sentence
            current_sentence = []

    # Yield any remaining words as a final sentence
    if current_sentence:
        yield current_sentence


def _split_at_clause_boundaries(
    sentence: list[Word], *, force_line_limit: bool = False
) -> list[list[Word]]:
    """Split a sentence into clauses at natural boundaries.

    This function prefers splitting at commas, but avoids creating clauses that
    end with awkward words like articles, prepositions, or auxiliary verbs.

    Args:
        sentence: Words forming the sentence.
        force_line_limit: Whether to force downstream splitting to respect
            ``constants.MAX_LINE_CHARS`` in addition to block limits.

    Returns:
        list[list[Word]]: Clause chunks derived from ``sentence``.
    """
    sentence_text = " ".join(w.text for w in sentence)
    hard_limit = constants.MAX_LINE_CHARS if force_line_limit else None

    # If the sentence is not too long and has few commas, return as single clause
    if len(sentence_text) <= constants.MAX_BLOCK_CHARS and sentence_text.count(",") < 2:
        return [sentence]

    # First, try to split at commas (existing logic)
    clauses = []
    current_clause = []
    for word in sentence:
        current_clause.append(word)
        if "," in word.text:
            clauses.append(current_clause)
            current_clause = []
    if current_clause:
        clauses.append(current_clause)

    # If no commas were found or we still have very long clauses, apply
    # additional splitting
    if not clauses or any(
        len(" ".join(w.text for w in clause)) > constants.MAX_BLOCK_CHARS
        for clause in clauses
    ):
        return _split_long_text_aggressively(sentence, max_chars=hard_limit)

    # If we have clauses but some are still too long (by char or duration), split them
    final_clauses = []
    for clause in clauses:
        clause_text = " ".join(w.text for w in clause)
        clause_duration = clause[-1].end - clause[0].start
        clause_limit = hard_limit or constants.MAX_BLOCK_CHARS

        # Check both character length AND duration
        if (
            len(clause_text) > clause_limit
            or clause_duration > constants.MAX_SEGMENT_DURATION_SEC
        ):
            # Split this clause into smaller chunks
            logger.debug(
                "    Clause exceeds limits (dur=%.2fs, len=%d), splitting by duration",
                clause_duration,
                len(clause_text),
            )
            sub_clauses = _split_by_duration(clause, constants.MAX_SEGMENT_DURATION_SEC)
            logger.debug("    -> Split into %d sub-clauses", len(sub_clauses))
            for sub in sub_clauses:
                sub_dur = sub[-1].end - sub[0].start
                logger.debug("      Sub-clause: dur=%.2fs", sub_dur)
            final_clauses.extend(sub_clauses)
        else:
            final_clauses.append(clause)

    return final_clauses


def _split_by_duration(words: list[Word], max_duration: float) -> list[list[Word]]:
    """Split words into chunks where each chunk duration <= max_duration.

    Args:
        words: List of Word objects to split.
        max_duration: Maximum duration (in seconds) for each chunk.

    Returns:
        List of word chunks, each respecting the duration limit.
    """
    if not words:
        return []

    chunks = []
    current_chunk = [words[0]]

    for word in words[1:]:
        # Calculate duration if we add this word
        potential_duration = word.end - current_chunk[0].start

        if potential_duration > max_duration:
            # Current chunk would exceed limit, finalize it and start new one
            chunks.append(current_chunk)
            current_chunk = [word]
        else:
            current_chunk.append(word)

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _split_long_text_aggressively(
    words: list[Word], *, max_chars: int | None = None
) -> list[list[Word]]:
    """Aggressively split long text into smaller chunks.

    Args:
        words: Sequence of words to split.
        max_chars: Optional override for the maximum characters per chunk.

    Returns:
        list[list[Word]]: Chunks whose joined text respects ``max_chars``.
    """
    if not words:
        return []

    limit = max_chars or constants.MAX_BLOCK_CHARS
    text = " ".join(w.text for w in words)
    if len(text) <= limit:
        wrapped_lines = split_lines(text).split("\n")
        if all(len(line) <= constants.MAX_LINE_CHARS for line in wrapped_lines):
            return [words]

    # Try to find natural split points first
    natural_splits = _find_natural_split_points(words)
    if natural_splits:
        chunks = []
        start_idx = 0
        for split_idx in natural_splits:
            if start_idx < split_idx:
                chunk = words[start_idx:split_idx]
                if chunk:
                    chunks.append(chunk)
            start_idx = split_idx
        if start_idx < len(words):
            chunk = words[start_idx:]
            if chunk:
                chunks.append(chunk)

        # Check if all chunks respect limits
        if all(len(" ".join(w.text for w in chunk)) <= limit for chunk in chunks):
            # Clean up awkward endings before returning
            return _clean_awkward_endings(chunks)

    # If natural splits do not work, use word-based chunking with the same limit
    return _chunk_by_word_limits(words, max_chars=limit)


def _find_natural_split_points(words: list[Word]) -> list[int]:
    """Find natural split points in text (conjunctions, relative pronouns, etc.).

    Args:
        words: List of words to analyze.

    Returns:
        List of indices where natural splits occur.
    """
    natural_boundary_words = {
        "and",
        "but",
        "or",
        "so",
        "for",
        "nor",
        "yet",
        "while",
        "although",
        "though",
        "even",
        "whereas",
        "however",
        "therefore",
        "moreover",
        "furthermore",
        "consequently",
        "meanwhile",
        "otherwise",
        "instead",
        "besides",
        "additionally",
        "similarly",
        "likewise",
        "further",
        "also",
        "plus",
    }

    split_points = []
    for i, word in enumerate(words):
        word_text = word.text.strip(".,!?;:").lower()
        if word_text in natural_boundary_words and i > 0 and i < len(words) - 1:
            split_points.append(i + 1)

    return split_points


def _reapply_character_limits(segments: list[Segment]) -> list[Segment]:
    """Re-apply character limits after word expansion.

    This function checks if any segments exceed character limits after word expansion
    and breaks them up if necessary. Only breaks up segments that are truly problematic.

    Args:
        segments: List of segments that may need re-splitting.

    Returns:
        List of segments with appropriate character limits applied.
    """
    result = []

    for seg in segments:
        # Get the text without newlines for accurate length checking
        clean_text = seg.text.replace("\n", " ")
        seg_text = clean_text

        wrapped_text = split_lines(seg_text)
        wrapped_lines = wrapped_text.split("\n")
        max_line_length = (
            max(len(line) for line in wrapped_lines) if wrapped_lines else 0
        )

        lines_within_limit = (
            len(wrapped_lines) <= 2 and max_line_length <= constants.MAX_LINE_CHARS
        )

        # Prefer per-line wrapping over block length: if wrapped into two lines
        # within per-line cap, keep the segment intact even if block length
        # exceeds MAX_BLOCK_CHARS.
        if lines_within_limit:
            result.append(seg)
            continue

        # Otherwise split aggressively to enforce both block and line limits
        sub_segments = _split_long_text_aggressively(
            seg.words, max_chars=constants.MAX_LINE_CHARS
        )
        for sub_seg_words in sub_segments:
            if sub_seg_words:
                result.append(
                    Segment(
                        text=" ".join(w.text for w in sub_seg_words),
                        start=sub_seg_words[0].start,
                        end=sub_seg_words[-1].end,
                        words=sub_seg_words,
                    )
                )

    return result


def _clean_awkward_endings(chunks: list[list[Word]]) -> list[list[Word]]:
    """Move awkward endings from one chunk to the next.

    Words that should not end a segment (articles, prepositions, conjunctions)
    are moved to the following chunk to avoid mid-phrase breaks.

    Args:
        chunks: List of word chunks to clean up.

    Returns:
        Adjusted chunks with awkward endings moved to next chunk.
    """
    awkward_endings = {
        "a",
        "an",
        "the",
        "in",
        "on",
        "at",
        "of",
        "to",
        "for",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
    }

    adjusted_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunk) > 1:  # Only adjust multi-word chunks
            last_word = chunk[-1].text.strip(".!?,;:").lower()

            # Check if there's a next chunk that's a single word
            has_single_word_next = i + 1 < len(chunks) and len(chunks[i + 1]) == 1

            # Move a word if:
            # 1. Current chunk ends awkwardly, OR
            # 2. Next chunk is a single word (to prevent orphaned words)
            should_move = last_word in awkward_endings or has_single_word_next

            if should_move and i + 1 < len(chunks):
                # When next chunk is a single word, move TWO words if possible
                # to avoid creating a new awkward ending
                if has_single_word_next and len(chunk) > 3:
                    # Move last 2 words to prevent orphaned single-word AND
                    # avoid creating a new awkward ending
                    words_to_move = chunk[-2:]
                    adjusted_chunks.append(chunk[:-2])
                    for word in reversed(words_to_move):
                        chunks[i + 1].insert(0, word)
                elif len(chunk) > 2:
                    # Normal case: move one word
                    awkward_word = chunk[-1]
                    adjusted_chunks.append(chunk[:-1])
                    chunks[i + 1].insert(0, awkward_word)
                else:
                    # Can't move without creating problems
                    adjusted_chunks.append(chunk)
            else:
                adjusted_chunks.append(chunk)
        else:
            adjusted_chunks.append(chunk)

    # Filter out empty chunks
    return [chunk for chunk in adjusted_chunks if chunk]


def _chunk_by_word_limits(
    words: list[Word], *, max_chars: int | None = None
) -> list[list[Word]]:
    """Split text into chunks that respect character limits.

    This function creates chunks based on character limits and then cleans up
    awkward endings.

    Args:
        words: Words to split while preserving boundaries.
        max_chars: Optional override for the maximum characters per chunk.

    Returns:
        list[list[Word]]: Word chunks whose joined text length obeys ``max_chars``.
    """
    limit = max_chars or constants.MAX_BLOCK_CHARS

    # Create chunks based on character limit
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_text = word.text + " "  # Include space
        word_length = len(word_text)

        # If adding this word would exceed the limit and we have content,
        # start new chunk
        if current_chunk and current_length + word_length > limit:
            chunks.append(current_chunk)
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)

    # Clean up awkward endings
    return _clean_awkward_endings(chunks)


def _merge_short_segments(segments: list[Segment]) -> list[Segment]:
    """Merge short segments with their neighbors.

    This function merges segments that are too short (by duration or word count)
    with neighboring segments to improve readability. Single-word segments without
    sentence-ending punctuation are always merged.

    Args:
        segments: A list of Segment objects.

    Returns:
        A new list of Segment objects with short segments merged.
    """
    if not segments:
        return []

    merged_segments: list[Segment] = []

    # Work on copies of the original segments so callers' data is never mutated.
    def _clone_segment(seg: Segment) -> Segment:
        """Create a deep copy of a segment.

        Args:
            seg: The segment to clone.

        Returns:
            A new Segment object with the same text, timing, and words.
        """
        return Segment(
            text=seg.text,
            start=seg.start,
            end=seg.end,
            words=list(seg.words),
        )

    current_segment = _clone_segment(segments[0])

    for next_segment in segments[1:]:
        duration = current_segment.end - current_segment.start
        word_count = len(current_segment.words)
        has_sentence_end = any(p in current_segment.text for p in [".", "!", "?"])

        # Merge if:
        # 1. Duration is too short AND no sentence-ending punctuation, OR
        # 2. Single word without sentence-ending punctuation (avoid orphaned words)
        is_short_duration = duration < constants.MIN_SEGMENT_DURATION_SEC
        is_single_word = word_count == 1

        should_merge = (is_short_duration or is_single_word) and not has_sentence_end

        if should_merge:
            # Check if merging would exceed MAX_SEGMENT_DURATION_SEC
            merged_duration = next_segment.end - current_segment.start
            if merged_duration <= constants.MAX_SEGMENT_DURATION_SEC:
                # Safe to merge: construct a new Segment instance combining both.
                combined_words = list(current_segment.words) + list(next_segment.words)
                current_segment = Segment(
                    text=f"{current_segment.text} {next_segment.text}",
                    start=current_segment.start,
                    end=next_segment.end,
                    words=combined_words,
                )
            else:
                # Merging would exceed duration limit, finalize current and move on
                merged_segments.append(current_segment)
                current_segment = _clone_segment(next_segment)
        else:
            merged_segments.append(current_segment)
            current_segment = _clone_segment(next_segment)

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
    # Small epsilon for floating-point comparison tolerance
    eps = 1e-6
    enforced: list[Segment] = []
    for seg in segments:
        words = seg.words
        if not words:
            enforced.append(seg)
            continue

        # Skip enforcement if this segment already respects CPS limits.
        # Use epsilon tolerance to handle floating-point precision issues.
        seg_text = " ".join(w.text for w in words)
        seg_duration = words[-1].end - words[0].start
        seg_cps = (len(seg_text) / seg_duration) if seg_duration > 0 else float("inf")
        if constants.MIN_CPS - eps <= seg_cps <= constants.MAX_CPS + eps:
            enforced.append(seg)
            continue

        # For sufficiently long segments, prefer keeping them intact; wrapping
        # will be handled by split_lines(). Only enforce CPS for very short
        # segments where duration is below the minimum threshold.
        # Note: Even when duration is above the minimum, we still need to enforce
        # CPS if it is too high or too low for readability; proceed to greedy
        # splitting below instead of skipping.

        # If the original duration is too short to ever meet MAX_CPS for this
        # amount of text, synthesize timing by chunking on word boundaries and
        # assigning durations that satisfy CPS constraints.

        # Only use synthetic timing if the original duration is also below max duration
        # and we can actually fix the CPS issue without exceeding duration limits
        if (
            seg_duration * constants.MAX_CPS < len(seg_text)
            and seg_duration <= constants.MAX_SEGMENT_DURATION_SEC
        ):
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
                    # Cap duration to not exceed maximum segment duration
                    dur = min(dur, constants.MAX_SEGMENT_DURATION_SEC)
                    end_time = current_time + dur

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
                # Cap duration to not exceed maximum segment duration
                dur = min(dur, constants.MAX_SEGMENT_DURATION_SEC)
                end_time = current_time + dur
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


def _enforce_duration_limits(segments: list[Segment]) -> list[Segment]:
    """Split segments that exceed the maximum allowed duration.

    Args:
        segments: Candidate segments after CPS enforcement.

    Returns:
        Segments whose durations do not exceed ``MAX_SEGMENT_DURATION_SEC``.
    """
    max_duration = constants.MAX_SEGMENT_DURATION_SEC
    min_duration = constants.MIN_SEGMENT_DURATION_SEC
    eps = 1e-6
    enforced: list[Segment] = []

    for seg in segments:
        duration = seg.end - seg.start
        if duration <= max_duration + eps or not seg.words:
            enforced.append(seg)
            continue

        chunk_words: list[Word] = []
        for word in seg.words:
            if not chunk_words:
                chunk_words.append(word)
                continue

            tentative_duration = word.end - chunk_words[0].start
            if tentative_duration <= max_duration + eps:
                chunk_words.append(word)
                continue

            enforced.append(
                Segment(
                    text=" ".join(w.text for w in chunk_words),
                    start=chunk_words[0].start,
                    end=chunk_words[-1].end,
                    words=list(chunk_words),
                )
            )
            chunk_words = [word]

        if not chunk_words:
            continue

        trailing_duration = chunk_words[-1].end - chunk_words[0].start
        if trailing_duration < min_duration - eps and enforced:
            prev_seg = enforced.pop()
            combined_words = prev_seg.words + chunk_words
            combined_duration = combined_words[-1].end - combined_words[0].start
            if combined_duration <= max_duration + eps:
                enforced.append(
                    Segment(
                        text=" ".join(w.text for w in combined_words),
                        start=combined_words[0].start,
                        end=combined_words[-1].end,
                        words=combined_words,
                    )
                )
                continue
            enforced.append(prev_seg)

        enforced.append(
            Segment(
                text=" ".join(w.text for w in chunk_words),
                start=chunk_words[0].start,
                end=chunk_words[-1].end,
                words=list(chunk_words),
            )
        )

    return enforced


def _ensure_monotonic_segments(segments: list[Segment]) -> list[Segment]:
    """Return segments with non-decreasing start times.

    Args:
        segments: Candidate segments that may include synthetic durations.

    Returns:
        Segments adjusted so each start time is greater than or equal to the
        previous segment end time. Word timings are shifted consistently.
    """
    adjusted: list[Segment] = []
    prev_end = 0.0
    for seg in segments:
        start = seg.start
        end = seg.end
        words = seg.words

        if start < prev_end:
            shift = prev_end - start
            start += shift
            end += shift
            words = [
                Word(text=w.text, start=w.start + shift, end=w.end + shift)
                for w in words
            ]

        if end < start:
            end = start
            words = [Word(text=w.text, start=start, end=start) for w in words]

        adjusted.append(Segment(text=seg.text, start=start, end=end, words=words))
        prev_end = end

    return adjusted


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
