"""Tests to verify API routes pass only valid parameters to WhisperPipeline.process().

This test ensures that stabilize/demucs/vad/vad_threshold parameters are NOT
passed to the pipeline's process() method, since these are handled by a
separate post-processing step (stabilize_timestamps).
"""

from __future__ import annotations

import inspect


class TestRouteProcessParameters:
    """Test that routes only pass valid parameters to WhisperPipeline.process()."""

    def test_process_method_signature__does_not_accept_stabilize_params(self) -> None:
        """Verify WhisperPipeline.process() doesn't accept stabilization params.

        This test documents the expected behavior: the process() method should
        NOT accept stabilize, demucs, vad, or vad_threshold parameters. These
        are handled by stabilize_timestamps() after transcription.
        """
        from insanely_fast_whisper_api.core.pipeline import WhisperPipeline

        sig = inspect.signature(WhisperPipeline.process)
        param_names = set(sig.parameters.keys())

        # These params should NOT be in the process() signature
        invalid_params = {"stabilize", "demucs", "vad", "vad_threshold"}
        actual_invalid = param_names & invalid_params

        assert len(actual_invalid) == 0, (
            f"process() should not accept {actual_invalid} — these are post-processing params"
        )

    def test_transcription_route__does_not_pass_stabilize_to_process(self) -> None:
        """Verify create_transcription doesn't pass stabilize params to process().

        The route should call asr_pipeline.process() WITHOUT stabilize/demucs/vad
        parameters. Stabilization is handled separately via stabilize_timestamps().
        """
        from insanely_fast_whisper_api.api import routes

        # Get the source code of create_transcription
        source = inspect.getsource(routes.create_transcription)

        # Check that the process() call doesn't include stabilization params
        # This is a simple heuristic: if 'stabilize=stabilize' appears in the
        # process() call context, it's passing invalid params
        process_call_section = _extract_process_call(source)

        assert "stabilize=stabilize" not in process_call_section, (
            "create_transcription passes 'stabilize' to process() but it's not a valid param"
        )
        assert "demucs=demucs" not in process_call_section, (
            "create_transcription passes 'demucs' to process() but it's not a valid param"
        )
        assert "vad=vad" not in process_call_section, (
            "create_transcription passes 'vad' to process() but it's not a valid param"
        )
        assert "vad_threshold=vad_threshold" not in process_call_section, (
            "create_transcription passes 'vad_threshold' to process() but it's not a valid param"
        )

    def test_translation_route__does_not_pass_stabilize_to_process(self) -> None:
        """Verify create_translation doesn't pass stabilize params to process().

        The route should call asr_pipeline.process() WITHOUT stabilize/demucs/vad
        parameters. Stabilization is handled separately via stabilize_timestamps().
        """
        from insanely_fast_whisper_api.api import routes

        # Get the source code of create_translation
        source = inspect.getsource(routes.create_translation)

        # Check that the process() call doesn't include stabilization params
        process_call_section = _extract_process_call(source)

        assert "stabilize=stabilize" not in process_call_section, (
            "create_translation passes 'stabilize' to process() but it's not a valid param"
        )
        assert "demucs=demucs" not in process_call_section, (
            "create_translation passes 'demucs' to process() but it's not a valid param"
        )
        assert "vad=vad" not in process_call_section, (
            "create_translation passes 'vad' to process() but it's not a valid param"
        )
        assert "vad_threshold=vad_threshold" not in process_call_section, (
            "create_translation passes 'vad_threshold' to process() but it's not a valid param"
        )


def _extract_process_call(source: str) -> str:
    """Extract the asr_pipeline.process() call section from source code.

    This function finds the process() call and extracts only the code until
    the closing parenthesis of that specific call, avoiding capturing
    subsequent code like stabilize_timestamps().

    Args:
        source: Source code string to search.

    Returns:
        The substring containing only the process() call, or empty if not found.
    """
    idx = source.find("asr_pipeline.process(")
    if idx == -1:
        return ""

    # Find the matching closing parenthesis for the process() call
    # by counting nested parentheses
    paren_count = 0
    end_idx = idx
    started = False
    for i, char in enumerate(source[idx:], start=idx):
        if char == "(":
            paren_count += 1
            started = True
        elif char == ")":
            paren_count -= 1
            if started and paren_count == 0:
                end_idx = i + 1
                break

    return source[idx:end_idx]
