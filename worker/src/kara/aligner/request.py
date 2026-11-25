"""What the karaoke pipeline sends the aligner's worker process for one song.

Kept free of the aligner's dependencies, so the pipeline can build requests without loading them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from worker.kara.config import LabelRules


class AlignRequest(BaseModel):
    """One song to align: its audio, its lyric lines, and the decoder settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    audio_path: str
    text: str
    """The lyrics, one line per source line."""
    model_path: str
    """The checkpoint snapshot directory."""
    language: str = "auto"
    detect_language: tuple[str, ...] = ()
    """With language "auto": the languages the lyrics may be identified as (lid.176)."""
    label_rules: LabelRules = "v1"
    min_letter_ms: int = 0
    min_token_ms: int = 0
    timing_offset_ms: int = 0
    duration_pace_ratio: float = 0.0
    vocal_activity_weight: float = 0.0
    vocal_activity_source: Literal["energy", "vad", "external"] = "energy"
    vocal_activity_file: str | None = None
    """float32 .npy of per-20 ms singing activity, with ``vocal_activity_source`` external."""
    exact_frame_clock: bool = False
    background_decoding: bool = False
    """Decode lines that are one bracketed span on the checkpoint's background head."""
    blank_free_tokens: bool = False
    inline_backing: bool = False
    """Re-place inline bracketed words on the background head inside their line's window."""
    optional_brackets: bool = False
    """Experiment, off: inline bracketed words may leave the lead path."""
    gap_onset_pull: bool = False
    fusion: Literal["off", "on", "gate0"] = "off"
    audio_speed: float = 1.0
