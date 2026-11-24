"""Karaoke alignment configuration: the Echora Kara checkpoint and its decoder."""

from __future__ import annotations

from typing import Literal

from core.config import EchoraConfig, compute, setting

LabelRules = Literal["v1", "v2", "v2-orthographic", "v3", "v3-orthographic", "v3-sung"]


class KaraConfig(EchoraConfig, env_prefix="ECHORA_KARA_"):
    """The aligner model and decoder. The defaults are what the model was evaluated with.

    Echora Kara 700M MuQ (github.com/hcx-labs/echora-kara) is trained on full mixes with MuQ fusion,
    a backing-vocal head and a text head; it needs the frozen MuQ-large encoder (``fusion="on"``).
    Change decoder values together, and only with an evaluation.
    """

    model_id: str = compute("hcxlabs/echora-kara-700m-muq", legacy="FA_KARA_MODEL_ID")
    revision: str = compute("7c0cda150c8be886ed5ff2f903460f112cf4c7a2", legacy="FA_KARA_REVISION")
    model_path: str | None = compute(
        None,
        legacy="FA_KARA_MODEL_PATH",
        description="A local checkpoint directory; overrides the model ID snapshot.",
    )
    audio_speed: float = compute(1.0, ge=0.5, le=1.5, legacy="FA_KARA_AUDIO_SPEED")
    aligner: Literal["yohane", "mms"] = compute("yohane", legacy="FA_KARA_ALIGNER")
    min_letter_ms: int = compute(
        40,
        ge=0,
        legacy="FA_KARA_MIN_LETTER_MS",
        description="Minimum letter duration on the audio-only path (0: off).",
    )
    min_token_ms: int = compute(100, ge=0, legacy="FA_KARA_MIN_TOKEN_MS")
    timing_offset_ms: int = compute(
        0,
        ge=-500,
        le=500,
        legacy="FA_KARA_TIMING_OFFSET_MS",
        description="Shift of every time; 0 puts words on their sung onset.",
    )
    duration_pace_ratio: float = compute(
        0.9,
        ge=0.0,
        le=5.0,
        legacy="FA_KARA_DURATION_PACE_RATIO",
        description="Minimum durations follow each line's pace times this (0: fixed floor).",
    )
    vocal_activity_weight: float = compute(
        2.0,
        ge=0.0,
        le=20.0,
        legacy="FA_KARA_VOCAL_ACTIVITY_WEIGHT",
        description="Extra cost per letter frame where the vocals are quiet (0: off).",
    )
    fusion: Literal["on", "gate0", "off"] = compute(
        "on",
        legacy="FA_KARA_FUSION",
        description="on: load the MuQ branch; gate0: run without it (ablation); off: no fusion.",
    )
    label_rules: LabelRules = compute(
        "v3-sung",
        legacy="FA_KARA_LABEL_RULES",
        description="Label rules the model was trained with (vendor/fa_kara/label_rules.py).",
    )
    exact_frame_clock: bool = compute(True, legacy="FA_KARA_EXACT_FRAME_CLOCK")
    blank_free_tokens: bool = compute(True, legacy="FA_KARA_BLANK_FREE_TOKENS")
    background_decoding: bool = compute(
        True,
        legacy="FA_KARA_BACKGROUND_DECODING",
        description="Decode lines written in brackets on the backing-vocal head.",
    )
    text_bonus: float = compute(
        0.4,
        ge=0.0,
        le=5.0,
        legacy="FA_KARA_TEXT_BONUS",
        description="Weight of the text head's evidence (0: off, head not loaded).",
    )
    in_line_gap_cost: float = compute(0.2, ge=0.0, le=20.0, legacy="FA_KARA_IN_LINE_GAP_COST")
    audio_input: Literal["mix", "vocals"] = compute(
        "mix",
        legacy="FA_KARA_AUDIO_INPUT",
        description="What the aligner hears: the song as stored, or the separated vocals.",
    )
    gap_onset_pull: bool = compute(
        False,
        legacy="FA_KARA_GAP_ONSET_PULL",
        description="Pull late words after silence to the first vocal energy (vocal stem only).",
    )
    inline_backing: bool = compute(
        True,
        legacy="FA_KARA_INLINE_BACKING",
        description="Place inline bracketed words on the backing-vocal head.",
    )
    job_timeout_seconds: float = setting(1800, gt=0, legacy="ECHORA_KARAOKE_JOB_TIMEOUT_SECONDS")
    shutdown_grace_seconds: float = setting(
        10, gt=0, legacy="ECHORA_KARAOKE_SHUTDOWN_GRACE_SECONDS"
    )
    kill_grace_seconds: float = setting(5, gt=0, legacy="ECHORA_KARAOKE_KILL_GRACE_SECONDS")
    lock_poll_seconds: float = setting(0.25, gt=0, legacy="ECHORA_KARAOKE_LOCK_POLL_SECONDS")
