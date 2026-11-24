from __future__ import annotations

from core.settings import get_settings

import threading
from typing import Iterable

import numpy as np
import torch

_PREDEFINED: tuple[dict[str, object], ...] = (
    {"name": "Dreamy", "group": "mood", "positive_prompts": ["dreamy atmospheric music", "ethereal floating music", "soft hazy production"], "negative_prompts": ["dry aggressive music"]},
    {"name": "Melancholic", "group": "mood", "positive_prompts": ["melancholic reflective music", "sad introspective song"], "negative_prompts": ["joyful celebratory music"]},
    {"name": "Joyful", "group": "mood", "positive_prompts": ["joyful uplifting music", "bright celebratory song"], "negative_prompts": ["ominous melancholic music"]},
    {"name": "Peaceful", "group": "mood", "positive_prompts": ["peaceful calming music", "gentle serene sound"], "negative_prompts": ["tense chaotic music"]},
    {"name": "Tense", "group": "mood", "positive_prompts": ["tense suspenseful music", "anxious uneasy atmosphere"], "negative_prompts": ["peaceful relaxing music"]},
    {"name": "Aggressive", "group": "mood", "positive_prompts": ["aggressive intense music", "harsh forceful sound"], "negative_prompts": ["gentle peaceful music"]},
    {"name": "Energetic", "group": "motion", "positive_prompts": ["high energy music", "fast driving rhythm"], "negative_prompts": ["slow calm music"]},
    {"name": "Danceable", "group": "motion", "positive_prompts": ["danceable music with a strong groove", "club rhythm made for dancing"], "negative_prompts": ["arrhythmic ambient music"]},
    {"name": "Slow-burning", "group": "motion", "positive_prompts": ["slow-building music", "gradual restrained musical development"], "negative_prompts": ["immediate high energy music"]},
    {"name": "Chaotic", "group": "motion", "positive_prompts": ["chaotic unpredictable music", "frantic irregular sound"], "negative_prompts": ["orderly minimal music"]},
    {"name": "Acoustic", "group": "texture", "positive_prompts": ["acoustic instruments and natural room sound", "unplugged acoustic music"], "negative_prompts": ["synthetic electronic production"]},
    {"name": "Electronic", "group": "texture", "positive_prompts": ["electronic music with synthesizers", "synthetic electronic production"], "negative_prompts": ["unplugged acoustic music"]},
    {"name": "Distorted", "group": "texture", "positive_prompts": ["distorted saturated music", "fuzzy overdriven sound"], "negative_prompts": ["clean transparent production"]},
    {"name": "Lo-fi", "group": "texture", "positive_prompts": ["lo-fi music with rough warm production", "grainy homemade recording"], "negative_prompts": ["polished pristine production"]},
    {"name": "Dense", "group": "texture", "positive_prompts": ["dense layered musical arrangement", "thick wall of sound"], "negative_prompts": ["sparse minimal arrangement"]},
    {"name": "Minimal", "group": "texture", "positive_prompts": ["minimal sparse music", "few instruments and lots of space"], "negative_prompts": ["dense layered arrangement"]},
    {"name": "Atmospheric", "group": "texture", "positive_prompts": ["atmospheric spacious music", "immersive ambient soundscape"], "negative_prompts": ["dry direct production"]},
    {"name": "Instrumental", "group": "voice", "positive_prompts": ["instrumental music without vocals"], "negative_prompts": ["song with prominent lead vocals"]},
    {"name": "Female vocals", "group": "voice", "positive_prompts": ["music with female lead vocals", "woman singing lead vocal"], "negative_prompts": ["instrumental music without vocals"]},
    {"name": "Male vocals", "group": "voice", "positive_prompts": ["music with male lead vocals", "man singing lead vocal"], "negative_prompts": ["instrumental music without vocals"]},
    {"name": "Group vocals", "group": "voice", "positive_prompts": ["group vocals and vocal harmonies", "choir or ensemble singing"], "negative_prompts": ["solo instrumental music"]},
    {"name": "Guitar-driven", "group": "instrumentation", "positive_prompts": ["guitar-driven music", "prominent electric or acoustic guitar"], "negative_prompts": ["music dominated by synthesizers"]},
    {"name": "Piano-led", "group": "instrumentation", "positive_prompts": ["piano-led music", "prominent piano performance"], "negative_prompts": ["music without piano"]},
    {"name": "Synth-led", "group": "instrumentation", "positive_prompts": ["synthesizer-led music", "prominent electronic synthesizers"], "negative_prompts": ["unplugged acoustic music"]},
    {"name": "Orchestral", "group": "instrumentation", "positive_prompts": ["orchestral music with strings and ensemble", "symphonic arrangement"], "negative_prompts": ["minimal electronic beat"]},
)

_model = None
_model_lock = threading.Lock()


def predefined_concepts() -> list[dict[str, object]]:
    return [dict(item) for item in _PREDEFINED]


_GROUPS_TO_EXPAND = {"voice", "mood", "texture"}

_WORD_CANON = {"voices": "vocal", "voice": "vocal", "vocals": "vocal", "vocal": "vocal", "singing": "sing", "sung": "sing"}


def _canon_words(value: str) -> set[str]:
    """Reduce tag/concept words so 'female voice' matches 'female vocals'."""
    words = set()
    for word in value.casefold().split():
        word = _WORD_CANON.get(word, word)
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.add(_WORD_CANON.get(word, word))
    return words


def expand_tag_groups(tags: list[str]) -> list[list[str]]:
    """Same expansion rules as expand_prompts, preserving tag boundaries."""
    groups: list[list[str]] = []
    for tag in tags:
        value = tag.strip()
        if not value:
            continue
        words = _canon_words(value)
        match = next((definition for definition in _PREDEFINED
                      if words and words <= _canon_words(definition["name"])), None)
        groups.append(list(match["positive_prompts"]) if match is not None and match["group"] in _GROUPS_TO_EXPAND else [value])
    return groups or [[tag] for tag in tags if tag.strip()]


def expand_prompts(tags: list[str]) -> list[str]:
    """Expand user tags into retrieval phrases, but only when phrasing helps.

    Genre and instrument words (rock, hip hop, guitar) are caption vocabulary
    the audio-text model already understands, and the measured rankings show
    bare words beat paraphrases for them, so they pass through untouched.
    Voice, mood, and texture attributes are relational ("female" only means
    something as "female vocals"), so tags in those groups score against the
    concept's full phrase set.
    """
    return [phrase for group in expand_tag_groups(tags) for phrase in group]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def embed_texts(texts: list[str]) -> np.ndarray:
    global _model
    if not texts:
        raise ValueError("At least one text prompt is required")
    with _model_lock:
        if _model is None:
            from muq import MuQMuLan

            model = MuQMuLan.from_pretrained(
                get_settings().muq_model_id,
                revision=get_settings().muq_revision,
            )
            if torch.cuda.is_available():
                try:
                    model = model.to("cuda")
                except torch.OutOfMemoryError:
                    # Inference workers can temporarily occupy the GPU. Text
                    # queries must remain available instead of returning 500.
                    torch.cuda.empty_cache()
                    model = model.to("cpu")
            _model = model.eval()
        with torch.inference_mode():
            vectors = _model(texts=texts).detach().float().cpu().numpy()
    return _normalize_rows(vectors)


def empirical_percentiles(scores: np.ndarray, available: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    mask = np.ones(len(values), dtype=bool) if available is None else np.asarray(available, dtype=bool)
    mask = mask & np.isfinite(values)
    percentiles = np.full(len(values), 0.5, dtype=np.float32)
    indices = np.flatnonzero(mask)
    if len(indices) < 2:
        return percentiles
    order = indices[np.argsort(values[indices], kind="stable")]
    # Average the ranks of exact ties. A constant corpus has no ranking evidence,
    # and must remain neutral regardless of track IDs or database row order.
    starts = np.r_[0, np.flatnonzero(np.diff(values[order])) + 1]
    ends = np.r_[starts[1:], len(order)]
    ranks = (starts + ends - 1) / (2.0 * (len(order) - 1))
    percentiles[order] = np.repeat(ranks, ends - starts)
    return percentiles


def combine_concept_percentiles(
    semantic: np.ndarray, lyrics: np.ndarray, lyrics_available: np.ndarray,
    semantic_weight: float = 0.45, lyrics_weight: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    semantic_percentiles = empirical_percentiles(semantic)
    lyrics_percentiles = empirical_percentiles(lyrics, lyrics_available)
    combined = semantic_weight * semantic_percentiles + lyrics_weight * lyrics_percentiles
    return combined.astype(np.float32), empirical_percentiles(combined)


def score_concept(
    tracks: np.ndarray,
    positive_prompts: Iterable[str],
    negative_prompts: Iterable[str] = (),
    positive_examples: np.ndarray | None = None,
    negative_examples: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    positives = [value.strip() for value in positive_prompts if value.strip()]
    negatives = [value.strip() for value in negative_prompts if value.strip()]
    positive_parts: list[np.ndarray] = []
    negative_parts: list[np.ndarray] = []
    if positives:
        positive_parts.extend(embed_texts(positives))
    if negatives:
        negative_parts.extend(embed_texts(negatives))
    if positive_examples is not None and len(positive_examples):
        positive_parts.extend(_normalize_rows(positive_examples))
    if negative_examples is not None and len(negative_examples):
        negative_parts.extend(_normalize_rows(negative_examples))
    if not positive_parts:
        raise ValueError("A concept needs a positive prompt or positive example track")
    positive_center = _normalize_rows(np.mean(positive_parts, axis=0, keepdims=True))[0]
    raw = _normalize_rows(tracks) @ positive_center
    if negative_parts:
        negative_center = _normalize_rows(np.mean(negative_parts, axis=0, keepdims=True))[0]
        raw = raw - (_normalize_rows(tracks) @ negative_center)
    return raw.astype(np.float32), empirical_percentiles(raw)
