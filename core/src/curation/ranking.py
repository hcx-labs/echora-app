"""Ranking a curation's candidate tracks from its prompts, examples and sound profile."""

from __future__ import annotations

import random
import re
import unicodedata

import numpy as np

import core.curation.concepts as concepts
from core.curation.concepts import combine_concept_percentiles, empirical_percentiles

# Minimum blended percentile for a track to enter the matching pool. Last.fm
# promotion and the discovery pool both draw from inside this pool; anything
# below it only appears when the pool cannot fill the playlist.
MATCH_PERCENTILE = 0.75
CURATION_SCORING_REVISION = 7
INSTRUMENTAL_ONLY_THRESHOLD = 0.5
EXAMPLE_COMPONENT_WEIGHTS = {
    "muq_global": 0.50,
    "muq_modes": 0.20,
    "mert_global": 0.20,
    "mert_modes": 0.10,
}

ModeProfile = tuple[np.ndarray, np.ndarray]


def _normalized_rows(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)


def mode_transport_similarity(left: ModeProfile, right: ModeProfile) -> float:
    """Duration-conserving entropic transport between two small mode mixtures."""
    left_vectors, left_weights = left
    right_vectors, right_weights = right
    similarity = _normalized_rows(left_vectors) @ _normalized_rows(right_vectors).T
    source = np.asarray(left_weights, dtype=np.float64)
    target = np.asarray(right_weights, dtype=np.float64)
    source /= max(float(source.sum()), 1e-12)
    target /= max(float(target.sum()), 1e-12)
    kernel = np.exp((similarity - float(similarity.max())) / 0.05).astype(np.float64)
    left_scale = np.ones(len(source), dtype=np.float64)
    right_scale = np.ones(len(target), dtype=np.float64)
    for _ in range(100):
        left_scale = source / np.maximum(kernel @ right_scale, 1e-12)
        right_scale = target / np.maximum(kernel.T @ left_scale, 1e-12)
    transport = left_scale[:, None] * kernel * right_scale[None, :]
    transport /= max(float(transport.sum()), 1e-12)
    return float(np.sum(transport * similarity))


def _example_percentiles(
    matrix: np.ndarray,
    positive_indices: list[int],
    negative_indices: list[int],
    modes: list[ModeProfile | None] | None,
    acoustic_matrix: np.ndarray | None,
    acoustic_available: np.ndarray | None,
    acoustic_modes: list[ModeProfile | None] | None,
    evidence_prefix: str = "",
) -> tuple[np.ndarray | None, np.ndarray | None, list[dict[str, float | bool]]]:
    count = len(matrix)
    if not positive_indices and not negative_indices:
        return None, None, [{} for _ in range(count)]
    semantic = _normalized_rows(matrix)
    acoustic = _normalized_rows(acoustic_matrix) if acoustic_matrix is not None else None
    evidence: list[dict[str, float | bool]] = [{} for _ in range(count)]

    def component_scores(references: list[int], prefix: str) -> np.ndarray:
        combined = np.zeros(count, dtype=np.float32)
        available_weight = np.zeros(count, dtype=np.float32)

        muq_global = np.max(semantic @ semantic[references].T, axis=1)
        combined += EXAMPLE_COMPONENT_WEIGHTS["muq_global"] * muq_global
        available_weight += EXAMPLE_COMPONENT_WEIGHTS["muq_global"]
        for index in range(count):
            evidence[index][f"{prefix}_muq_global_similarity"] = float(muq_global[index])

        if modes is not None:
            reference_modes = [modes[index] for index in references if modes[index] is not None]
            for index, profile in enumerate(modes):
                if profile is None or not reference_modes:
                    continue
                value = max(
                    mode_transport_similarity(profile, reference) for reference in reference_modes
                )
                best_passage = max(
                    float(np.max(_normalized_rows(profile[0]) @ _normalized_rows(reference[0]).T))
                    for reference in reference_modes
                )
                combined[index] += EXAMPLE_COMPONENT_WEIGHTS["muq_modes"] * value
                available_weight[index] += EXAMPLE_COMPONENT_WEIGHTS["muq_modes"]
                evidence[index][f"{prefix}_muq_mode_similarity"] = value
                evidence[index][f"{prefix}_muq_best_passage_similarity"] = best_passage

        if acoustic is not None and acoustic_available is not None:
            references_with_acoustic = [index for index in references if acoustic_available[index]]
            if references_with_acoustic:
                values = np.max(acoustic @ acoustic[references_with_acoustic].T, axis=1)
                for index in np.flatnonzero(acoustic_available):
                    combined[index] += EXAMPLE_COMPONENT_WEIGHTS["mert_global"] * values[index]
                    available_weight[index] += EXAMPLE_COMPONENT_WEIGHTS["mert_global"]
                    evidence[index][f"{prefix}_mert_global_similarity"] = float(values[index])

        if acoustic_modes is not None:
            reference_modes = [
                acoustic_modes[index] for index in references if acoustic_modes[index] is not None
            ]
            for index, profile in enumerate(acoustic_modes):
                if profile is None or not reference_modes:
                    continue
                value = max(
                    mode_transport_similarity(profile, reference) for reference in reference_modes
                )
                best_passage = max(
                    float(np.max(_normalized_rows(profile[0]) @ _normalized_rows(reference[0]).T))
                    for reference in reference_modes
                )
                combined[index] += EXAMPLE_COMPONENT_WEIGHTS["mert_modes"] * value
                available_weight[index] += EXAMPLE_COMPONENT_WEIGHTS["mert_modes"]
                evidence[index][f"{prefix}_mert_mode_similarity"] = value
                evidence[index][f"{prefix}_mert_best_passage_similarity"] = best_passage
        return combined / np.maximum(available_weight, 1e-8)

    positive_label = f"{evidence_prefix}positive" if evidence_prefix else "positive"
    negative_label = f"{evidence_prefix}negative" if evidence_prefix else "negative"
    positive = component_scores(positive_indices, positive_label) if positive_indices else None
    negative = component_scores(negative_indices, negative_label) if negative_indices else None
    positive_percentiles = empirical_percentiles(positive) if positive is not None else None
    negative_exclusion = 1.0 - empirical_percentiles(negative) if negative is not None else None
    for index in range(count):
        evidence[index]["profile_enhanced"] = bool(
            any("_muq_mode_" in key or "_mert_" in key for key in evidence[index])
        )
    return positive_percentiles, negative_exclusion, evidence


def _normalize(value: str | None) -> str:
    if not value:
        return ""
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def resolve_references(prompt: str, rows: list[dict[str, object]]) -> list[dict[str, str]]:
    normalized_prompt = f" {_normalize(prompt)} "
    quoted = {_normalize(value) for value in re.findall(r'["“”]([^"“”]+)["“”]', prompt)}
    found: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        for kind, field in (("track", "title"), ("artist", "artist"), ("album", "album")):
            display = str(row.get(field) or "").strip()
            value = _normalize(display)
            if not value:
                continue
            explicit = value in quoted
            mentioned = len(value) >= 4 and f" {value} " in normalized_prompt
            if explicit or mentioned:
                key = (kind, value)
                found[key] = {"kind": kind, "name": display}
    return sorted(found.values(), key=lambda item: (item["kind"], item["name"].casefold()))


def rank_curation(
    rows: list[dict[str, object]],
    matrix: np.ndarray,
    positive_prompt: str,
    negative_prompt: str,
    track_limit: int,
    refresh_mode: str,
    existing_track_ids: list[str] | None = None,
    positive_track_ids: list[str] | None = None,
    negative_track_ids: list[str] | None = None,
    lyrics_matrix: np.ndarray | None = None,
    lyrics_available: np.ndarray | None = None,
    lyrics_positive_queries: np.ndarray | None = None,
    lyrics_negative_queries: np.ndarray | None = None,
    listen_counts: dict[str, int] | None = None,
    recent_track_ids: set[str] | None = None,
    familiarity_percent: int = 70,
    shuffle_seed: int | None = None,
    sound_prompts: list[list[str]] | None = None,
    themes_prompts: list[list[str]] | None = None,
    sound_negative_prompts: list[list[str]] | None = None,
    themes_negative_prompts: list[list[str]] | None = None,
    sound_weight: int | None = None,
    voice_matrix: np.ndarray | None = None,
    voice_available: np.ndarray | None = None,
    semantic_modes: list[ModeProfile | None] | None = None,
    acoustic_matrix: np.ndarray | None = None,
    acoustic_available: np.ndarray | None = None,
    acoustic_modes: list[ModeProfile | None] | None = None,
    context_track_ids: list[str] | None = None,
    eligible_track_ids: set[str] | None = None,
    sound_profile: dict[str, float] | None = None,
    instrumental_only: bool = False,
    minimum_match_percentile: float = MATCH_PERCENTILE,
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, str]]]]:
    """Rank a curation corpus across the sound (semantic) and themes (lyrics) channels.

    When any prompt list is provided the recipe is structured: each sound tag
    scores only the semantic channel and each themes tag only the lyrics
    channel. Tags are scored independently and their percentiles combine with
    a geometric mean, so a track has to satisfy every tag (a soft AND), and
    sound_weight (0-100) sets the semantic share of the blend. When all are
    None the recipe is legacy: the single positive_prompt scores both channels
    at the historical 45/55 split.

    Selection: tracks scoring at or above MATCH_PERCENTILE form the matching
    pool. Listened tracks inside the pool are guaranteed slots (up to the
    familiarity target); remaining slots come from the top-scoring unheard
    tracks in the pool, then from remaining familiar matches. The playlist may
    be shorter than requested. If no listened track clears the threshold, Last.fm is
    ignored and the playlist is all discovery regardless of the slider.
    """
    if not 0 <= minimum_match_percentile <= 1:
        raise ValueError("minimum_match_percentile must be between zero and one")
    # A language-only recipe has no musical requirement. Missing or tied
    # evidence for a requested musical direction is not a language-only recipe.
    musical_requirement = any(
        (
            positive_prompt.strip(),
            negative_prompt.strip(),
            positive_track_ids,
            negative_track_ids,
            context_track_ids,
            sound_prompts,
            themes_prompts,
            sound_negative_prompts,
            themes_negative_prompts,
            lyrics_positive_queries is not None and len(lyrics_positive_queries) > 0,
            lyrics_negative_queries is not None and len(lyrics_negative_queries) > 0,
            sound_profile,
            instrumental_only,
        )
    )
    structured = any(
        value is not None
        for value in (
            sound_prompts,
            themes_prompts,
            sound_negative_prompts,
            themes_negative_prompts,
        )
    )
    positive_references = resolve_references(positive_prompt, rows)
    negative_references = resolve_references(negative_prompt, rows)
    row_by_id = {str(row["id"]): row for row in rows}
    for identifiers, references in (
        (positive_track_ids or [], positive_references),
        (negative_track_ids or [], negative_references),
    ):
        for identifier in identifiers:
            row = row_by_id.get(str(identifier))
            if row is not None and not any(
                item["kind"] == "track" and item["name"] == str(row["title"]) for item in references
            ):
                references.append({"kind": "track", "name": str(row["title"])})

    def indices(references: list[dict[str, str]]) -> list[int]:
        matches: list[int] = []
        for index, row in enumerate(rows):
            for reference in references:
                field = "title" if reference["kind"] == "track" else reference["kind"]
                if _normalize(str(row.get(field) or "")) == _normalize(reference["name"]):
                    matches.append(index)
                    break
        return sorted(set(matches))

    positive_indices = sorted(
        set(
            indices(positive_references)
            + [
                index
                for index, row in enumerate(rows)
                if str(row["id"]) in set(positive_track_ids or [])
            ]
        )
    )
    negative_indices = sorted(
        set(
            indices(negative_references)
            + [
                index
                for index, row in enumerate(rows)
                if str(row["id"]) in set(negative_track_ids or [])
            ]
        )
    )
    context_indices = [
        index for index, row in enumerate(rows) if str(row["id"]) in set(context_track_ids or [])
    ]
    example_positive, example_negative, example_evidence = _example_percentiles(
        matrix,
        positive_indices,
        negative_indices,
        semantic_modes,
        acoustic_matrix,
        acoustic_available,
        acoustic_modes,
    )
    context_positive, _, context_evidence = _example_percentiles(
        matrix,
        context_indices,
        [],
        semantic_modes,
        acoustic_matrix,
        acoustic_available,
        acoustic_modes,
        "time_",
    )
    for index in range(len(example_evidence)):
        enhanced = bool(example_evidence[index].get("profile_enhanced")) or bool(
            context_evidence[index].get("profile_enhanced")
        )
        example_evidence[index].update(
            {
                key: value
                for key, value in context_evidence[index].items()
                if key != "profile_enhanced"
            }
        )
        if example_evidence[index] or context_evidence[index]:
            example_evidence[index]["profile_enhanced"] = enhanced
    tag_evidence: dict[str, list[dict[str, object]]] = {str(row["id"]): [] for row in rows}

    def _geomean_percentiles(per_tag: list[np.ndarray]) -> np.ndarray:
        stacked = np.clip(np.stack(per_tag), 1e-4, 1.0)
        return np.exp(np.mean(np.log(stacked), axis=0)).astype(np.float32)

    def _semantic_prompt_raw(
        positives: list[str],
        negatives: list[str] | None = None,
    ) -> np.ndarray:
        """Score a MuQ text direction, using duration modes where available."""
        positive = np.asarray(concepts.embed_texts(positives), dtype=np.float32).mean(axis=0)
        positive /= max(float(np.linalg.norm(positive)), 1e-8)
        normalized_global = _normalized_rows(matrix)
        raw = normalized_global @ positive
        negative: np.ndarray | None = None
        if negatives:
            negative = np.asarray(concepts.embed_texts(negatives), dtype=np.float32).mean(axis=0)
            negative /= max(float(np.linalg.norm(negative)), 1e-8)
            raw -= normalized_global @ negative
        if semantic_modes is None:
            return np.asarray(raw, dtype=np.float32)
        mode_raw = np.zeros(len(rows), dtype=np.float32)
        mode_available = np.zeros(len(rows), dtype=bool)
        for index, profile in enumerate(semantic_modes):
            if profile is None:
                continue
            vectors, weights = profile
            normalized_weights = weights / max(float(weights.sum()), 1e-8)
            value = (_normalized_rows(vectors) @ positive) @ normalized_weights
            if negative is not None:
                value -= (_normalized_rows(vectors) @ negative) @ normalized_weights
            mode_raw[index] = float(value)
            mode_available[index] = True
        return np.where(mode_available, 0.75 * raw + 0.25 * mode_raw, raw).astype(np.float32)

    if structured:
        negative_groups = sound_negative_prompts or []
        semantic_groups = sound_prompts or []

        def _voice_kind(group: list[str]) -> str | None:
            text = " ".join(group).casefold()
            if "instrumental" in text or "without vocal" in text or "no singing" in text:
                return "instrumental"
            vocal = any(word in text for word in ("vocal", "voice", "sing"))
            if not vocal:
                return None
            if "female" in text or "woman" in text:
                return "female"
            if ("male" in text and "female" not in text) or "man " in text or "man singing" in text:
                return "male"
            return None

        def _voice_percentile(kind: str) -> np.ndarray | None:
            if voice_matrix is None or voice_available is None or not voice_available.any():
                return None
            instrumental, female, male = (
                voice_matrix[:, 0],
                voice_matrix[:, 1],
                voice_matrix[:, 2],
            )
            evidence = {"female": female, "male": male, "instrumental": instrumental}[kind]
            return empirical_percentiles(evidence, voice_available)

        def _semantic_percentile(group: list[str]) -> np.ndarray:
            raw = _semantic_prompt_raw(group)
            values = empirical_percentiles(raw)
            kind = _voice_kind(group)
            voice_percentiles = _voice_percentile(kind) if kind else None
            if voice_percentiles is not None:
                # Direct voice labels use the dedicated classifier. MuQ remains
                # the fallback only for tracks without classifier evidence.
                values = np.where(voice_available, voice_percentiles, values)
            return values

        per_tag_semantic: list[np.ndarray] = [
            _semantic_percentile(group) for group in semantic_groups
        ]
        semantic_negative_count = len(negative_groups)
        for group in negative_groups:
            per_tag_semantic.append(1.0 - _semantic_percentile(group))
        if per_tag_semantic:
            semantic_percentiles = _geomean_percentiles(per_tag_semantic)
        else:
            semantic_percentiles = np.full(len(rows), 0.5, dtype=np.float32)
        lyrics_percentiles = np.full(len(rows), 0.5, dtype=np.float32)
        per_tag_lyrics: list[np.ndarray] = []
        lyrics_negative_count = 0
        if lyrics_matrix is not None and lyrics_available is not None:
            normalized_lyrics = lyrics_matrix / np.maximum(
                np.linalg.norm(lyrics_matrix, axis=1, keepdims=True), 1e-8
            )

            def _lyrics_percentile(center: np.ndarray) -> np.ndarray:
                value = np.asarray(center, dtype=np.float32)
                value /= max(float(np.linalg.norm(value)), 1e-8)
                return np.asarray(
                    empirical_percentiles(normalized_lyrics @ value, lyrics_available),
                    dtype=np.float32,
                )

            positive_centers = (
                np.asarray(lyrics_positive_queries) if lyrics_positive_queries is not None else []
            )
            for center in positive_centers:
                per_tag_lyrics.append(_lyrics_percentile(center))
            negative_centers = (
                np.asarray(lyrics_negative_queries) if lyrics_negative_queries is not None else []
            )
            for center in negative_centers:
                per_tag_lyrics.append(1.0 - _lyrics_percentile(center))
                lyrics_negative_count += 1
            if per_tag_lyrics:
                lyrics_percentiles = _geomean_percentiles(per_tag_lyrics)
        channels: list[tuple[np.ndarray, float]] = []
        if per_tag_semantic and per_tag_lyrics:
            channels.extend(
                [
                    (
                        semantic_percentiles,
                        (sound_weight if sound_weight is not None else 50) / 100.0,
                    ),
                    (
                        lyrics_percentiles,
                        1.0 - (sound_weight if sound_weight is not None else 50) / 100.0,
                    ),
                ]
            )
        elif per_tag_semantic:
            channels.append((semantic_percentiles, 1.0))
        elif per_tag_lyrics:
            channels.append((lyrics_percentiles, 1.0))
        explicit_example_parts = [
            value for value in (example_positive, example_negative) if value is not None
        ]
        if explicit_example_parts:
            channels.append((_geomean_percentiles(explicit_example_parts), 1.0))
        if context_positive is not None:
            channels.append((context_positive, 1.0))
        active_weight = sum(weight for _, weight in channels)
        if active_weight > 0:
            score = sum(values * weight for values, weight in channels) / active_weight
            percentiles = np.asarray(score, dtype=np.float32)
        else:
            score = percentiles = np.full(len(rows), 0.5, dtype=np.float32)
        semantic_positive_count = len(per_tag_semantic) - semantic_negative_count
        for tag_index, tag_percentiles in enumerate(per_tag_semantic):
            for index, row in enumerate(rows):
                tag_evidence[str(row["id"])].append(
                    {
                        "channel": "semantic",
                        "tag": tag_index + 1,
                        "negative": tag_index >= semantic_positive_count,
                        "percentile": float(tag_percentiles[index]),
                    }
                )
        lyrics_positive_count = len(per_tag_lyrics) - lyrics_negative_count
        for tag_index, tag_percentiles in enumerate(per_tag_lyrics):
            for index, row in enumerate(rows):
                tag_evidence[str(row["id"])].append(
                    {
                        "channel": "lyrics",
                        "tag": tag_index + 1,
                        "negative": tag_index >= lyrics_positive_count,
                        "percentile": float(tag_percentiles[index]),
                    }
                )
    else:
        semantic_positives = [positive_prompt] if positive_prompt.strip() else []
        semantic_negatives = [negative_prompt] if negative_prompt.strip() else []
        semantic_active = (
            bool(semantic_positives) or bool(positive_indices) or bool(context_indices)
        )
        if semantic_positives:
            semantic_raw = _semantic_prompt_raw(semantic_positives, semantic_negatives)
            percentiles = empirical_percentiles(semantic_raw)
        else:
            semantic_raw = np.full(len(rows), 0.5, dtype=np.float32)
            percentiles = np.full(len(rows), 0.5, dtype=np.float32)
        score = semantic_raw
        semantic_percentiles = percentiles.copy()
        lyrics_percentiles = np.full(len(rows), 0.5, dtype=np.float32)
        all_positive_indices = sorted(set(positive_indices + context_indices))
        has_lyrics_examples = (
            any(lyrics_available[index] for index in all_positive_indices)
            if lyrics_available is not None
            else False
        )
        if (
            lyrics_matrix is not None
            and lyrics_available is not None
            and (lyrics_positive_queries is not None or has_lyrics_examples)
        ):
            positive_parts = (
                list(lyrics_positive_queries) if lyrics_positive_queries is not None else []
            )
            positive_parts.extend(
                lyrics_matrix[index] for index in all_positive_indices if lyrics_available[index]
            )
            positive_center = np.mean(positive_parts, axis=0)
            positive_center /= max(float(np.linalg.norm(positive_center)), 1e-8)
            normalized_lyrics = lyrics_matrix / np.maximum(
                np.linalg.norm(lyrics_matrix, axis=1, keepdims=True), 1e-8
            )
            lyrics_raw = normalized_lyrics @ positive_center
            negative_parts = (
                list(lyrics_negative_queries) if lyrics_negative_queries is not None else []
            )
            negative_parts.extend(
                lyrics_matrix[index] for index in negative_indices if lyrics_available[index]
            )
            if negative_parts:
                negative_center = np.mean(negative_parts, axis=0)
                negative_center /= max(float(np.linalg.norm(negative_center)), 1e-8)
                lyrics_raw -= normalized_lyrics @ negative_center
            lyrics_percentiles = empirical_percentiles(lyrics_raw, lyrics_available)
            if semantic_active:
                score, percentiles = combine_concept_percentiles(
                    semantic_raw,
                    lyrics_raw,
                    lyrics_available,
                    0.45,
                    0.55,
                )
            else:
                score, percentiles = lyrics_raw, lyrics_percentiles
        example_parts = [
            value
            for value in (example_positive, context_positive, example_negative)
            if value is not None
        ]
        if example_parts:
            semantic_parts = (
                [semantic_percentiles] if semantic_active and semantic_positives else []
            )
            semantic_parts.extend(example_parts)
            semantic_percentiles = _geomean_percentiles(semantic_parts)
            if (
                lyrics_matrix is not None
                and lyrics_available is not None
                and (lyrics_positive_queries is not None or has_lyrics_examples)
            ):
                score = 0.45 * semantic_percentiles + 0.55 * lyrics_percentiles
            else:
                score = semantic_percentiles
            percentiles = np.asarray(score, dtype=np.float32)
    sound_profile = sound_profile or {}
    profile_evidence: list[dict[str, object]] = [{} for _ in rows]
    if sound_profile:
        profile_parts: list[np.ndarray] = []
        for axis, target in sound_profile.items():
            raw = np.asarray(
                [row.get("sound_descriptors", {}).get(axis, np.nan) for row in rows],
                dtype=np.float32,
            )
            available = np.isfinite(raw)
            if available.sum() < 2:
                continue
            # Every axis is percentile-ranked within this library. The target
            # is therefore meaningful for a collection with unusual mastering
            # or a narrow tempo range, while raw units stay in the evidence.
            actual = empirical_percentiles(raw, available)
            if axis == "vocals":
                # One vertex controls a continuous vocal-to-instrumental
                # preference. Instrumental is the inverse of this same vocal
                # signal, so it must not become a second, double-counted axis.
                vocal_weight = float(target)
                instrumental_weight = 1.0 - vocal_weight
                closeness = vocal_weight * actual + instrumental_weight * (1.0 - actual)
            else:
                closeness = 1.0 - np.abs(actual - float(target))
            profile_parts.append(np.where(available, closeness, np.nan))
            for index in np.flatnonzero(available):
                evidence = {
                    "target": float(target),
                    "actual_percentile": float(actual[index]),
                    "value": float(raw[index]),
                    "closeness": float(closeness[index]),
                }
                if axis == "vocals":
                    evidence.update(
                        {
                            "vocal_weight": vocal_weight,
                            "instrumental_weight": instrumental_weight,
                            "instrumental_percentile": float(1.0 - actual[index]),
                        }
                    )
                profile_evidence[index][axis] = evidence
        if profile_parts:
            profile_values = np.asarray(profile_parts, dtype=np.float32)
            profile_score = np.nanmean(profile_values, axis=0)
            profile_available = np.isfinite(profile_score)
            # An unavailable descriptor removes this signal for that track;
            # it is not evidence that the track is a poor match.
            score = np.where(profile_available, (percentiles + profile_score) / 2, percentiles)
            percentiles = np.asarray(score, dtype=np.float32)

    adjusted = score.copy()
    existing = set(existing_track_ids or [])
    if refresh_mode == "stable" and existing:
        for index, row in enumerate(rows):
            if str(row["id"]) in existing:
                adjusted[index] += 0.035

    counts = listen_counts or {}
    eligible = eligible_track_ids
    if instrumental_only:
        # This is deliberately a hard classifier filter, unlike the vocals
        # profile axis, which only softly ranks vocal activity relative to the
        # user's library. Unknown classifications do not qualify.
        classified_instrumentals = {
            str(rows[index]["id"])
            for index in range(len(rows))
            if voice_matrix is not None
            and voice_available is not None
            and bool(voice_available[index])
            and float(voice_matrix[index, 0]) >= INSTRUMENTAL_ONLY_THRESHOLD
        }
        eligible = (
            classified_instrumentals if eligible is None else eligible & classified_instrumentals
        )
    # Embedding score filters first; Last.fm only promotes within the high scorers.
    matching_order = [
        int(index)
        for index in np.argsort(adjusted, kind="stable")[::-1]
        if (not musical_requirement or float(percentiles[index]) >= minimum_match_percentile)
        and (eligible is None or str(rows[index]["id"]) in eligible)
    ]
    familiar_matching = [
        index for index in matching_order if counts.get(str(rows[index]["id"]), 0) > 0
    ]
    familiar_set = set(familiar_matching)
    new_order = [index for index in matching_order if index not in familiar_set]
    # The slider guides how many familiar slots to aim for, but familiar tracks
    # only come from the matching pool. If none of them cleared the threshold,
    # Last.fm is irrelevant and the whole playlist comes from new tracks.
    familiar_target = (
        min(round(track_limit * familiarity_percent / 100), len(familiar_matching))
        if listen_counts is not None
        else 0
    )
    selected_indices: list[int] = []
    artist_counts: dict[str, int] = {}
    used_recording_groups: set[str] = set()

    def take(order: list[int], amount: int) -> None:
        for index in order:
            if len(selected_indices) >= amount or index in selected_indices:
                continue
            recording_group = str(rows[index].get("recording_group_id") or "")
            if recording_group and recording_group in used_recording_groups:
                continue
            artist_key = _normalize(str(rows[index].get("artist") or "Unknown artist"))
            if artist_counts.get(artist_key, 0) >= 2:
                continue
            artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
            if recording_group:
                used_recording_groups.add(recording_group)
            selected_indices.append(index)

    take(familiar_matching, familiar_target)
    take(new_order, track_limit)
    # Relax the familiarity target, never the musical requirement.
    take(matching_order, track_limit)
    selected: list[dict[str, object]] = []
    for index in selected_indices:
        row = rows[index]
        count = counts.get(str(row["id"]), 0)
        familiar = count > 0
        selected.append(
            {
                **row,
                "score": float(score[index]),
                "percentile": float(percentiles[index]),
                "retained": str(row["id"]) in existing,
                "evidence": {
                    "match_basis": "library_relative"
                    if musical_requirement
                    else "no_musical_requirement",
                    "match_confidence": None,
                    "representation_runs": row.get("representation_runs", {}),
                    "minimum_match_percentile": minimum_match_percentile
                    if musical_requirement
                    else None,
                    "semantic_percentile": float(semantic_percentiles[index]),
                    "lyrics_percentile": float(lyrics_percentiles[index]),
                    "lyrics_available": bool(lyrics_available[index])
                    if lyrics_available is not None
                    else False,
                    "listen_count": count,
                    "selection_pool": "familiar" if familiar else "discovery",
                    "tag_percentiles": tag_evidence.get(str(row["id"]), []),
                    "sound_profile": profile_evidence[index],
                    "instrumental_only": (
                        {
                            "threshold": INSTRUMENTAL_ONLY_THRESHOLD,
                            "confidence": float(voice_matrix[index, 0]),
                        }
                        if instrumental_only and voice_matrix is not None
                        else None
                    ),
                    "example_similarity": example_evidence[index],
                },
            }
        )
    random.Random(shuffle_seed).shuffle(selected)
    return selected, {"positive": positive_references, "negative": negative_references}
