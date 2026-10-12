import numpy as np
import pytest

import echora_analysis.concepts as concepts
from echora_analysis.curations import mode_transport_similarity, rank_curation


@pytest.fixture()
def fake_text_embeddings(monkeypatch):
    """Replace the MuQ text encoder with a deterministic lookup table."""
    table = {
        "guitars and piano": [1.0, 0.0],
        "distorted guitars": [1.0, 0.0],
        "guitars": [1.0, 0.0],
        "distortion": [0.9, 0.1],
        "warm acoustic": [0.7, 0.7],
        "pop": [1.0, 0.0],
        "japanese": [0.0, 1.0],
    }

    def embed(texts):
        return np.asarray([table[text] for text in texts], dtype=np.float32)

    monkeypatch.setattr(concepts, "embed_texts", embed)
    return table


def test_structured_sound_prompt_never_touches_the_lyrics_channel(fake_text_embeddings):
    rows = [
        {"id": "guitar-no-lyrics", "title": "Riff", "artist": "A", "recording_group_id": None},
        {"id": "on-theme-lyrics", "title": "Ballad", "artist": "B", "recording_group_id": None},
    ]
    # The guitar instrumental is semantically closer to the sound prompt, while
    # the vocal track wins on lyrics. Sound-only ranking must pick the guitar.
    matrix = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    lyrics_matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    lyrics_available = np.asarray([False, True])
    themes_query = np.asarray([[1.0, 0.0]], dtype=np.float32)

    selected, _ = rank_curation(
        rows,
        matrix,
        "",
        "",
        track_limit=2,
        refresh_mode="fresh",
        lyrics_matrix=lyrics_matrix,
        lyrics_available=lyrics_available,
        shuffle_seed=1,
        sound_prompts=[["guitars and piano"]],
        themes_prompts=[],
        sound_weight=50,
    )
    top = max(selected, key=lambda row: row["score"])
    assert top["id"] == "guitar-no-lyrics"

    # One available lyrics vector cannot establish relative relevance. Do not
    # promote it to a perfect match just because the other track lacks lyrics.
    selected, evidence = rank_curation(
        rows,
        matrix,
        "",
        "",
        track_limit=2,
        refresh_mode="fresh",
        lyrics_matrix=lyrics_matrix,
        lyrics_available=lyrics_available,
        lyrics_positive_queries=themes_query,
        shuffle_seed=1,
        sound_prompts=[],
        themes_prompts=["heartbreak and love"],
        sound_weight=50,
    )
    assert selected == []


def test_structured_weight_shifts_the_blend(fake_text_embeddings):
    rows = [
        {"id": "sound-track", "title": "Riff", "artist": "A", "recording_group_id": None},
        {"id": "theme-track", "title": "Ballad", "artist": "B", "recording_group_id": None},
    ]
    matrix = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    lyrics_matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    lyrics_available = np.asarray([True, True])
    themes_query = np.asarray([[1.0, 0.0]], dtype=np.float32)

    def winner(weight: int) -> str:
        selected, _ = rank_curation(
            rows,
            matrix,
            "",
            "",
            track_limit=1,
            refresh_mode="fresh",
            lyrics_matrix=lyrics_matrix,
            lyrics_available=lyrics_available,
            lyrics_positive_queries=themes_query,
            shuffle_seed=1,
            sound_prompts=[["distorted guitars"]],
            themes_prompts=[["heartbreak"]],
            sound_weight=weight,
        )
        return max(selected, key=lambda row: row["score"])["id"]

    assert winner(100) == "sound-track"
    assert winner(0) == "theme-track"


def test_negative_tags_only_affect_their_own_channel(fake_text_embeddings):
    rows = [
        {"id": "clean-guitar", "title": "Riff", "artist": "A", "recording_group_id": None},
        {"id": "distorted-guitar", "title": "Wall", "artist": "B", "recording_group_id": None},
    ]
    # Both tracks look identical to the positive sound tag; only the negative
    # sound tag separates them. Themes tags are absent, so the lyrics channel
    # stays inactive and cannot interfere.
    matrix = np.asarray([[1.0, 0.0], [0.95, 0.05]], dtype=np.float32)
    lyrics_matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    lyrics_available = np.asarray([False, False])

    selected, _ = rank_curation(
        rows,
        matrix,
        "",
        "",
        track_limit=1,
        refresh_mode="fresh",
        lyrics_matrix=lyrics_matrix,
        lyrics_available=lyrics_available,
        shuffle_seed=1,
        sound_prompts=[["guitars"]],
        sound_negative_prompts=[["distortion"]],
        sound_weight=50,
    )
    top = max(selected, key=lambda row: row["score"])
    assert top["id"] == "clean-guitar"
    assert all(row["evidence"]["lyrics_percentile"] == 0.5 for row in selected)


def test_negative_sound_tag_is_an_independent_soft_exclusion(fake_text_embeddings):
    rows = [
        {"id": "japanese", "title": "Japanese", "artist": "A", "recording_group_id": None},
        {"id": "english", "title": "English", "artist": "B", "recording_group_id": None},
        {"id": "other", "title": "Other", "artist": "C", "recording_group_id": None},
    ]
    matrix = np.asarray([[0.9, 0.8], [0.8, 0.05], [0.1, 0.2]], dtype=np.float32)

    selected, _ = rank_curation(
        rows,
        matrix,
        "",
        "",
        track_limit=3,
        refresh_mode="fresh",
        shuffle_seed=1,
        sound_prompts=[["pop"]],
        sound_negative_prompts=[["japanese"]],
        sound_weight=100,
    )

    top = max(selected, key=lambda row: row["score"])
    assert top["id"] == "english"


def test_negative_theme_query_uses_embedded_vectors_as_soft_exclusions(fake_text_embeddings):
    rows = [
        {"id": "japanese", "title": "Japanese", "artist": "A", "recording_group_id": None},
        {"id": "english", "title": "English", "artist": "B", "recording_group_id": None},
        {"id": "other", "title": "Other", "artist": "C", "recording_group_id": None},
    ]
    matrix = np.eye(3, dtype=np.float32)
    lyrics = np.asarray([[0.8, 0.9], [0.9, 0.05], [0.1, 0.2]], dtype=np.float32)

    selected, _ = rank_curation(
        rows,
        matrix,
        "",
        "",
        track_limit=3,
        refresh_mode="fresh",
        shuffle_seed=1,
        lyrics_matrix=lyrics,
        lyrics_available=np.ones(3, dtype=bool),
        lyrics_positive_queries=np.asarray([[1.0, 0.0]], dtype=np.float32),
        lyrics_negative_queries=np.asarray([[0.0, 1.0]], dtype=np.float32),
        sound_prompts=[],
        themes_prompts=[["pop"]],
        themes_negative_prompts=[["japanese"]],
        sound_weight=0,
    )

    top = max(selected, key=lambda row: row["score"])
    assert top["id"] == "english"


def test_legacy_prompt_keeps_the_historical_blend(fake_text_embeddings):
    rows = [
        {"id": "a", "title": "A", "artist": "A", "recording_group_id": None},
        {"id": "b", "title": "B", "artist": "B", "recording_group_id": None},
    ]
    matrix = np.eye(2, dtype=np.float32)
    legacy, _ = rank_curation(
        rows,
        matrix,
        "warm acoustic",
        "",
        track_limit=2,
        refresh_mode="fresh",
        shuffle_seed=1,
    )
    structured, _ = rank_curation(
        rows,
        matrix,
        "",
        "",
        track_limit=2,
        refresh_mode="fresh",
        shuffle_seed=1,
        sound_prompts=[["warm acoustic"]],
        themes_prompts=[["warm acoustic"]],
        sound_weight=45,
    )
    assert [row["id"] for row in legacy] == [row["id"] for row in structured]
    assert [row["score"] for row in legacy] == [row["score"] for row in structured]


def test_curation_selects_only_one_track_per_recording_group():
    rows = [
        {"id": "reference", "title": "Reference", "artist": "Artist A", "recording_group_id": None},
        {
            "id": "duplicate-high",
            "title": "Duplicate",
            "artist": "Artist B",
            "recording_group_id": "same-recording",
        },
        {
            "id": "duplicate-low",
            "title": "Duplicate reissue",
            "artist": "Artist C",
            "recording_group_id": "same-recording",
        },
        {"id": "other", "title": "Other", "artist": "Artist D", "recording_group_id": None},
    ]
    matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )

    # Inspect all component scores here; production cutoff is tested separately.
    selected, _ = rank_curation(
        rows,
        matrix,
        "",
        "",
        track_limit=4,
        refresh_mode="fresh",
        positive_track_ids=["reference"],
        shuffle_seed=1,
        minimum_match_percentile=0,
    )

    selected_groups = [row.get("recording_group_id") for row in selected]
    assert selected_groups.count("same-recording") == 1
    assert len(selected) == 3


def test_mode_transport_conserves_duration_weights():
    left = (
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([0.9, 0.1], dtype=np.float32),
    )
    mostly_matching = (
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([0.8, 0.2], dtype=np.float32),
    )
    brief_matching = (
        np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
        np.asarray([0.1, 0.9], dtype=np.float32),
    )

    assert mode_transport_similarity(left, mostly_matching) > mode_transport_similarity(
        left,
        brief_matching,
    )


def test_song_examples_use_mert_when_both_tracks_have_it():
    rows = [
        {"id": "reference", "title": "Reference", "artist": "A", "recording_group_id": None},
        {"id": "acoustic-match", "title": "Match", "artist": "B", "recording_group_id": None},
        {"id": "acoustic-miss", "title": "Miss", "artist": "C", "recording_group_id": None},
    ]
    # MuQ cannot distinguish the two candidates; MERT can.
    semantic = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.8, 0.2]], dtype=np.float32)
    acoustic = np.asarray([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]], dtype=np.float32)

    # Inspect all component scores here; production cutoff is tested separately.
    selected, _ = rank_curation(
        rows,
        semantic,
        "",
        "",
        track_limit=3,
        refresh_mode="fresh",
        positive_track_ids=["reference"],
        shuffle_seed=1,
        acoustic_matrix=acoustic,
        acoustic_available=np.ones(3, dtype=bool),
        minimum_match_percentile=0,
    )
    by_id = {row["id"]: row for row in selected}

    assert by_id["acoustic-match"]["score"] > by_id["acoustic-miss"]["score"]
    assert (
        "positive_mert_global_similarity"
        in by_id["acoustic-match"]["evidence"]["example_similarity"]
    )


def test_song_examples_fall_back_to_muq_when_the_reference_has_no_mert():
    rows = [
        {"id": "reference", "title": "Reference", "artist": "A", "recording_group_id": None},
        {"id": "semantic-match", "title": "Match", "artist": "B", "recording_group_id": None},
        {"id": "semantic-miss", "title": "Miss", "artist": "C", "recording_group_id": None},
    ]
    semantic = np.asarray([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]], dtype=np.float32)
    acoustic = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    # Inspect all component scores here; production cutoff is tested separately.
    selected, _ = rank_curation(
        rows,
        semantic,
        "",
        "",
        track_limit=3,
        refresh_mode="fresh",
        positive_track_ids=["reference"],
        shuffle_seed=1,
        acoustic_matrix=acoustic,
        acoustic_available=np.asarray([False, True, True]),
        minimum_match_percentile=0,
    )
    by_id = {row["id"]: row for row in selected}

    assert by_id["semantic-match"]["score"] > by_id["semantic-miss"]["score"]
    assert (
        "positive_mert_global_similarity"
        not in by_id["semantic-match"]["evidence"]["example_similarity"]
    )


def test_no_musical_direction_is_explicitly_neutral():
    rows = [
        {"id": "a", "title": "A", "artist": "A", "recording_group_id": None},
        {"id": "b", "title": "B", "artist": "B", "recording_group_id": None},
    ]

    selected, _ = rank_curation(
        rows,
        np.eye(2, dtype=np.float32),
        "",
        "",
        track_limit=2,
        refresh_mode="fresh",
        shuffle_seed=1,
    )

    assert {track["score"] for track in selected} == {0.5}
    assert {track["percentile"] for track in selected} == {0.5}


def test_free_form_sound_direction_uses_duration_weighted_muq_modes(fake_text_embeddings):
    rows = [
        {"id": "mode-match", "title": "Match", "artist": "A", "recording_group_id": None},
        {"id": "mode-miss", "title": "Miss", "artist": "B", "recording_group_id": None},
    ]
    # Global vectors tie; their duration-weighted local identities separate them.
    global_vectors = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    modes = [
        (np.asarray([[1.0, 0.0]], dtype=np.float32), np.asarray([1.0], dtype=np.float32)),
        (np.asarray([[0.0, 1.0]], dtype=np.float32), np.asarray([1.0], dtype=np.float32)),
    ]

    # Inspect all component scores here; production cutoff is tested separately.
    selected, _ = rank_curation(
        rows,
        global_vectors,
        "pop",
        "",
        track_limit=2,
        refresh_mode="fresh",
        shuffle_seed=1,
        semantic_modes=modes,
        minimum_match_percentile=0,
    )
    by_id = {track["id"]: track for track in selected}

    assert by_id["mode-match"]["score"] > by_id["mode-miss"]["score"]


def test_manual_and_time_examples_are_independent_active_signals():
    rows = [
        {"id": "manual", "title": "Manual", "artist": "A", "recording_group_id": None},
        {"id": "time", "title": "Time", "artist": "B", "recording_group_id": None},
        {"id": "balanced", "title": "Balanced", "artist": "C", "recording_group_id": None},
        {"id": "manual-only", "title": "Manual only", "artist": "D", "recording_group_id": None},
        {"id": "time-only", "title": "Time only", "artist": "E", "recording_group_id": None},
    ]
    matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
            [0.99, 0.1],
            [0.1, 0.99],
        ],
        dtype=np.float32,
    )

    # Inspect all component scores here; production cutoff is tested separately.
    selected, _ = rank_curation(
        rows,
        matrix,
        "",
        "",
        track_limit=5,
        refresh_mode="fresh",
        positive_track_ids=["manual"],
        context_track_ids=["time"],
        shuffle_seed=1,
        minimum_match_percentile=0,
    )
    by_id = {track["id"]: track for track in selected}

    assert by_id["balanced"]["score"] > by_id["manual-only"]["score"]
    assert by_id["balanced"]["score"] > by_id["time-only"]["score"]
    assert (
        "time_positive_muq_global_similarity" in by_id["balanced"]["evidence"]["example_similarity"]
    )


def test_ineligible_reference_can_guide_an_eligible_language_pool():
    rows = [
        {
            "id": "outside-reference",
            "title": "Reference",
            "artist": "A",
            "recording_group_id": None,
        },
        {"id": "eligible-match", "title": "Match", "artist": "B", "recording_group_id": None},
        {"id": "eligible-miss", "title": "Miss", "artist": "C", "recording_group_id": None},
    ]
    matrix = np.asarray([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]], dtype=np.float32)

    # Inspect all component scores here; production cutoff is tested separately.
    selected, _ = rank_curation(
        rows,
        matrix,
        "",
        "",
        track_limit=2,
        refresh_mode="fresh",
        positive_track_ids=["outside-reference"],
        eligible_track_ids={"eligible-match", "eligible-miss"},
        shuffle_seed=1,
        minimum_match_percentile=0,
    )

    assert {track["id"] for track in selected} == {"eligible-match", "eligible-miss"}
    by_id = {track["id"]: track for track in selected}
    assert by_id["eligible-match"]["score"] > by_id["eligible-miss"]["score"]


def test_playlist_stays_short_instead_of_filling_below_threshold(fake_text_embeddings):
    rows = [{"id": str(i), "title": str(i), "artist": str(i)} for i in range(5)]
    matrix = np.asarray([[1, 0], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0, 1]], dtype=np.float32)
    selected, _ = rank_curation(
        rows,
        matrix,
        "pop",
        "",
        track_limit=5,
        refresh_mode="stable",
        existing_track_ids=["4"],
        shuffle_seed=1,
    )
    assert {track["id"] for track in selected} == {"0", "1"}
    assert all(track["percentile"] >= 0.75 for track in selected)
    assert all(track["evidence"]["match_confidence"] is None for track in selected)


def test_tied_requested_direction_is_not_a_neutral_language_recipe(fake_text_embeddings):
    rows = [{"id": str(i), "title": str(i), "artist": str(i)} for i in range(4)]
    matrix = np.ones((4, 2), dtype=np.float32)
    selected, _ = rank_curation(rows, matrix, "pop", "", 4, "fresh")
    assert selected == []
    neutral, _ = rank_curation(rows, matrix, "", "", 4, "fresh", eligible_track_ids={"1", "2"})
    assert {track["id"] for track in neutral} == {"1", "2"}


def test_remaining_familiar_matches_are_labeled_as_familiar(fake_text_embeddings):
    rows = [{"id": str(i), "title": str(i), "artist": str(i)} for i in range(5)]
    matrix = np.asarray([[1, 0], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0, 1]], dtype=np.float32)
    selected, _ = rank_curation(
        rows,
        matrix,
        "pop",
        "",
        5,
        "fresh",
        familiarity_percent=0,
        listen_counts={"0": 2, "1": 1},
    )
    assert len(selected) == 2
    assert {track["evidence"]["selection_pool"] for track in selected} == {"familiar"}


def test_sound_profile_uses_library_relative_measured_descriptors():
    rows = [
        {
            "id": "slow",
            "title": "Slow",
            "artist": "A",
            "sound_descriptors": {"pace": 80.0, "energy": -22.0},
        },
        {
            "id": "target",
            "title": "Target",
            "artist": "B",
            "sound_descriptors": {"pace": 120.0, "energy": -14.0},
        },
        {
            "id": "fast",
            "title": "Fast",
            "artist": "C",
            "sound_descriptors": {"pace": 150.0, "energy": -7.0},
        },
    ]
    selected, _ = rank_curation(
        rows,
        np.eye(3, dtype=np.float32),
        "",
        "",
        track_limit=3,
        refresh_mode="fresh",
        sound_profile={"pace": 0.5, "energy": 0.5},
        shuffle_seed=1,
        minimum_match_percentile=0,
    )
    by_id = {track["id"]: track for track in selected}
    assert by_id["target"]["score"] > by_id["slow"]["score"]
    assert by_id["target"]["score"] > by_id["fast"]["score"]
    assert by_id["target"]["evidence"]["sound_profile"]["pace"]["value"] == 120.0


def test_sound_profile_ignores_unavailable_axes_per_track():
    rows = [
        {
            "id": "measured",
            "title": "Measured",
            "artist": "A",
            "sound_descriptors": {"pace": 100.0},
        },
        {"id": "missing", "title": "Missing", "artist": "B", "sound_descriptors": {}},
        {"id": "other", "title": "Other", "artist": "C", "sound_descriptors": {"pace": 140.0}},
    ]
    selected, _ = rank_curation(
        rows,
        np.eye(3, dtype=np.float32),
        "",
        "",
        track_limit=3,
        refresh_mode="fresh",
        sound_profile={"pace": 0.0},
        shuffle_seed=1,
        minimum_match_percentile=0,
    )
    by_id = {track["id"]: track for track in selected}
    assert by_id["missing"]["score"] == 0.5
    assert by_id["missing"]["evidence"]["sound_profile"] == {}


def test_sound_profile_blends_library_percentiles_for_matching(fake_text_embeddings):
    rows = [
        {"id": "missing-profile", "title": "Missing", "artist": "A", "sound_descriptors": {}},
        {"id": "middle", "title": "Middle", "artist": "B", "sound_descriptors": {"pace": 100.0}},
        {
            "id": "opposite-profile",
            "title": "Opposite",
            "artist": "C",
            "sound_descriptors": {"pace": 200.0},
        },
    ]
    matrix = np.asarray([[0.8, 0.6], [0.6, 0.8], [0, 1]], dtype=np.float32)

    selected, _ = rank_curation(
        rows,
        matrix,
        "pop",
        "",
        track_limit=3,
        refresh_mode="fresh",
        sound_profile={"pace": 0},
        shuffle_seed=1,
        minimum_match_percentile=0.8,
    )

    assert [track["id"] for track in selected] == ["missing-profile"]
    assert selected[0]["percentile"] == 1.0


def test_sound_profile_vocals_blends_vocal_and_instrumental_rankings():
    rows = [
        {
            "id": "instrumental",
            "title": "Instrumental",
            "artist": "A",
            "sound_descriptors": {"vocals": 0.05},
        },
        {"id": "mixed", "title": "Mixed", "artist": "B", "sound_descriptors": {"vocals": 0.5}},
        {"id": "vocal", "title": "Vocal", "artist": "C", "sound_descriptors": {"vocals": 0.95}},
    ]

    def scores(target: float) -> dict[str, float]:
        selected, _ = rank_curation(
            rows,
            np.eye(3, dtype=np.float32),
            "",
            "",
            track_limit=3,
            refresh_mode="fresh",
            sound_profile={"vocals": target},
            shuffle_seed=1,
            minimum_match_percentile=0,
        )
        return {track["id"]: track["score"] for track in selected}

    instrumental = scores(0)
    vocal = scores(1)
    assert instrumental["instrumental"] > instrumental["vocal"]
    assert vocal["vocal"] > vocal["instrumental"]


def test_instrumental_only_is_a_hard_voice_classifier_filter():
    rows = [
        {"id": "instrumental", "title": "Instrumental", "artist": "A"},
        {"id": "low-vocals", "title": "Low vocals", "artist": "B"},
        {"id": "vocal", "title": "Vocal", "artist": "C"},
    ]
    selected, _ = rank_curation(
        rows,
        np.eye(3, dtype=np.float32),
        "",
        "",
        track_limit=3,
        refresh_mode="fresh",
        voice_matrix=np.asarray(
            [[0.9, 0.05, 0.05], [0.49, 0.25, 0.26], [0.05, 0.5, 0.45]], dtype=np.float32
        ),
        voice_available=np.asarray([True, True, True]),
        instrumental_only=True,
        shuffle_seed=1,
        minimum_match_percentile=0,
    )
    assert [track["id"] for track in selected] == ["instrumental"]
    assert selected[0]["evidence"]["instrumental_only"] == {
        "threshold": 0.5,
        "confidence": pytest.approx(0.9),
    }
