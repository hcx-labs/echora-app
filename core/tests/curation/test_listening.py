from datetime import datetime, timezone

from core.curation.listening import (
    Listen,
    TopTrack,
    match_navidrome_play_counts,
    match_top_tracks,
    navidrome_play_counts,
    track_listen_counts,
)


def test_track_listen_counts_matches_tracks_and_overnight_periods():
    rows = [{"id": "one", "title": "A Song", "artist": "An Artist"}]
    listens = [
        Listen("An Artist", "A Song", datetime(2026, 8, 24, 23, 30, tzinfo=timezone.utc)),
        Listen("An Artist", "A Song", datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)),
    ]

    all_counts, period_counts = track_listen_counts(rows, listens, "UTC", "22:00", "02:00")

    assert all_counts == {"one": 2}
    assert period_counts == {"one": 1}


def test_match_top_tracks_keeps_lastfm_order_and_skips_missing_and_duplicates():
    rows = [
        {"id": "a", "title": "Believer", "artist": "Imagine Dragons"},
        {"id": "b", "title": "Unique Song", "artist": "Someone"},
    ]
    entries = [
        TopTrack("Unknown", "Not In Library", 40),
        TopTrack("Someone Else", "Unique Song", 30),
        TopTrack("Imagine Dragons", "Believer", 20),
        TopTrack("imagine dragons", "BELIEVER", 10),
    ]

    matched = match_top_tracks(rows, entries)

    assert [(row["id"], count) for row, count in matched] == [("b", 30), ("a", 20)]


def test_navidrome_play_counts_ranks_played_songs_with_stable_ties():
    songs = [
        {"id": "unplayed", "playCount": 0},
        {"id": "b", "playCount": 5, "played": "2026-09-01T10:00:00Z"},
        {"id": "a", "playCount": 5, "played": "2026-09-20T10:00:00Z"},
        {"id": "no-date", "playCount": 5},
        {"id": "top", "playCount": 12, "played": "2026-01-01T00:00:00Z"},
        {"id": "bad", "playCount": "many"},
        {"playCount": 9},
    ]
    # Ties: the most recent play wins and songs with no play date come last.
    assert navidrome_play_counts(songs) == [("top", 12), ("a", 5), ("b", 5), ("no-date", 5)]


def test_match_navidrome_play_counts_maps_source_ids_once():
    rows = [
        {"id": "t1", "title": "One", "source_id": "nd-1"},
        {"id": "t2", "title": "Two", "source_id": "nd-2"},
        {"id": "t3", "title": "Unlinked", "source_id": None},
    ]
    ranked = [("nd-2", 9), ("missing", 7), ("nd-1", 3), ("nd-2", 1)]
    assert [(row["id"], count) for row, count in match_navidrome_play_counts(rows, ranked)] == [
        ("t2", 9),
        ("t1", 3),
    ]
