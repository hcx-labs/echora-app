from core.library.relink import plan_relinks


def song(isrc=None, path=None, size=1):
    return {"isrc": [isrc] if isrc else [], "path": path, "size": size}


def test_a_renumbered_song_is_matched_by_isrc_even_when_the_file_moved():
    known = {"old": song("USAAA0000001", "a/1.flac"), "kept": song("USAAA0000002", "a/2.flac")}
    catalog = {"new": song("US-AAA-00-00001", "moved/1.flac"), "kept": known["kept"]}
    assert plan_relinks(known, catalog) == {"old": "new"}


def test_a_recording_on_two_albums_is_told_apart_by_path():
    known = {"o1": song("X1", "single/1.flac"), "o2": song("X1", "album/3.flac")}
    catalog = {"n1": song("X1", "album/3.flac"), "n2": song("X1", "single/1.flac")}
    assert plan_relinks(known, catalog) == {"o1": "n2", "o2": "n1"}


def test_an_ambiguous_match_is_left_alone():
    known = {"o1": song("X1", "gone/1.flac")}
    catalog = {"n1": song("X1", "a/1.flac"), "n2": song("X1", "b/1.flac")}
    assert plan_relinks(known, catalog) == {}


def test_two_vanished_sources_cannot_claim_the_same_song():
    known = {"o1": song("X1", "a.flac"), "o2": song("X1", "b.flac")}
    assert plan_relinks(known, {"n1": song("X1", "c.flac")}) == {}


def test_without_isrc_the_same_file_is_the_same_song():
    known = {"old": song(None, "a/1.flac", 10), "other": song(None, "a/2.flac", 10)}
    catalog = {"new": song(None, "a/1.flac", 10), "resized": song(None, "a/2.flac", 11)}
    assert plan_relinks(known, catalog) == {"old": "new"}


def test_new_songs_and_removed_songs_are_not_linked_to_each_other():
    assert plan_relinks({"old": song("X1", "a.flac")}, {"new": song("X2", "b.flac")}) == {}
    assert plan_relinks({"same": song("X1", "a.flac")}, {"same": song("X1", "a.flac")}) == {}
