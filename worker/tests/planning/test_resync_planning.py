"""Resync requests and the realign switch in the planners' SQL."""

from unittest.mock import MagicMock

from core.planning import store
from core.stems import store as stems


def _executed(call):
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = []
    call(connection)
    return cursor.execute.call_args.args


def test_karaoke_planning_honours_requests_and_the_realign_switch():
    sql, params = _executed(
        lambda db: store.songs_needing_karaoke(db, ["a"], None, "v12", "model", False)
    )
    assert "r.pipeline='karaoke'" in sql
    # The switch is a parameter: off, an alignment of any revision is kept.
    assert params[-4:] == [False, "v12", "model", "model"]
    _, params = _executed(lambda db: store.songs_needing_karaoke(db, ["a"], None, "v12", None))
    assert params[-4] is True


def test_a_stems_request_retries_a_failed_recipe():
    sql, _ = _executed(lambda db: stems.songs_without_stems(db, "library", ["a"], "recipe"))
    assert "r.pipeline='karaoke_stems'" in sql


def test_every_known_song_of_the_selection_can_be_asked_for():
    import pytest

    from core.resync import store as resync

    connection = MagicMock()
    resync.request_library(connection, "library", ["a", "b"], ["karaoke", "themes"], "user")
    calls = [call.args for call in connection.execute.call_args_list]
    assert [params[0] for _, params in calls] == ["karaoke", "themes"]
    assert calls[0][1] == ("karaoke", "user", "library", ["a", "b"])
    with pytest.raises(ValueError):
        resync.request_library(connection, "library", ["a"], ["transcription"])
