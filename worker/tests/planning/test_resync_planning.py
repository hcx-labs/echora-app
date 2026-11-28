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


def test_only_songs_of_an_older_revision_are_asked_for():
    import pytest

    from core.resync import store as resync

    connection = MagicMock()
    resync.request_outdated(
        connection, "library", ["a", "b"], {"karaoke": "v12 m", "themes": "7"}, "user"
    )
    (karaoke_sql, karaoke), (themes_sql, themes) = [
        call.args for call in connection.execute.call_args_list
    ]
    assert "kv.model<>'navidrome'" in karaoke_sql and "<> %s" in karaoke_sql
    assert karaoke == ("karaoke", "user", "library", ["a", "b"], "v12 m")
    assert "th.prompt_revision" in themes_sql and themes[-1] == "7"
    with pytest.raises(ValueError):
        resync.request_outdated(connection, "library", ["a"], {"transcription": "x"})
