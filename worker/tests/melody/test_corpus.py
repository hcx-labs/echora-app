import sys
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import worker.melody.corpus as corpus
from worker.processing import ProcessingContext, run_stage

A, B, C = (uuid.UUID(f"00000000-0000-4000-8000-00000000000{n}") for n in (1, 2, 3))
CONTOURS = {s: (np.zeros(2), np.ones(2, dtype=bool)) for s in ("full-mix", "vocals")}


class Client:
    def __init__(self, *credentials):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def audio_bytes(self, external_id):
        if external_id == "c":
            raise OSError("unreachable")
        return b"audio"


def corpus_context(**overrides):
    return ProcessingContext(
        user_id="user",
        library=("url", "u", "p"),
        track_ids=("a", "b", "c"),
        options={"corpus_id": "corpus"},
        **overrides,
    )


@pytest.fixture
def database(monkeypatch):
    planning, connection = MagicMock(), MagicMock()
    planning.__enter__.return_value.cursor.return_value.__enter__.return_value.fetchone.return_value = (
        "corpus",
    )
    cursor = connection.cursor.return_value.__enter__.return_value
    # A's contours are already in the run; B needs extracting; C's audio is unreachable.
    cursor.fetchone.side_effect = [(3,), (0,), (0,)]
    connections = iter([planning, connection])
    monkeypatch.setattr(corpus, "connect", lambda **_: next(connections))
    monkeypatch.setattr(corpus, "source_rows", lambda *a: [("a", A), ("b", B), ("c", C)])
    monkeypatch.setattr(corpus, "NavidromeClient", Client)
    stored = []
    index = SimpleNamespace(
        create_sync_run=lambda connection: "run",
        track_contours=lambda audio: CONTOURS,
        store_track_contours=lambda conn, track, run, audio, usable: (
            stored.append(track) or len(usable)
        ),
    )
    monkeypatch.setitem(sys.modules, "worker.melody.index", index)
    preprocessing = SimpleNamespace(prepare_audio=MagicMock())
    monkeypatch.setitem(sys.modules, "worker.media.preprocessing", preprocessing)
    return SimpleNamespace(connection=connection, cursor=cursor, stored=stored)


def test_indexes_missing_contours_and_adds_every_song(database):
    summary = run_stage(corpus.MelodyCorpusProcessor(), corpus_context()).as_dict()
    # The summary keys the hum corpus batch always had, at the top level.
    assert summary == {"tracks": 2, "contours": 2, "failed": 1}
    assert database.stored == [B]
    added = [
        call.args[1]
        for call in database.cursor.execute.call_args_list
        if "hum_corpus_tracks" in call.args[0]
    ]
    assert added == [("corpus", A), ("corpus", B)]


def test_remote_batches_extract_on_modal(database):
    remote = SimpleNamespace(melody_one=MagicMock(return_value=CONTOURS))
    run_stage(corpus.MelodyCorpusProcessor(), corpus_context(remote=remote))
    remote.melody_one.assert_called_once_with(b"audio")


def test_unknown_corpus_fails_the_job(monkeypatch):
    planning = MagicMock()
    planning.__enter__.return_value.cursor.return_value.__enter__.return_value.fetchone.return_value = None
    monkeypatch.setattr(corpus, "connect", lambda **_: planning)
    with pytest.raises(ValueError, match="Unknown hum corpus"):
        run_stage(corpus.MelodyCorpusProcessor(), corpus_context())
