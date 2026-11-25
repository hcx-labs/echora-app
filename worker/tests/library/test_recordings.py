import uuid
from unittest.mock import MagicMock

import pytest

import worker.library.recordings as recordings
from worker.processing import ProcessingContext, run_stage

A, B, C = (uuid.UUID(f"00000000-0000-4000-8000-00000000000{n}") for n in (1, 2, 3))


def test_fingerprints_missing_songs_and_counts_matches(monkeypatch):
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    # A has a fingerprint already; B and C need one.
    cursor.fetchone.side_effect = [None, (200.0,), (180.0,)]
    monkeypatch.setattr(recordings, "connect", lambda: connection)
    monkeypatch.setattr(recordings, "source_rows", lambda *a: [("a", A), ("b", B), ("c", C)])

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
            return b"audio-" + external_id.encode()

    monkeypatch.setattr(recordings, "NavidromeClient", Client)
    monkeypatch.setattr(recordings, "compute_chromaprint", lambda audio: (b"fp", 0.0, "1"))
    stored = []

    def store(connection, track_id, fingerprint, duration, version):
        stored.append((track_id, duration))
        return {"matched": True}

    monkeypatch.setattr(recordings, "store_fingerprint", store)
    context = ProcessingContext(
        user_id="user", library=("url", "u", "p"), track_ids=("a", "b", "c")
    )
    summary = run_stage(recordings.FingerprintProcessor(), context).as_dict()
    # The summary keys the recordings batch always had, at the top level.
    assert summary == {"fingerprinted": 1, "matched": 1, "failed": 1}
    # fpcalc measured no duration, so the library's is stored.
    assert stored == [(B, 200.0)]
    connection.commit.assert_called_once()


def test_storage_failure_rolls_back(monkeypatch):
    connection = MagicMock()
    processor = recordings.FingerprintProcessor()
    processor.connection = connection

    def store(*a):
        raise RuntimeError("constraint")

    monkeypatch.setattr(recordings, "store_fingerprint", store)
    item = recordings.FingerprintItem(key="b", track_id=B, external_id="b", duration=1.0)
    with pytest.raises(RuntimeError):
        processor.persist(ProcessingContext(), item, recordings.Fingerprint(b"fp", 1.0, "1"))
    connection.rollback.assert_called_once()
