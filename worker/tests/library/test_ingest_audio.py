"""Execution binds canonical artifacts to hashed source bytes, not mutable aliases."""

from core.settings import get_settings

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from uuid import uuid4
import os

import psycopg
import pytest

import worker.library.ingest as ingest
from worker.planning.audio import AudioProcessingPlan, plan_audio


@pytest.fixture
def execution(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    client = MagicMock()
    client.__enter__.return_value = client
    monkeypatch.setattr(ingest.psycopg, "connect", Mock(return_value=connection))
    monkeypatch.setattr(ingest, "NavidromeClient", Mock(return_value=client))
    monkeypatch.setattr(ingest, "configure_representations", Mock())
    monkeypatch.setattr(ingest, "_library", Mock(return_value=uuid4()))
    monkeypatch.setattr(ingest, "active_cache", lambda: None)
    monkeypatch.setattr(ingest, "get_check", lambda: lambda: None)
    monkeypatch.setattr(ingest.torch.cuda, "is_available", lambda: False)
    old_lookup = Mock(side_effect=AssertionError("must not reread mutable source identity"))
    monkeypatch.setattr(ingest, "_source_track_id", old_lookup)
    upsert = Mock()
    monkeypatch.setattr(ingest, "_upsert_track", upsert)
    waveform = Mock(return_value=True)
    monkeypatch.setattr(ingest, "store_waveform", waveform)
    planner = Mock()
    monkeypatch.setattr(ingest, "plan_audio", planner)
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    get_settings.cache_clear()
    return connection, client, upsert, waveform, planner


def song(name):
    return SimpleNamespace(id=name, title=name, artist="Artist", album="Album", duration=2)


def waveform_plan(*ids):
    return AudioProcessingPlan(
        frozenset(), frozenset(), frozenset(), frozenset(), waveform_external_ids=frozenset(ids)
    )


def test_artifact_rejects_changed_bytes_when_download_cache_misses(execution):
    _, client, upsert, waveform, planner = execution
    client.tracks.return_value = [song("source")]
    client.audio_bytes.side_effect = [b"identity bytes", b"replaced bytes"]
    upsert.return_value = (uuid4(), False)
    planner.return_value = waveform_plan("source")
    result = ingest.ingest_navidrome("url", "user", "password", ["source"])
    waveform.assert_not_called()
    assert result.failed == 1


def test_artifact_and_plan_keep_resolved_id_after_alias_can_change(execution):
    _, client, upsert, waveform, planner = execution
    track = uuid4()
    client.tracks.return_value = [song("source")]
    client.audio_bytes.return_value = b"identity bytes"
    upsert.return_value = (track, False)
    planner.return_value = waveform_plan("source")
    result = ingest.ingest_navidrome("url", "user", "password", ["source"])
    assert result.failed == 0
    assert planner.call_args.kwargs["resolved_track_ids"] == {"source": track}
    assert waveform.call_args.args[1:3] == (track, b"identity bytes")


def test_unavailable_source_does_not_starve_healthy_siblings_and_can_retry(execution):
    connection, client, upsert, waveform, planner = execution
    a, c = uuid4(), uuid4()
    client.tracks.return_value = [song("a"), song("bad"), song("c")]
    client.audio_bytes.side_effect = [b"a", OSError("unavailable"), b"c", b"a", b"c"]
    upsert.side_effect = [(a, False), (c, False)]
    planner.return_value = waveform_plan("a", "c")
    result = ingest.ingest_navidrome("url", "user", "password", ["a", "bad", "c"])
    assert result.failed == 1 and result.downloaded == 2
    assert waveform.call_count == 2
    assert planner.call_args.args[2] == ["a", "c"]
    connection.rollback.assert_called()
    client.tracks.return_value = [song("bad")]
    client.audio_bytes.side_effect = [b"healthy now", b"healthy now"]
    upsert.side_effect = [(uuid4(), False)]
    planner.return_value = waveform_plan("bad")
    result = ingest.ingest_navidrome("url", "user", "password", ["bad"])
    assert result.failed == 0 and waveform.call_count == 3


def test_real_audio_plan_uses_bound_identity_after_source_remap(monkeypatch):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL required")
    import worker.recordings.encoder as recording_encoder

    representation = "e" * 64
    monkeypatch.setattr(
        recording_encoder,
        "config_from_env",
        lambda: SimpleNamespace(representation_id=representation),
    )
    library, namespace, old, replacement = [uuid4() for _ in range(4)]
    db = psycopg.connect(url)
    try:
        db.execute(
            "INSERT INTO libraries(id,namespace,name,root_path) VALUES (%s,%s,'test','https://test')",
            (library, namespace),
        )
        for track in [old, replacement]:
            db.execute(
                "INSERT INTO tracks(id,audio_hash,title,duration_seconds) VALUES (%s,%s,'test',1)",
                (track, str(track).replace("-", "") * 2),
            )
        db.execute(
            "INSERT INTO track_sources(library_id,source_type,external_id,track_id,source_data) VALUES (%s,'subsonic','source',%s,'{}')",
            (library, replacement),
        )
        db.execute(
            "INSERT INTO recording_fingerprints(track_id,representation_id,segment_count,fingerprints) VALUES (%s,%s,1,%s)",
            (old, representation, bytes(512)),
        )
        pinned = plan_audio(db, library, ["source"], resolved_track_ids={"source": old})
        current = plan_audio(db, library, ["source"])
        assert not pinned.recording_fingerprint_external_ids
        assert current.recording_fingerprint_external_ids == frozenset({"source"})
    finally:
        db.rollback()
        db.close()
