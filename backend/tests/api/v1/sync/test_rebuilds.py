from unittest.mock import MagicMock, Mock

import pytest

import backend.api.v1.sync.audio_profiles_rebuild as rebuild
import core.jobs.queue as jobs

from ._fakes import snapshot


@pytest.fixture
def enqueue(monkeypatch):
    fake = Mock(side_effect=lambda kind, *a, **k: snapshot(kind))
    monkeypatch.setattr(jobs, "enqueue", fake)
    return fake


def test_semantic_fusion_rebuild_is_one_job_for_the_corpus(client, user, enqueue):
    response = client.post("/api/v1/sync/semantic-fusion")
    assert response.status_code == 202
    enqueue.assert_called_once_with(
        "semantic_fusion_build", "analysis", user["id"], dedupe_key="semantic_fusion_build"
    )


def test_audio_profile_rebuild_lists_each_song_once(client, user, enqueue, monkeypatch):
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [("t2", "mert"), ("t1", "mert"), ("t2", "muq_mulan")]
    monkeypatch.setattr(rebuild, "connect", lambda **_: connection)
    assert client.post("/api/v1/sync/audio-profiles").status_code == 202
    assert enqueue.call_args.kwargs == {
        "payload": {"track_ids": ["t1", "t2"]},
        "dedupe_key": f"audio_profiles:{user['id']}",
    }
