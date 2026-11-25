import uuid

import numpy as np

import backend.api.v1.library.concept_create as create
import backend.api.v1.library.concept_preview as preview

from .fakes import install


def test_create_needs_a_positive_example(client):
    response = client.post("/api/v1/library/concepts", json={"name": "Calm"})
    assert response.status_code == 422
    assert response.json() == {"detail": "A concept needs a positive prompt or positive track"}


def test_create_stores_trimmed_name(client, user, monkeypatch):
    stored = {
        "id": uuid.UUID(int=1),
        "name": "Calm",
        "description": "",
        "positive_prompts": ["calm piano"],
        "negative_prompts": [],
        "positive_track_ids": [],
        "negative_track_ids": [],
        "enabled": True,
        "created_at": "2026-11-25T00:00:00+00:00",
        "updated_at": None,
    }
    db = install(monkeypatch, create, [stored])
    response = client.post(
        "/api/v1/library/concepts", json={"name": " Calm ", "positive_prompts": ["calm piano"]}
    )
    assert response.status_code == 201
    assert response.json()["name"] == "Calm"
    assert db.executed[0][1][:3] == (user["id"], "Calm", "")


def test_preview_ranks_by_score_and_uses_example_tracks(client, monkeypatch):
    ids = [uuid.UUID(int=i) for i in range(3)]
    rows = [{"id": i, "title": f"t{n}", "artist": None, "album": None} for n, i in enumerate(ids)]
    matrix = np.eye(3)
    monkeypatch.setattr(preview, "semantic_corpus", lambda user_id: (rows, matrix, "run"))
    seen = {}

    def score(tracks, positives, negatives, positive_examples, negative_examples):
        seen["examples"] = positive_examples
        return np.array([0.1, 0.9, 0.5], dtype=np.float32), np.array([0.0, 1.0, 0.5])

    monkeypatch.setattr(preview, "score_concept", score)
    body = client.post(
        "/api/v1/library/concepts/preview",
        json={"positive_track_ids": [str(ids[2]), str(uuid.uuid4())], "limit": 2},
    ).json()
    assert [track["title"] for track in body["tracks"]] == ["t1", "t2"]
    assert body["run_id"] == "run" and body["corpus_size"] == 3
    assert seen["examples"].tolist() == [[0.0, 0.0, 1.0]]
