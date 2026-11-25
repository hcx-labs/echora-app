import uuid

import backend.api.v1.library.track_lyrics as lyrics_get
import backend.api.v1.library.track_lyrics_status as lyrics_status
import backend.api.v1.library.track_lyrics_update as lyrics_update
import backend.api.v1.library.track_transcription_language as transcription

from .fakes import install

TRACK = uuid.UUID(int=5)
BASE = f"/api/v1/library/tracks/{TRACK}"


def test_update_lyrics_keeps_text_and_resets_karaoke_when_changed(client, user, monkeypatch):
    db = install(monkeypatch, lyrics_update, [{"x": 1}, {"text": "old", "language": None}])
    response = client.put(f"{BASE}/lyrics", json={"text": "  new\n words ", "language": " JA "})
    assert response.status_code == 200
    assert response.json() == {
        "track_id": str(TRACK),
        "karaoke_pending": True,
        "lyrics_embedding_pending": True,
    }
    assert db.executed[0][1] == (user["id"], TRACK)
    assert db.executed[2] == (lyrics_update.UPSERT_SQL, (TRACK, "new\n words", "ja"))
    assert "karaoke_lyrics_variants" in db.executed[3][0]


def test_update_unchanged_lyrics_keeps_karaoke(client, monkeypatch):
    db = install(monkeypatch, lyrics_update, [{"x": 1}, {"text": "same", "language": "en"}])
    body = client.put(f"{BASE}/lyrics", json={"text": "same"}).json()
    assert body["karaoke_pending"] is False
    assert len(db.executed) == 3


def test_update_lyrics_rejects_blank_unknown_fields_and_foreign_tracks(client, monkeypatch):
    install(monkeypatch, lyrics_update, [None])
    assert client.put(f"{BASE}/lyrics", json={"text": "   "}).json() == {
        "detail": "Lyrics cannot be empty"
    }
    assert client.put(f"{BASE}/lyrics", json={"text": "a", "extra": 1}).status_code == 422
    response = client.put(f"{BASE}/lyrics", json={"text": "a"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Track is not in your library"}


def test_lyrics_status(client, monkeypatch):
    db = install(monkeypatch, lyrics_status, [{"x": 1}])
    response = client.put(f"{BASE}/lyrics/status", json={"status": "instrumental"})
    assert response.json() == {"track_id": str(TRACK), "status": "instrumental"}
    assert db.executed[1] == (lyrics_status.UPSERT_SQL, (TRACK, "instrumental", "instrumental"))
    assert client.put(f"{BASE}/lyrics/status", json={"status": "bad"}).status_code == 422


def test_forced_transcription_needs_ai_lyrics(client, monkeypatch):
    install(monkeypatch, transcription, [{"x": 1}, {"enabled": False}])
    response = client.put(f"{BASE}/lyrics/transcription-language", json={"language": "ja"})
    assert response.status_code == 409
    assert response.json()["detail"].startswith("Enable AI lyric generation")
    db = install(monkeypatch, transcription, [{"x": 1}, {"enabled": True}])
    response = client.put(f"{BASE}/lyrics/transcription-language", json={"language": "pt-BR"})
    assert response.json() == {
        "track_id": str(TRACK),
        "language": "pt-BR",
        "transcription_pending": True,
    }
    assert db.executed[2] == (transcription.UPSERT_SQL, (TRACK, "pt-BR", "pt-BR"))


def test_lyrics_prefer_karaoke_lines_and_list_ready_translations(client, monkeypatch):
    row = {
        "text": "one\ntwo",
        "language": "ja",
        "source": "navidrome",
        "provenance": {"lines": [{"text": "one"}, {"text": "two"}]},
        "availability_status": "available",
        "karaoke_lines": [{"text": "one", "words": []}],
        "karaoke_model": "kara",
        "karaoke_model_revision": "r",
        "karaoke_bounded": False,
    }
    install(monkeypatch, lyrics_get, [row])
    monkeypatch.setattr(
        lyrics_get,
        "load_translations",
        lambda track_id, lines: [
            {
                "status": "ready",
                "source_language": "ja",
                "target_language": "en",
                "lines": [{"text": "uno"}],
            },
            {
                "status": "ready",
                "source_language": "ja",
                "target_language": "de",
                "lines": [{"text": " "}],
            },
            {"status": "failed", "source_language": "ja", "target_language": "fr", "lines": []},
        ],
    )
    body = client.get(f"{BASE}/lyrics").json()
    assert body["available"] is True and body["karaoke"] is True
    assert body["lines"] == row["karaoke_lines"]
    assert [t["target_language"] for t in body["translations"]] == ["en"]
    assert body["translations"][0]["source_lines"] == ["one", "two"]
    assert "karaoke_ass" not in body


def test_lyrics_of_a_foreign_track_are_404(client, monkeypatch):
    install(monkeypatch, lyrics_get, [None])
    assert client.get(f"{BASE}/lyrics").status_code == 404
