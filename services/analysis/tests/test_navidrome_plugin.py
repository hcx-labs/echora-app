"""Ranking behavior, native lyrics mapping, credential lifecycle and source isolation."""

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from xml.etree import ElementTree as ET

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import numpy as np
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
import pytest

from echora_analysis import navidrome_integration as auth
from echora_analysis import plugin_ranking as ranking
from echora_analysis import plugin_lyrics as lyrics
from echora_analysis.plugin_routes import router


def vector(dimension, *values):
    result = np.zeros(dimension)
    result[: len(values)] = values
    return result


def row(source, artist, muq, lyric=None, *, canonical=None, group=None):
    return {
        "id": canonical or uuid4(),
        "source_id": source,
        "title": source,
        "artist": artist,
        "source_artist": artist,
        "artist_id": artist,
        "album": "Album",
        "duration_seconds": 180,
        "recording_group_id": group,
        "muq": vector(512, *muq),
        "mert": vector(768, *muq),
        "lyrics": vector(1024, *lyric) if lyric else None,
    }


def test_weights_reverse_order_without_new_representations():
    corpus = ranking.corpus_from_rows(
        [
            row("seed", "a", [1, 0], [1, 0]),
            row("musical", "b", [1, 0], [0, 1]),
            row("lyrical", "c", [0, 1], [1, 0]),
        ]
    )
    for weight, expected in [(80, "musical"), (20, "lyrical")]:
        result = ranking.similar_tracks(
            corpus, "seed", 2, auth.RankingProfile(musical_weight=weight)
        )
        assert result["matches"][0]["song"]["id"] == expected
        assert result["matches"][0]["similarity"] == pytest.approx(0.8)


def test_missing_lyrics_policy_and_pure_audio_mode():
    corpus = ranking.corpus_from_rows(
        [row("seed", "a", [1, 0], [1, 0]), row("missing", "b", [1, 0])]
    )
    fallback = ranking.similar_tracks(corpus, "seed", 5, auth.RankingProfile())
    assert fallback["matches"][0]["similarity"] == 1
    excluded = ranking.similar_tracks(
        corpus, "seed", 5, auth.RankingProfile(missing_lyrics="exclude")
    )
    assert excluded["matches"] == []
    audio = ranking.similar_tracks(
        corpus, "seed", 5, auth.RankingProfile(missing_lyrics="exclude", musical_weight=100)
    )
    assert len(audio["matches"]) == 1


def test_semantic_acoustic_mix_changes_ranking():
    rows = [row("seed", "a", [1, 0]), row("semantic", "b", [1, 0]), row("acoustic", "c", [0, 1])]
    rows[1]["mert"] = vector(768, 0, 1)
    rows[2]["mert"] = vector(768, 1, 0)
    corpus = ranking.corpus_from_rows(rows)
    for weight, expected in [(100, "semantic"), (0, "acoustic")]:
        result = ranking.similar_tracks(
            corpus,
            "seed",
            2,
            auth.RankingProfile(musical_weight=100, musical_semantic_weight=weight),
        )
        assert result["matches"][0]["song"]["id"] == expected


def test_alternate_ids_recording_duplicates_and_artist_caps():
    seed = row("seed", "a", [1, 0], group="seed-group")
    duplicate = row("alternate-seed", "a", [1, 0], canonical=seed["id"])
    equivalent = row("same-recording", "a", [1, 0], group="seed-group")
    corpus = ranking.corpus_from_rows(
        [
            seed,
            duplicate,
            equivalent,
            row("b1", "b", [1, 0]),
            row("b2", "b", [1, 0]),
            row("c", "c", [0.8, 0.2]),
        ]
    )
    result = ranking.similar_tracks(corpus, "seed", 20, auth.RankingProfile(max_per_artist=1))
    assert [m["song"]["id"] for m in result["matches"]] == ["b1", "c"]
    with pytest.raises(HTTPException) as error:
        corpus.seed("not-visible")
    assert error.value.status_code == 404


def test_paths_keep_endpoints_and_deduplicate_source_ids():
    start, end = row("start", "a", [1, 0]), row("end", "z", [0, 1])
    corpus = ranking.corpus_from_rows(
        [
            start,
            end,
            row("middle", "b", [1, 1]),
            row("start-alias", "a", [1, 0], canonical=start["id"]),
        ]
    )
    result = ranking.sonic_path(corpus, "start", "end", 20, auth.RankingProfile())
    ids = [m["song"]["id"] for m in result["matches"]]
    assert ids == ["start", "middle", "end"]
    assert all(m["similarity"] == -1 for m in result["matches"])
    with pytest.raises(HTTPException):
        ranking.sonic_path(corpus, "start", "start-alias", 3, auth.RankingProfile())


def test_artist_providers_return_navidrome_ids_and_representative_songs():
    corpus = ranking.corpus_from_rows(
        [
            row("a1", "a", [1, 0]),
            row("a2", "a", [1, 0.1]),
            row("b1", "b", [1, 0.1]),
            row("c1", "c", [0, 1]),
        ]
    )
    artists = ranking.similar_artists(corpus, "a", 5, auth.RankingProfile())
    assert artists["artists"][0]["id"] == "b"
    radio = ranking.artist_radio(corpus, "a", 5, auth.RankingProfile(max_per_artist=1))
    assert {song["id"] for song in radio["songs"]} == {"a1", "b1", "c1"}


def lyric_example():
    row = {
        "text": "Hello & world\nこんにちは",
        "language": "en",
        "provenance": {
            "ai_generated": True,
            "lines": [
                {"text": "Hello & world", "start_ms": 1000},
                {"text": "こんにちは", "start_ms": 5000},
            ],
        },
        "karaoke_lines": [
            {
                "text": "Hello & world",
                "start_ms": 1000,
                "end_ms": 3000,
                "syllables": [
                    {"text": "Hello ", "start_ms": 1000, "end_ms": 2000},
                    {"text": "& world", "start_ms": 2000, "end_ms": 3000},
                ],
            },
            {"text": "こんにちは", "start_ms": 5000, "end_ms": 6000},
        ],
    }
    translation = {
        "target_language": "es",
        "source_language": "en",
        "provenance": "ai",
        "source_checksum": lyrics.source_checksum(lyrics.original_lines(row)),
        "status": "ready",
        "lines": [{"id": 0, "text": "Hola & mundo"}, {"id": 1, "text": "Hola"}],
    }
    return row, translation


def backing_example():
    """A lead line with an inline echo on the second voice, and a whole backing line."""
    return {
        "text": "Lonely (so lonely) tonight\n(One, two, three)",
        "language": "en",
        "provenance": {
            "lines": [{"text": "Lonely (so lonely) tonight"}, {"text": "(One, two, three)"}]
        },
        "karaoke_lines": [
            {
                "text": "Lonely (so lonely) tonight",
                "start_ms": 1000,
                "end_ms": 3200,
                "syllables": [
                    {"text": "Lone", "start_ms": 1000, "end_ms": 1300},
                    {"text": "ly (", "start_ms": 1300, "end_ms": 1600},
                    {"text": "so ", "start_ms": 1500, "end_ms": 1800, "voice": 2},
                    {"text": "lone", "start_ms": 1800, "end_ms": 2100, "voice": 2},
                    {"text": "ly) ", "start_ms": 2100, "end_ms": 2400, "voice": 2},
                    {"text": "tonight", "start_ms": 2600, "end_ms": 3200},
                ],
            },
            {
                "text": "(One, two, three)",
                "start_ms": 4000,
                "end_ms": 5500,
                "voice": "background",
                "syllables": [
                    {"text": "(One, ", "start_ms": 4000, "end_ms": 4500},
                    {"text": "two, ", "start_ms": 4500, "end_ms": 5000},
                    {"text": "three)", "start_ms": 5000, "end_ms": 5500},
                ],
            },
        ],
    }


def test_ttml_marks_backing_vocals_with_the_background_role():
    root = ET.fromstring(lyrics.ttml(backing_example(), []))
    role = f"{{{lyrics.TTM}}}role"
    lead, backing_line = root.findall(f".//{{{lyrics.TT}}}p")
    # The echo is one x-bg span, timed over its words, nested in the lead paragraph.
    children = list(lead)
    assert [child.get(role) for child in children] == [None, None, "x-bg", None]
    echo = children[2]
    assert (echo.get("begin"), echo.get("end")) == ("00:00:01.500", "00:00:02.400")
    assert [span.text for span in echo] == ["so ", "lone", "ly) "]
    assert echo[0].get("begin") == "00:00:01.500"
    assert "".join(lead.itertext()) == "Lonely (so lonely) tonight"
    # A whole backing line carries the role on its paragraph; its words are plain spans.
    assert backing_line.get(role) == "x-bg"
    assert [span.get(role) for span in backing_line] == [None, None, None]
    assert "".join(backing_line.itertext()) == "(One, two, three)"
    assert root.get(role) is None


def test_ttml_marks_bracketed_words_in_rows_stored_without_voice_marks():
    row = backing_example()
    for line in row["karaoke_lines"] if "karaoke_lines" in row else []:
        line.pop("voice", None)
        for syllable in line["syllables"]:
            syllable.pop("voice", None)
    assert lyrics.ttml(row, []) == lyrics.ttml(backing_example(), [])


def test_backing_fixture_matches_current_export():
    exported = lyrics.lyrics_response(backing_example(), [], auth.RankingProfile())["lyrics"][0][
        "text"
    ]
    fixture = (
        Path(__file__).resolve().parents[3]
        / "plugins/navidrome/contract-tests/testdata/lyrics-backing.ttml"
    )
    assert exported == fixture.read_text(encoding="utf-8")


def test_ttml_preserves_unicode_cues_translation_kind_and_line_mapping():
    row, translated = lyric_example()
    result = lyrics.lyrics_response(row, [translated], auth.RankingProfile())
    root = ET.fromstring(result["lyrics"][0]["text"])
    paragraphs = root.findall(f".//{{{lyrics.TT}}}p")
    assert "".join(paragraphs[0].itertext()) == "Hello & world"
    assert paragraphs[0].attrib["begin"] == "00:00:01.000"
    assert paragraphs[0].find(f"{{{lyrics.TT}}}span").attrib["end"] == "00:00:02.000"
    track = root.find(f".//{{{lyrics.TTM}}}translation")
    assert track.attrib[f"{{{lyrics.XML}}}lang"] == "es"
    assert track[0].attrib["for"] == paragraphs[0].attrib["key"]
    assert track[0].text == "Hola & mundo"


@pytest.mark.parametrize("format", ["ttml", "lrc"])
def test_native_parser_fixtures_match_current_export(format):
    row, translated = lyric_example()
    exported = lyrics.lyrics_response(row, [translated], auth.RankingProfile(lyrics_format=format))[
        "lyrics"
    ][0]["text"]
    fixture = (
        Path(__file__).resolve().parents[3]
        / "plugins/navidrome/contract-tests/testdata"
        / f"lyrics.{format}"
    )
    assert exported == fixture.read_text(encoding="utf-8")


@pytest.mark.parametrize("change", ["stale", "checksum", "line-count"])
def test_invalid_translations_are_never_attached(change):
    row, translated = lyric_example()
    if change == "stale":
        translated["status"] = "stale"
    if change == "checksum":
        translated["source_checksum"] = "0" * 64
    if change == "line-count":
        translated["lines"].pop()
    root = ET.fromstring(lyrics.ttml(row, [translated]))
    assert root.find(f".//{{{lyrics.TTM}}}translation") is None


def test_karaoke_text_difference_does_not_discard_translation():
    row, translated = lyric_example()
    row["karaoke_lines"][0]["text"] = "Hello, world!"
    root = ET.fromstring(lyrics.ttml(row, [translated]))
    track = root.find(f".//{{{lyrics.TTM}}}translation")
    assert [line.text for line in track] == ["Hola & mundo", "Hola"]


@pytest.mark.parametrize("format", ["ttml", "lrc"])
def test_changed_karaoke_segmentation_does_not_discard_source_translation_lines(format):
    row, translated = lyric_example()
    row["karaoke_lines"] = [{"text": "Hello & world こんにちは", "start_ms": 1000}]
    result = lyrics.lyrics_response(row, [translated], auth.RankingProfile(lyrics_format=format))
    if format == "ttml":
        root = ET.fromstring(result["lyrics"][0]["text"])
        track = root.find(f".//{{{lyrics.TTM}}}translation")
        assert [line.text for line in track] == ["Hola & mundo", "Hola"]
        paragraphs = root.findall(f".//{{{lyrics.TT}}}p")
        assert [line.text for line in paragraphs] == ["Hello & world", "こんにちは"]
        assert paragraphs[1].attrib["begin"] == "00:00:05.000"
    else:
        assert result["lyrics"][1]["text"] == "[00:01.000]Hola & mundo\n[00:05.000]Hola"


@pytest.mark.parametrize("format", ["ttml", "lrc"])
def test_translation_maps_source_ids_after_karaoke_removes_blank_stanzas(format):
    row, translated = lyric_example()
    # Mirrors alignment: blank stanza lines have no tokens, and edges are trimmed.
    row["text"] = "  Hello & world  \n\nこんにちは\n\nHello & world"
    row["provenance"] = {}
    row["karaoke_lines"].append({"text": "Hello & world", "start_ms": 9000, "end_ms": 10000})
    translated["source_checksum"] = lyrics.source_checksum(lyrics.original_lines(row))
    translated["lines"] = [
        {"id": 0, "text": "Hola & mundo"},
        {"id": 1, "text": ""},
        {"id": 2, "text": "Hola"},
        {"id": 3, "text": ""},
        {"id": 4, "text": "Hola de nuevo"},
    ]
    result = lyrics.lyrics_response(row, [translated], auth.RankingProfile(lyrics_format=format))
    if format == "ttml":
        root = ET.fromstring(result["lyrics"][0]["text"])
        track = root.find(f".//{{{lyrics.TTM}}}translation")
        assert [(line.attrib["for"], line.text) for line in track] == [
            ("line-0", "Hola & mundo"),
            ("line-1", "Hola"),
            ("line-2", "Hola de nuevo"),
        ]
        paragraphs = root.findall(f".//{{{lyrics.TT}}}p")
        assert paragraphs[1].attrib["begin"] == "00:00:05.000"
        assert paragraphs[2].attrib["begin"] == "00:00:09.000"
        assert len(paragraphs[0].findall(f"{{{lyrics.TT}}}span")) == 2
    else:
        assert result["lyrics"][1]["text"] == (
            "[00:01.000]Hola & mundo\n[00:05.000]Hola\n[00:09.000]Hola de nuevo"
        )


def test_blank_stanza_mapping_still_rejects_stale_translation():
    row, translated = lyric_example()
    row["text"] = "Hello & world\n\nこんにちは"
    row["provenance"] = {}
    # The old translation's checksum and IDs must not be reused for changed input.
    root = ET.fromstring(lyrics.ttml(row, [translated]))
    assert root.find(f".//{{{lyrics.TTM}}}translation") is None


@pytest.mark.parametrize("format", ["ttml", "lrc"])
def test_identical_translation_line_is_omitted_without_shifting_remaining_timing(format):
    row, translated = lyric_example()
    translated["lines"][0]["text"] = "  Hello & world  "
    result = lyrics.lyrics_response(row, [translated], auth.RankingProfile(lyrics_format=format))
    if format == "ttml":
        root = ET.fromstring(result["lyrics"][0]["text"])
        track = root.find(f".//{{{lyrics.TTM}}}translation")
        assert len(track) == 1
        assert track[0].attrib["for"] == "line-1"
        assert track[0].text == "Hola"
        assert len(root.findall(f".//{{{lyrics.TT}}}p")) == 2
        fixture = (
            Path(__file__).resolve().parents[3]
            / "plugins/navidrome/contract-tests/testdata/lyrics-filtered.ttml"
        )
        assert result["lyrics"][0]["text"] == fixture.read_text(encoding="utf-8")
    else:
        assert result["lyrics"][1]["text"] == "[00:05.000]Hola"
        assert "Hello & world" in result["lyrics"][0]["text"]


@pytest.mark.parametrize("format", ["ttml", "lrc"])
def test_entire_identical_translation_track_is_omitted(format):
    row, translated = lyric_example()
    translated["lines"] = [
        {"id": i, "text": text} for i, text in enumerate(lyrics.original_lines(row))
    ]
    result = lyrics.lyrics_response(row, [translated], auth.RankingProfile(lyrics_format=format))
    assert len(result["lyrics"]) == 1
    if format == "ttml":
        root = ET.fromstring(result["lyrics"][0]["text"])
        assert root.find(f".//{{{lyrics.TTM}}}translation") is None


def test_plain_transcripts_lrc_fallback_and_lyrics_toggle():
    row, translated = lyric_example()
    lrc = lyrics.lyrics_response(row, [translated], auth.RankingProfile(lyrics_format="lrc"))
    assert [entry["lang"] for entry in lrc["lyrics"]] == ["en", "es"]
    assert "[00:01.000]Hola & mundo" in lrc["lyrics"][1]["text"]
    assert lyrics.lyrics_response(row, [translated], auth.RankingProfile(serve_lyrics=False)) == {
        "lyrics": []
    }
    assert lyrics.lyrics_response(None, [], auth.RankingProfile()) == {"lyrics": []}
    row["karaoke_lines"] = []
    row["provenance"] = {}
    root = ET.fromstring(
        lyrics.lyrics_response(row, [], auth.RankingProfile())["lyrics"][0]["text"]
    )
    assert root.find(f".//{{{lyrics.TT}}}p").attrib == {"key": "line-0"}


@pytest.fixture
def database(monkeypatch):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is required for credential and isolation tests")
    schema = "test_nd_plugin_" + uuid4().hex
    with psycopg.connect(url, autocommit=True) as db:
        db.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

    @contextmanager
    def connect():
        with psycopg.connect(url, options=f"-c search_path={schema}", row_factory=dict_row) as db:
            yield db

    monkeypatch.setattr(auth, "connect", connect)
    monkeypatch.setattr(ranking, "connect", connect)
    monkeypatch.setattr(lyrics, "connect", connect)
    with connect() as db:
        db.execute("""
            CREATE TABLE users(id uuid PRIMARY KEY,is_blocked boolean NOT NULL DEFAULT false);
            CREATE TABLE navidrome_connections(id uuid PRIMARY KEY,owner_user_id uuid REFERENCES users,url text);
            CREATE TABLE libraries(id uuid PRIMARY KEY,namespace uuid);
            CREATE TABLE tracks(id uuid PRIMARY KEY,title text,artist text,album text,duration_seconds float);
            CREATE TABLE user_source_memberships(user_id uuid,library_id uuid,external_id text);
            CREATE TABLE track_sources(library_id uuid,external_id text,track_id uuid,source_type text,source_data jsonb);
            CREATE TABLE recording_group_members(track_id uuid,group_id uuid);
            CREATE TABLE analysis_runs(id uuid PRIMARY KEY,model_name text,created_at timestamptz DEFAULT now());
            CREATE TABLE current_embeddings(track_id uuid,run_id uuid,embedding_type text,window_index integer,embedding text);
            CREATE TABLE lyrics(track_id uuid,text text,language text,provenance jsonb);
            CREATE TABLE karaoke_lyrics_variants(track_id uuid,bounded boolean,lines jsonb);
            CREATE TABLE lyric_translations(track_id uuid,status text,source_checksum text,target_language text,
                source_language text,provenance text,lines jsonb);
        """)
        for filename in (
            "0034_jobs.py",
            "0055_navidrome_integration.py",
            "0056_navidrome_original_lyrics.py",
        ):
            migration_path = Path(__file__).parents[1] / "alembic/versions" / filename
            spec = importlib.util.spec_from_file_location("integration_migration", migration_path)
            migration = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(migration)
            migration.op = SimpleNamespace(execute=db.execute)
            migration.upgrade()
        owner, other, conn = uuid4(), uuid4(), uuid4()
        db.execute("INSERT INTO users(id) VALUES (%s),(%s)", (owner, other))
        db.execute(
            "INSERT INTO navidrome_connections VALUES (%s,%s,'http://navidrome')", (conn, owner)
        )
    caller = {"id": owner, "navidrome_connection_id": conn}
    app = FastAPI()
    app.include_router(auth.settings_router(lambda: caller))
    app.include_router(router)
    try:
        yield SimpleNamespace(
            connect=connect,
            owner=owner,
            other=other,
            connection=conn,
            caller=caller,
            client=TestClient(app),
        )
    finally:
        with psycopg.connect(url, autocommit=True) as db:
            db.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def enable(db):
    response = db.client.put(
        "/settings/integrations/navidrome",
        json={"enabled": True, "connection_id": str(db.connection)},
    )
    assert response.status_code == 200, response.text
    return response.json()["generated_key"]["secret"]


def test_key_generation_hashing_rotation_and_cross_user_revocation(database):
    db = database
    token = enable(db)
    settings = db.client.get("/settings/integrations/navidrome")
    assert settings.headers["cache-control"] == "no-store"
    assert token not in settings.text
    with db.connect() as connection:
        stored = connection.execute("SELECT token_hash FROM navidrome_api_keys").fetchone()
    assert stored["token_hash"] == auth.digest(token)
    assert auth.authenticate("Bearer " + token)["user_id"] == db.owner
    rotated = db.client.post(
        "/settings/integrations/navidrome/keys", json={"label": "Rotation", "expiry_days": 7}
    )
    assert rotated.status_code == 201
    key = rotated.json()
    assert (datetime.fromisoformat(key["expires_at"]) - datetime.now(timezone.utc)).days == 6
    assert auth.authenticate("Bearer " + key["secret"])["user_id"] == db.owner
    db.caller["id"] = db.other
    assert db.client.delete("/settings/integrations/navidrome/keys/" + key["id"]).status_code == 404
    db.caller["id"] = db.owner
    assert db.client.delete("/settings/integrations/navidrome/keys/" + key["id"]).status_code == 204
    with pytest.raises(HTTPException):
        auth.authenticate("Bearer " + key["secret"])
    assert auth.authenticate("Bearer " + token)


@pytest.mark.parametrize(
    "rejection", ["expired", "revoked", "blocked", "disabled", "wrong", "missing"]
)
def test_invalid_keys_never_reach_provider(database, rejection):
    db = database
    token = enable(db)
    with db.connect() as connection:
        if rejection == "expired":
            connection.execute(
                "UPDATE navidrome_api_keys SET created_at=now()-interval '2 days',expires_at=now()-interval '1 day'"
            )
        if rejection == "revoked":
            connection.execute("UPDATE navidrome_api_keys SET revoked_at=now()")
        if rejection == "blocked":
            connection.execute("UPDATE users SET is_blocked=true WHERE id=%s", (db.owner,))
        if rejection == "disabled":
            connection.execute("UPDATE navidrome_integrations SET enabled=false")
    headers = (
        {}
        if rejection == "missing"
        else {"Authorization": "Bearer " + ("wrong" if rejection == "wrong" else token)}
    )
    response = db.client.post(
        "/integrations/navidrome/v1/similar-tracks", headers=headers, json={"song": {"id": "seed"}}
    )
    assert response.status_code == 401


def test_connection_changes_revoke_old_keys_and_reject_foreign_connections(database):
    db = database
    token = enable(db)
    foreign, new = uuid4(), uuid4()
    with db.connect() as connection:
        connection.execute(
            "INSERT INTO navidrome_connections VALUES (%s,%s,'http://foreign'),(%s,%s,'http://new')",
            (foreign, db.other, new, db.owner),
        )
    assert (
        db.client.put(
            "/settings/integrations/navidrome",
            json={"enabled": True, "connection_id": str(foreign)},
        ).status_code
        == 404
    )
    assert auth.authenticate("Bearer " + token)
    response = db.client.put(
        "/settings/integrations/navidrome", json={"enabled": True, "connection_id": str(new)}
    )
    assert response.status_code == 200
    with pytest.raises(HTTPException):
        auth.authenticate("Bearer " + token)
    assert (
        auth.authenticate("Bearer " + response.json()["generated_key"]["secret"])["connection_id"]
        == new
    )


def test_sql_corpus_and_lyrics_are_scoped_before_ranking(database):
    from psycopg.types.json import Jsonb

    db = database
    token = enable(db)
    principal = auth.authenticate("Bearer " + token)
    library, foreign_lib, run = uuid4(), uuid4(), uuid4()
    owned, private, other_server = uuid4(), uuid4(), uuid4()
    with db.connect() as connection:
        connection.execute(
            "INSERT INTO libraries VALUES (%s,%s),(%s,%s)",
            (library, ranking.library_namespace(principal), foreign_lib, uuid4()),
        )
        connection.execute(
            "INSERT INTO analysis_runs(id,model_name) VALUES (%s,'muq_mulan')", (run,)
        )
        for track, source, lib in [
            (owned, "owned", library),
            (private, "private", library),
            (other_server, "owned", foreign_lib),
        ]:
            connection.execute(
                "INSERT INTO tracks VALUES (%s,%s,'artist','album',180)", (track, source)
            )
            connection.execute(
                "INSERT INTO track_sources VALUES (%s,%s,%s,'subsonic',%s)",
                (lib, source, track, Jsonb({"artistId": "artist", "artist": "Artist"})),
            )
            connection.execute(
                "INSERT INTO current_embeddings VALUES (%s,%s,'audio-track',NULL,%s)",
                (track, run, "[" + ",".join(map(str, vector(512, 1))) + "]"),
            )
            connection.execute("INSERT INTO lyrics VALUES (%s,%s,'en','{}')", (track, source))
        connection.execute(
            "INSERT INTO user_source_memberships VALUES (%s,%s,'owned'),(%s,%s,'private'),(%s,%s,'owned')",
            (db.owner, library, db.other, library, db.owner, foreign_lib),
        )
    corpus = ranking.load_corpus(principal)
    assert [r["id"] for r in corpus.rows] == [owned]
    assert lyrics.load_lyrics(principal, "private") == {"lyrics": []}
    assert "owned" in lyrics.load_lyrics(principal, "owned")["lyrics"][0]["text"]
    response = db.client.post(
        "/integrations/navidrome/v1/similar-tracks",
        headers={"Authorization": "Bearer " + token},
        json={"song": {"id": "private"}},
    )
    assert response.status_code == 404
    response = db.client.post(
        "/integrations/navidrome/v1/similar-tracks",
        headers={"Authorization": "Bearer " + token},
        json={"song": {"id": "owned"}, "count": 10000},
    )
    assert response.status_code == 422


def seed_saved_lyrics(db):
    from psycopg.types.json import Jsonb

    principal = auth.authenticate("Bearer " + enable(db))
    library, track = uuid4(), uuid4()
    with db.connect() as connection:
        connection.execute(
            "INSERT INTO libraries VALUES (%s,%s)", (library, ranking.library_namespace(principal))
        )
        connection.execute("INSERT INTO tracks VALUES (%s,'Song','Artist','Album',180)", (track,))
        connection.execute(
            "INSERT INTO track_sources VALUES (%s,'song',%s,'subsonic','{}')", (library, track)
        )
        connection.execute(
            "INSERT INTO user_source_memberships VALUES (%s,%s,'song')", (db.owner, library)
        )
        connection.execute(
            "INSERT INTO lyrics VALUES (%s,'Saved AI transcript','en',%s)",
            (track, Jsonb({"ai_generated": True})),
        )
    return principal, library, track


def insert_sync(
    db, *, connection_id=None, owner=None, kind="navidrome_sync", status="queued", operation=None
):
    from psycopg.types.json import Jsonb

    job_id = uuid4()
    with db.connect() as connection:
        connection.execute(
            """INSERT INTO jobs(id,kind,worker_type,user_id,connection_id,status,payload)
            VALUES (%s,%s,'analysis',%s,%s,%s,%s)""",
            (
                job_id,
                kind,
                owner or db.owner,
                connection_id or db.connection,
                status,
                Jsonb({"operation": operation} if operation else {}),
            ),
        )
    return job_id


@pytest.mark.parametrize("terminal", ["complete", "partial", "failed", "cancelled"])
def test_sync_pause_restores_saved_preference_on_every_terminal_state(
    database, terminal, monkeypatch
):
    db = database
    principal, _, _ = seed_saved_lyrics(db)
    job = insert_sync(db)
    assert db.client.get("/settings/integrations/navidrome").json()["lyrics_paused"] is True
    assert lyrics.load_lyrics(principal, "song") == {"lyrics": []}
    # The key and discovery API stay usable while only lyrics fall through.
    monkeypatch.setattr(
        ranking,
        "load_corpus",
        lambda _: ranking.corpus_from_rows(
            [row("song", "a", [1, 0]), row("neighbor", "b", [1, 0])]
        ),
    )
    with db.connect() as connection:
        key = auth.issue_key(connection, db.owner, db.connection, "Test", 1)
    headers = {"Authorization": "Bearer " + key["secret"]}
    response = db.client.post(
        "/integrations/navidrome/v1/lyrics", headers=headers, json={"track": {"id": "song"}}
    )
    assert response.status_code == 200 and response.json() == {"lyrics": []}
    assert db.client.post(
        "/integrations/navidrome/v1/similar-tracks", headers=headers, json={"song": {"id": "song"}}
    ).json()["matches"]
    with db.connect() as connection:
        connection.execute("UPDATE jobs SET status=%s WHERE id=%s", (terminal, job))
    settings = db.client.get("/settings/integrations/navidrome").json()
    assert settings["serve_lyrics"] is True and settings["lyrics_paused"] is False
    assert "Saved AI transcript" in lyrics.load_lyrics(principal, "song")["lyrics"][0]["text"]


def test_pause_covers_overlapping_accounts_batches_and_expired_worker_leases(database):
    db = database
    principal, _, _ = seed_saved_lyrics(db)
    same_server, foreign_server = uuid4(), uuid4()
    with db.connect() as connection:
        connection.execute(
            "INSERT INTO navidrome_connections VALUES (%s,%s,'http://navidrome/'),(%s,%s,'http://different-server')",
            (same_server, db.other, foreign_server, db.other),
        )
    foreign = insert_sync(db, connection_id=foreign_server, owner=db.other)
    assert lyrics.load_lyrics(principal, "song")["lyrics"]
    first = insert_sync(db, status="waiting")
    second = insert_sync(
        db,
        connection_id=same_server,
        owner=db.other,
        kind="analysis_batch",
        status="running",
        operation="lyrics_backfill",
    )
    with db.connect() as connection:
        connection.execute(
            "UPDATE jobs SET lease_until=now()-interval '1 hour' WHERE id=%s", (second,)
        )
        connection.execute("UPDATE jobs SET status='complete' WHERE id=%s", (first,))
    # A lost worker stays paused through retry/recovery, and a second owner is covered.
    assert lyrics.load_lyrics(principal, "song") == {"lyrics": []}
    with db.connect() as connection:
        connection.execute("UPDATE jobs SET status='failed' WHERE id=%s", (second,))
    assert lyrics.load_lyrics(principal, "song")["lyrics"]
    assert foreign


def test_manual_enable_is_blocked_and_off_preference_is_not_restored_as_on(database):
    db = database
    principal, _, _ = seed_saved_lyrics(db)
    body = {"enabled": True, "connection_id": str(db.connection), "serve_lyrics": False}
    assert db.client.put("/settings/integrations/navidrome", json=body).status_code == 200
    job = insert_sync(db)
    assert (
        db.client.put(
            "/settings/integrations/navidrome", json={**body, "serve_lyrics": True}
        ).status_code
        == 409
    )
    # Other settings can be saved while paused, including a preference already on.
    assert (
        db.client.put(
            "/settings/integrations/navidrome", json={**body, "musical_weight": 65}
        ).status_code
        == 200
    )
    with db.connect() as connection:
        connection.execute("UPDATE jobs SET status='cancelled' WHERE id=%s", (job,))
    settings = db.client.get("/settings/integrations/navidrome").json()
    assert not settings["serve_lyrics"] and not settings["lyrics_paused"]
    principal["profile"] = auth.RankingProfile(serve_lyrics=False)
    assert lyrics.load_lyrics(principal, "song") == {"lyrics": []}


def test_original_snapshots_keep_full_structure_last_available_and_enriched_lyrics(database):
    from echora_analysis.navidrome_lyrics_sources import store_original

    db = database
    _, library, track = seed_saved_lyrics(db)
    result = {
        "status": "available",
        "text": "Original lyrics",
        "source": "getLyricsBySongId",
        "structured_lyrics": [
            {
                "kind": "main",
                "lang": "en",
                "line": [
                    {
                        "value": "Original lyrics",
                        "start": 1000,
                        "end": 3000,
                        "cue": [{"value": "Original", "start": 1000, "end": 2000}],
                    }
                ],
            }
        ],
    }
    with db.connect() as connection:
        store_original(connection, library, "song", track, result)
        saved = connection.execute("SELECT * FROM navidrome_lyrics_sources").fetchone()
        assert saved["result"] == result and saved["last_available"] == result
        store_original(connection, library, "song", track, {"status": "unavailable", "text": None})
        saved = connection.execute("SELECT * FROM navidrome_lyrics_sources").fetchone()
        assert saved["result"]["status"] == "unavailable" and saved["last_available"] == result
        enriched = connection.execute("SELECT * FROM lyrics WHERE track_id=%s", (track,)).fetchone()
        assert enriched["text"] == "Saved AI transcript" and enriched["provenance"]["ai_generated"]
        foreign = uuid4()
        connection.execute("INSERT INTO libraries VALUES (%s,%s)", (foreign, uuid4()))
        store_original(
            connection,
            foreign,
            "song",
            track,
            {"status": "available", "text": "Different server original"},
        )
        assert (
            connection.execute("SELECT count(*) AS n FROM navidrome_lyrics_sources").fetchone()["n"]
            == 2
        )


def test_import_archives_provider_before_preserving_manual_lyrics_without_model_imports():
    """Execute the actual ingestion function with its GPU dependencies replaced."""
    import ast
    from contextlib import nullcontext
    from unittest.mock import Mock
    from collections.abc import Callable

    path = Path(__file__).parents[1] / "src/echora_analysis/lyrics_pipeline.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "backfill_lyrics"
    )
    connection, cursor, client = Mock(), Mock(), Mock()
    connection.cursor.side_effect = lambda: nullcontext(cursor)
    track, library = uuid4(), uuid4()
    cursor.fetchall.return_value = [(track, "source-song", "Song")]
    cursor.fetchone.return_value = ("Manual authoritative text", {}, "available", "manual")
    original = {
        "text": "Original Navidrome text",
        "status": "available",
        "source": "getLyricsBySongId",
    }
    client.lyrics.return_value = original
    events = []
    archived = Mock(side_effect=lambda *args: events.append("archive"))
    model = Mock()
    embedding_write = Mock(side_effect=lambda *args: events.append("embed"))
    globals_ = {
        "Callable": Callable,
        "uuid": __import__("uuid"),
        "psycopg": SimpleNamespace(connect=lambda _: nullcontext(connection)),
        "NavidromeClient": lambda *args: nullcontext(client),
        "get_settings": lambda: SimpleNamespace(
            database_url="unused", lyrics_model_id="unused", lyrics_revision="unused"
        ),
        "resolve_library_id": lambda *args: library,
        "configure_representations": Mock(),
        "plan_lyrics": lambda *args, **kwargs: SimpleNamespace(lyrics_external_ids={"source-song"}),
        "store_original": archived,
        "_store_lyrics": Mock(),
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        "LyricsEmbeddingModel": lambda *args: model,
        "_create_run": Mock(),
        "start_attempt": Mock(),
        "record_track": Mock(),
        "finish_attempt": Mock(),
        "release_model": Mock(),
        "_store_embeddings": embedding_write,
        "current_remote": lambda: None,
        "separation_phase": nullcontext,
        "SongTranscriber": Mock(),
        "needs_repair": lambda draft: False,
        "diagnostic_writer": Mock(),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), globals_)
    result = globals_["backfill_lyrics"]("http://navidrome", "user", "password")
    assert result["failed"] == 0 and result["embedded"] == 1
    assert events == ["archive", "embed"]
    archived.assert_called_once_with(connection, library, "source-song", track, original)
    model.embed.assert_called_once_with("Manual authoritative text")
    globals_["_store_lyrics"].assert_not_called()
    client.lyrics.assert_called_once_with("source-song")
