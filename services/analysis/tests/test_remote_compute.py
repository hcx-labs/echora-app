"""Analysis on Modal: deployment checks, sessions and every stage's Modal path, without Modal."""

from contextlib import nullcontext
from hashlib import sha256
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import httpx
import numpy as np
import pytest

from echora_analysis import external_processing, features, remote_compute
from echora_analysis.download_models import model_manifest
from echora_analysis.models import EmbeddingResult
from echora_analysis.settings import (
    PROCESSING_FIELDS,
    adopt_processing_settings,
    get_settings,
    processing_settings,
)

CONFIG = remote_compute.ModalConfig(
    "ak-test", "as-secret", "L40S", "ghcr.io/x/echora-analysis-gpu:1.0"
)


class NotFound(Exception):
    pass


class FakeModal:
    """The parts of the Modal client API a session uses."""

    def __init__(self, status=None, prepare_events=(), after_prepare=None):
        self.exception = SimpleNamespace(NotFoundError=NotFound)
        self.Client = SimpleNamespace(from_credentials=Mock(return_value="client"))
        self.volume = MagicMock()
        self.Volume = SimpleNamespace(from_name=Mock(return_value=self.volume))
        self.statuses = list(status or [])
        self.prepare = Mock()
        self.prepare.remote_gen.side_effect = lambda settings, hf_token=None: iter(prepare_events)
        self.analysis = Mock()
        self.Cls = SimpleNamespace(from_name=Mock(return_value=Mock(return_value=self.analysis)))
        self.Function = SimpleNamespace(from_name=Mock(side_effect=self._function))

    def _function(self, app, name, client):
        assert app == "echora-analysis" and client == "client"
        if name == "prepare_models":
            return self.prepare
        status = Mock()

        def remote():
            value = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            if value is None:
                raise NotFound()
            return value

        status.remote.side_effect = remote
        return status


def current_status(**changes):
    status = {
        "code": remote_compute.code_identity(),
        "deployment": CONFIG.deployment,
        "models": model_manifest(),
    }
    return {**status, **changes}


def ready_session(modal=None):
    modal = modal or FakeModal([current_status()])
    session = remote_compute.ModalSession(CONFIG, processing_settings(), modal_api=modal)
    session.ensure_ready()
    return session, modal


# Locations and sessions


def test_local_batches_open_no_session():
    with remote_compute.session("local") as remote:
        assert remote is None and remote_compute.current() is None


def test_modal_batches_expose_and_close_the_session():
    remote = Mock()
    with remote_compute.session("modal", factory=lambda: remote) as opened:
        assert opened is remote and remote_compute.current() is remote
    assert remote_compute.current() is None
    remote.close.assert_called_once()


def test_jobs_compute_in_one_location():
    assert remote_compute.location() == "local"
    with remote_compute.computing_on("modal"):
        assert remote_compute.location() == "modal"
    assert remote_compute.location() == "local"
    with pytest.raises(ValueError):
        with remote_compute.computing_on("elsewhere"):
            pass


def test_feature_switches_follow_the_job_location():
    cursor = MagicMock()
    cursor.fetchone.return_value = {
        "karaoke_processing_enabled": False,
        "karaoke_modal_enabled": True,
    }
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    assert not features.feature_enabled(connection, "karaoke")
    with remote_compute.computing_on("modal"):
        assert features.feature_enabled(connection, "karaoke")
    assert "karaoke_modal_enabled" in cursor.execute.call_args.args[0]
    cursor.fetchone.return_value = None
    assert features.feature_enabled(connection, "karaoke") and not features.feature_enabled(
        connection, "transcription"
    )


# Deployment


def test_ready_deployment_is_used_without_deploying(monkeypatch):
    deploy = Mock()
    monkeypatch.setattr(remote_compute, "deploy", deploy)
    session, modal = ready_session()
    deploy.assert_not_called()
    modal.prepare.remote_gen.assert_not_called()
    assert session.analysis is modal.analysis


def test_missing_or_stale_code_is_deployed_first(monkeypatch):
    for before in (
        None,
        current_status(code="old"),
        current_status(deployment={"image": "old", "gpu": "L40S"}),
    ):
        deploy = Mock()
        monkeypatch.setattr(remote_compute, "deploy", deploy)
        modal = FakeModal([before, current_status()])
        session = remote_compute.ModalSession(CONFIG, processing_settings(), modal_api=modal)
        assert session.ensure_ready()["deployed"]
        deploy.assert_called_once_with(CONFIG)


def test_deploy_waits_for_old_containers_to_roll_over(monkeypatch):
    monkeypatch.setattr(remote_compute, "deploy", Mock())
    monkeypatch.setattr(remote_compute, "ROLLOVER_POLL_SECONDS", 0)
    # Right after deploying, a still-warm old container answers once more.
    modal = FakeModal([current_status(code="old"), current_status(code="old"), current_status()])
    session = remote_compute.ModalSession(CONFIG, processing_settings(), modal_api=modal)
    assert session.ensure_ready()["deployed"]


def test_deployment_that_still_differs_is_refused(monkeypatch):
    monkeypatch.setattr(remote_compute, "ROLLOVER_SECONDS", 0)
    monkeypatch.setattr(remote_compute, "deploy", Mock())
    modal = FakeModal([None, current_status(code="other")])
    with pytest.raises(RuntimeError, match="does not match"):
        remote_compute.ModalSession(CONFIG, processing_settings(), modal_api=modal).ensure_ready()


def test_changed_models_are_downloaded_with_progress(monkeypatch):
    monkeypatch.setattr(remote_compute, "deploy", Mock())
    events = [
        {"completed": 0, "total": 2, "label": "model-a"},
        {"completed": 1, "total": 2, "label": "model-b"},
        {"completed": 2, "total": 2, "manifest": model_manifest()},
    ]
    modal = FakeModal([current_status(models="old"), current_status()], prepare_events=events)
    progress = []
    session = remote_compute.ModalSession(CONFIG, processing_settings(), modal_api=modal)
    assert session.ensure_ready(progress.append)["downloaded"]
    modal.prepare.remote_gen.assert_called_once_with(processing_settings(), None)
    assert [item["message"] for item in progress if item.get("unit") == "models"] == [
        "Preparing models on Modal: model-a",
        "Preparing models on Modal: model-b",
    ]


def test_incomplete_model_storage_is_refused(monkeypatch):
    modal = FakeModal([current_status(models="old")])
    with pytest.raises(RuntimeError, match="incomplete"):
        remote_compute.ModalSession(CONFIG, processing_settings(), modal_api=modal).ensure_ready()


def test_compute_requires_a_checked_deployment():
    session = remote_compute.ModalSession(
        CONFIG, processing_settings(), modal_api=FakeModal([current_status()])
    )
    with pytest.raises(RuntimeError, match="Check the Modal deployment"):
        session.voice(["digest"])


def test_cli_uses_only_the_stored_token(monkeypatch):
    monkeypatch.setenv("MODAL_PROFILE", "personal")
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-someone-else")
    environment = remote_compute._modal_environment(CONFIG)
    assert (
        environment["MODAL_TOKEN_ID"] == "ak-test"
        and environment["MODAL_TOKEN_SECRET"] == "as-secret"
    )
    assert "MODAL_PROFILE" not in environment
    assert (
        environment["ECHORA_MODAL_IMAGE"] == CONFIG.image
        and environment["ECHORA_MODAL_GPU"] == "L40S"
    )


def test_cli_errors_never_contain_the_token(monkeypatch):
    failed = SimpleNamespace(returncode=1, stdout="", stderr="Bad token as-secret for ak-test")
    monkeypatch.setattr(remote_compute.subprocess, "run", Mock(return_value=failed))
    with pytest.raises(RuntimeError) as raised:
        remote_compute.workspace(CONFIG)
    assert "as-secret" not in str(raised.value) and "ak-test" not in str(raised.value)


def test_workspace_comes_from_token_info(monkeypatch):
    info = SimpleNamespace(
        returncode=0, stdout="Token: ak-test\nWorkspace: hackslashx (ac-123)\n", stderr=""
    )
    monkeypatch.setattr(remote_compute.subprocess, "run", Mock(return_value=info))
    assert remote_compute.workspace(CONFIG) == "hackslashx"


# Settings and models


def test_remote_containers_adopt_the_workers_processing_settings(monkeypatch):
    for name in PROCESSING_FIELDS:
        # Set first, so monkeypatch restores whatever adopting writes to the environment.
        monkeypatch.setenv(name.upper(), str(processing_settings()[name]))
    values = {
        **processing_settings(),
        "fa_kara_revision": "a" * 40,
        "fa_kara_min_letter_ms": 60,
        "fa_kara_audio_speed": 0.9,
    }
    adopt_processing_settings(values)
    assert processing_settings() == values
    assert get_settings().fa_kara_min_letter_ms == 60
    with pytest.raises(ValueError, match="Not processing settings"):
        adopt_processing_settings({"database_url": "postgresql://elsewhere"})


def test_model_manifest_follows_the_model_settings(monkeypatch):
    before = model_manifest()
    monkeypatch.setenv("FA_KARA_REVISION", "b" * 40)
    get_settings.cache_clear()
    assert model_manifest() != before


# Compute choice and preparation


@pytest.mark.real_compute_choice
def test_compute_choice_respects_permission_and_default(monkeypatch):
    row = {
        "enabled": True,
        "token_id": "ak-x",
        "token_secret_encrypted": b"secret",
        "allow_users": False,
        "default_compute": "modal",
    }
    monkeypatch.setattr(external_processing, "_connect", lambda: nullcontext(MagicMock()))
    monkeypatch.setattr(external_processing, "load", lambda db=None: row)
    admin = {"value": True}
    monkeypatch.setattr(external_processing, "_is_admin", lambda db, user: admin["value"])
    assert external_processing.compute_for("user", None) == "modal"
    assert external_processing.compute_for("user", "local") == "local"
    admin["value"] = False
    assert external_processing.compute_for("user", None) == "local"
    with pytest.raises(PermissionError):
        external_processing.compute_for("user", "modal")
    row["allow_users"] = True
    assert external_processing.compute_for("user", "modal") == "modal"
    row["enabled"] = False
    assert external_processing.compute_for("user", None) == "local"


def preparation_database(monkeypatch):
    monkeypatch.setattr(external_processing, "remote_settings", processing_settings)
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = {"locked": True}
    monkeypatch.setattr(external_processing, "_connect", lambda: nullcontext(db))
    monkeypatch.setattr(external_processing, "config", lambda row=None: CONFIG)
    recorded = []
    monkeypatch.setattr(
        external_processing, "_record", lambda db, **values: recorded.append(values)
    )
    return db, recorded


def test_preparation_records_ready_state_under_a_lock(monkeypatch):
    db, recorded = preparation_database(monkeypatch)
    session = Mock()
    session.ensure_ready.return_value = {"code": "code", "models": "models"}
    factory = Mock(return_value=session)
    assert external_processing.prepare(session_factory=factory) is session
    factory.assert_called_once_with(CONFIG, processing_settings())
    assert recorded[0]["status"] == "preparing" and recorded[-1]["status"] == "ready"
    assert recorded[-1]["deployed_code"] == "code" and recorded[-1]["deployed_models"] == "models"
    statements = [call.args[0] for call in db.execute.call_args_list]
    assert any("pg_try_advisory_lock" in s for s in statements) and any(
        "pg_advisory_unlock" in s for s in statements
    )


def test_failed_preparation_is_recorded_and_raised(monkeypatch):
    _, recorded = preparation_database(monkeypatch)
    session = Mock()
    session.ensure_ready.side_effect = RuntimeError("Modal: quota exceeded")
    with pytest.raises(RuntimeError):
        external_processing.prepare(session_factory=Mock(return_value=session))
    assert recorded[-1]["status"] == "failed" and "quota" in recorded[-1]["status_detail"]


def test_settings_reject_malformed_tokens_and_gpus():
    with pytest.raises(ValueError):
        external_processing.SettingsUpdate(token_id="as-not-an-id")
    with pytest.raises(ValueError):
        external_processing.SettingsUpdate(gpu="RTX9000")


# Session calls


def test_audio_is_uploaded_once_and_removed_at_the_end():
    session, modal = ready_session()
    audio = b"source audio"
    digest = sha256(audio).hexdigest()
    session.upload(digest, audio)
    assert session.upload_audio(audio) == digest
    assert modal.volume.batch_upload.call_count == 1
    with pytest.raises(ValueError, match="content hash"):
        session.upload("0" * 64, audio)
    session.close()
    modal.volume.remove_file.assert_called_once_with(f"/{digest}")


def test_every_call_carries_the_exact_model_and_settings():
    session, modal = ready_session()
    session.embed_audio(remote_compute.RemoteModel("mert", "m-a-p/MERT-v1-95M", "rev"), ["a"])
    args, kwargs = modal.analysis.embed_audio.map.call_args
    assert args == (["a"],) and kwargs["kwargs"] == {
        "model": ("mert", "m-a-p/MERT-v1-95M", "rev"),
        "settings": processing_settings(),
    }
    session.karaoke(["a"], ["text"], ["ja"], [[]])
    args, kwargs = modal.analysis.karaoke.map.call_args
    assert args == (["a"], ["text"], ["ja"], [[]]) and kwargs["kwargs"] == {
        "settings": processing_settings()
    }
    assert kwargs["order_outputs"] and kwargs["return_exceptions"]


def test_transcription_relays_progress_and_diagnostics():
    session, modal = ready_session()
    modal.analysis.transcribe_generate.remote_gen.return_value = iter(
        [{"progress": "window 1"}, {"diagnostic": {"start": 0}}, {"draft": {"repairs": {}}}]
    )
    progress, diagnostics = [], []
    draft = session.transcribe_generate(
        b"audio",
        language="ja",
        vocal_activity=None,
        progress=progress.append,
        diagnostic_sink=diagnostics.append,
        check=lambda: None,
    )
    assert draft == {"repairs": {}} and progress == ["window 1"] and diagnostics == [{"start": 0}]
    modal.analysis.transcribe_generate.remote_gen.return_value = iter([{"progress": "window 1"}])
    with pytest.raises(RuntimeError, match="without a result"):
        session.transcribe_generate(
            b"audio",
            language=None,
            vocal_activity=None,
            progress=print,
            diagnostic_sink=print,
            check=lambda: None,
        )


def test_separation_phase_asks_for_exactly_the_stages_artifacts():
    session, modal = ready_session()
    modal.analysis.prepare_vocals.map.return_value = iter([None, None])
    assert list(session.prepare_vocals(["a", "b"], vocals=True, reference=True)) == [None, None]
    args, kwargs = modal.analysis.prepare_vocals.map.call_args
    assert args == (["a", "b"],)
    assert kwargs["kwargs"] == {
        "artifacts": {"vocals": True, "reference": True},
        "settings": processing_settings(),
    }
    assert list(session.prepare_vocals([], melody=True)) == []


# Stages


@pytest.fixture
def ingest_batch(monkeypatch):
    from echora_analysis import ingest

    connection = MagicMock()
    connection.__enter__.return_value = connection
    client = MagicMock()
    client.__enter__.return_value = client
    for name, value in {
        "configure_representations": Mock(),
        "_library": Mock(return_value=uuid4()),
        "active_cache": lambda: None,
        "get_check": lambda: lambda: None,
        "_create_run": Mock(return_value=uuid4()),
        "_model_has_embedding": Mock(return_value=False),
        "start_attempt": Mock(),
        "record_track": Mock(),
        "finish_attempt": Mock(),
        "_store_embedding": Mock(),
        "plan_audio": Mock(),
        "create_sync_run": Mock(return_value=uuid4()),
        "store_track_contours": Mock(return_value=2),
    }.items():
        monkeypatch.setattr(ingest, name, value)
    monkeypatch.setattr(ingest.psycopg, "connect", Mock(return_value=connection))
    monkeypatch.setattr(ingest, "NavidromeClient", Mock(return_value=client))
    monkeypatch.setattr(ingest.torch.cuda, "is_available", lambda: False)
    # A Modal batch must never load an embedding model locally.
    for name in ("MuQMuLanModel", "MertModel"):
        loader = Mock(side_effect=AssertionError("loaded a local model in a Modal batch"))
        loader.name = getattr(ingest, name).name
        monkeypatch.setattr(ingest, name, loader)
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    get_settings.cache_clear()
    tracks = {name: uuid4() for name in ("a", "bad", "c")}
    audio = {name: f"audio {name}".encode() for name in tracks}
    client.tracks.return_value = [
        SimpleNamespace(
            id=n, title=n, artist="Artist", album="Album", duration=2, year=None, genre=None, raw={}
        )
        for n in tracks
    ]
    client.audio_bytes.side_effect = lambda song_id: audio[song_id]
    monkeypatch.setattr(
        ingest, "_upsert_track", Mock(side_effect=lambda _c, _l, song, _h: (tracks[song.id], False))
    )
    return ingest, tracks, audio


def failing(audio, name, value):
    bad = sha256(audio["bad"]).hexdigest()
    return lambda *args: [
        RuntimeError("remote failure") if digest == bad else value for digest in args[-1]
    ]


def test_modal_batch_embeds_remotely_and_stores_each_result(ingest_batch):
    from echora_analysis.processing_plan import AudioProcessingPlan

    ingest, tracks, audio = ingest_batch
    ingest.plan_audio.return_value = AudioProcessingPlan(
        frozenset(tracks), frozenset(), frozenset(), frozenset()
    )
    result = EmbeddingResult(np.ones(4, np.float32), 5, None, (np.ones(4, np.float32),))
    remote = Mock()
    remote.embed_audio.side_effect = failing(audio, "bad", (result, [(0.0, 10.0)]))
    with remote_compute.session("modal", factory=lambda: remote):
        summary = ingest.ingest_navidrome("url", "user", "password", list(tracks))
    assert {call.args[0] for call in remote.upload.call_args_list} == {
        sha256(a).hexdigest() for a in audio.values()
    }
    model = remote.embed_audio.call_args.args[0]
    assert isinstance(model, remote_compute.RemoteModel) and model.name == "muq_mulan"
    assert [call.args[1] for call in ingest._store_embedding.call_args_list] == [
        tracks["a"],
        tracks["c"],
    ]
    assert ingest._store_embedding.call_args.args[3:] == (result, [(0.0, 10.0)])
    assert summary.embedded_muq == 2 and summary.failed == 1


def test_modal_batch_extracts_melody_remotely(ingest_batch):
    from echora_analysis.processing_plan import AudioProcessingPlan

    ingest, tracks, audio = ingest_batch
    ingest.plan_audio.return_value = AudioProcessingPlan(
        frozenset(), frozenset(), frozenset(), frozenset(tracks)
    )
    contours = {"vocals": (np.ones(40), np.ones(40, bool))}
    remote = Mock()
    remote.prepare_vocals.side_effect = lambda digests, **artifacts: [None for _ in digests]
    remote.melody.side_effect = failing(audio, "bad", contours)
    with remote_compute.session("modal", factory=lambda: remote):
        summary = ingest.ingest_navidrome("url", "user", "password", list(tracks))
    stored = ingest.store_track_contours.call_args_list
    assert [call.args[1] for call in stored] == [tracks["a"], tracks["c"]]
    assert all(call.kwargs["usable"] is contours for call in stored)
    assert summary.melody_indexed == 2 and summary.failed == 1
    # One separation phase for the melody songs, before any contour call.
    assert len(remote.prepare_vocals.call_args.args[0]) == 3
    assert remote.prepare_vocals.call_args.kwargs == {"melody": True}


def database(rows):
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    return connection, cursor


def test_modal_batch_classifies_voice_remotely(monkeypatch):
    from echora_analysis import voice_pipeline

    rows = [(uuid4(), "a", "A"), (uuid4(), "b", "B")]
    connection, _ = database(rows)
    client = MagicMock()
    client.__enter__.return_value = client
    client.audio_bytes.side_effect = lambda source: f"audio {source}".encode()
    for name, value in {
        "resolve_library_id": Mock(),
        "configure_representations": Mock(),
        "_create_run": Mock(return_value=uuid4()),
        "_store_activation": Mock(),
        "NavidromeClient": Mock(return_value=client),
        "shared_voice_model": Mock(side_effect=AssertionError("local model in Modal batch")),
    }.items():
        monkeypatch.setattr(voice_pipeline, name, value)
    monkeypatch.setattr(voice_pipeline.psycopg, "connect", Mock(return_value=connection))
    remote = Mock()
    remote.upload_audio.side_effect = lambda audio: sha256(audio).hexdigest()
    remote.voice.side_effect = lambda digests: [
        ({"female": 1.0}, {"windows": []}),
        RuntimeError("failed"),
    ]
    with remote_compute.session("modal", factory=lambda: remote):
        summary = voice_pipeline.backfill_voice("url", "user", "password")
    assert summary == {"total": 2, "classified": 1, "failed": 1}
    voice_pipeline._store_activation.assert_called_once()
    assert voice_pipeline._store_activation.call_args.args[1] == rows[0][0]


def test_modal_batch_aligns_karaoke_remotely(monkeypatch):
    from echora_analysis import karaoke_pipeline

    rows = [
        (uuid4(), "a", "A", "lyrics a", "ja", [{"text": "a"}]),
        (uuid4(), "b", "B", "lyrics b", None, None),
    ]
    connection, cursor = database(rows)
    client = MagicMock()
    client.__enter__.return_value = client
    client.audio_bytes.side_effect = lambda source: f"audio {source}".encode()
    for name, value in {
        "resolve_library_id": Mock(),
        "NavidromeClient": Mock(return_value=client),
        "plan_karaoke": Mock(return_value=SimpleNamespace(karaoke_external_ids={"a", "b"})),
        "prepare_audio": Mock(side_effect=AssertionError("local separation in Modal batch")),
        "_run_fa_kara": Mock(side_effect=AssertionError("local alignment in Modal batch")),
        "_stop_fa_kara_worker": Mock(),
    }.items():
        monkeypatch.setattr(karaoke_pipeline, name, value)
    monkeypatch.setattr(karaoke_pipeline.psycopg, "connect", Mock(return_value=connection))
    aligned = {
        "lines": [],
        "ass": "",
        "lrc": "",
        "alignment_document": {},
        "model": "m",
        "model_revision": "r",
        "diagnostics": {
            "audio_source": "roformer_vocals",
            "reference_audio_source": "full_mix",
            "separator_revision": "s",
        },
    }
    remote = Mock()
    remote.upload_audio.side_effect = lambda audio: sha256(audio).hexdigest()
    remote.prepare_vocals.side_effect = lambda digests, **artifacts: [None for _ in digests]
    remote.karaoke.side_effect = lambda digests, texts, languages, lines: [
        aligned,
        RuntimeError("failed"),
    ]
    with remote_compute.session("modal", factory=lambda: remote):
        summary = karaoke_pipeline.backfill_karaoke("url", "user", "password")
    assert summary == {"total": 2, "aligned": 1, "failed": 1}
    digests, texts, languages, lines = remote.karaoke.call_args.args
    assert (
        texts == ["lyrics a", "lyrics b"]
        and languages == ["ja", None]
        and lines == [[{"text": "a"}], []]
    )
    assert digests == [sha256(b"audio a").hexdigest(), sha256(b"audio b").hexdigest()]
    # Separated first, for exactly the songs being aligned, with what alignment reads.
    assert remote.prepare_vocals.call_args.args[0] == digests
    assert remote.prepare_vocals.call_args.kwargs == {"vocals": True}


# Motion artwork


def test_motion_artwork_models_count_only_when_modal_renders_them():
    assert model_manifest(True) != model_manifest(False)
    session = remote_compute.ModalSession(
        CONFIG,
        {**processing_settings(), "motion_artwork_models": True},
        modal_api=FakeModal([current_status()]),
    )
    assert session.expected()["models"] == model_manifest(True)


def test_remote_containers_adopt_the_motion_artwork_model_switch(monkeypatch):
    monkeypatch.setenv("ECHORA_MOTION_ARTWORK_MODELS", "false")
    adopt_processing_settings({"motion_artwork_models": True})
    assert get_settings().motion_artwork_models is True


def test_hugging_face_token_goes_only_to_model_preparation(monkeypatch):
    from dataclasses import replace

    config = replace(CONFIG, hf_token="hf_secret")
    modal = FakeModal([current_status(models="old"), current_status()])
    remote_compute.ModalSession(config, processing_settings(), modal_api=modal).ensure_ready()
    modal.prepare.remote_gen.assert_called_once_with(processing_settings(), "hf_secret")
    assert "hf_secret" not in remote_compute._modal_environment(config).values()
    assert "hf_secret" not in remote_compute._scrubbed("token hf_secret", config)


def test_motion_artwork_videos_come_back_through_the_transfer_volume(tmp_path):
    from echora_analysis.motion_artwork import MotionArtworkSettings

    session, modal = ready_session()
    artwork = Mock()
    artwork.render.remote_gen.return_value = iter(
        [
            {"progress": {"phase": "motion-artwork"}},
            {
                "rendered": "abc",
                "transfer": "/artwork/abc-1.mp4",
                "size": [512, 512],
                "frames": 192,
                "prompt": "p",
            },
            {"timings": {"video_seconds": 1}},
        ]
    )
    modal.Cls.from_name = Mock(return_value=Mock(return_value=artwork))
    modal.volume.read_file.return_value = iter([b"mp4-", b"bytes"])
    settings = MotionArtworkSettings(enabled=True, comfyui_url="http://comfy.local:8188")
    covers = [
        {
            "sha256": "abc",
            "data": b"cover",
            "content_type": "image/jpeg",
            "seed": 1,
            "track_id": "not sent",
            "cover_art_id": "not sent",
        }
    ]
    events = list(session.motion_artwork(settings, "recipe", covers))
    sent_settings, recipe, sent_covers, sent_processing = artwork.render.remote_gen.call_args.args
    assert (
        sent_settings.comfyui_url == ""
        and recipe == "recipe"
        and sent_processing == processing_settings()
    )
    assert sent_covers == [
        {"sha256": "abc", "data": b"cover", "content_type": "image/jpeg", "seed": 1}
    ]
    rendered = events[1]
    assert "transfer" not in rendered and Path(rendered["file"]).read_bytes() == b"mp4-bytes"
    modal.volume.remove_file.assert_called_with("/artwork/abc-1.mp4")
    Path(rendered["file"]).unlink()


def test_comfyui_finds_models_wherever_the_cache_lives(monkeypatch):
    from echora_analysis.comfyui import model_paths_config

    monkeypatch.setenv("HF_HOME", "/echora-models/huggingface")
    config = model_paths_config()
    assert (
        "base_path: /echora-models/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/" in config
    )
    assert (
        "base_path: /echora-models/huggingface/hub/models--Comfy-Org--gemma-4/snapshots/" in config
    )


# Hugging Face token (Settings → Analysis models)


def test_downloads_use_the_saved_token_unless_one_is_given(monkeypatch):
    from echora_analysis import download_models, huggingface_token

    calls = []
    monkeypatch.setattr(huggingface_token, "token_or_none", lambda: "hf_saved")
    for name in ("download_recording_model", "_download_essentia", "_prune_huggingface_cache"):
        monkeypatch.setattr(download_models, name, lambda *args: None)
    monkeypatch.setattr(
        download_models,
        "snapshot_download",
        lambda **kwargs: calls.append(os.environ.get("HF_TOKEN")) or "/tmp/snapshot",
    )
    monkeypatch.setattr(download_models, "_pin_main_ref", lambda path: None)
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    get_settings.cache_clear()
    try:
        download_models.main()
        assert set(calls) == {"hf_saved"}
        calls.clear()
        monkeypatch.setenv("HF_TOKEN", "hf_given")
        download_models.main()
        assert set(calls) == {"hf_given"}
    finally:
        os.environ.pop("HF_TOKEN", None)


def test_saved_token_is_verified_and_encrypted(monkeypatch):
    from cryptography.fernet import Fernet

    from echora_analysis import huggingface_token

    monkeypatch.setenv("CREDENTIAL_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    get_settings.cache_clear()
    db = MagicMock()
    monkeypatch.setattr(huggingface_token.psycopg, "connect", lambda url: nullcontext(db))
    monkeypatch.setattr(
        huggingface_token, "account", lambda token: "hcX02" if token == "hf_good" else 1 / 0
    )
    assert huggingface_token.save(" hf_good ") == "hcX02"
    encrypted, account = db.execute.call_args.args[1]
    assert account == "hcX02" and b"hf_good" not in encrypted
    with pytest.raises(ZeroDivisionError):
        huggingface_token.save("hf_bad")
    assert huggingface_token.save("") is None and db.execute.call_args.args[1] == (None, None)


def test_token_check_reaches_hugging_face_even_when_models_load_offline(monkeypatch):
    from echora_analysis import huggingface_token

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    seen = {}

    def get(url, headers, timeout):
        seen.update(url=url, auth=headers["Authorization"])
        status = 200 if headers["Authorization"] == "Bearer hf_good" else 401
        return httpx.Response(status, json={"name": "hcX02"}, request=httpx.Request("GET", url))

    monkeypatch.setattr(
        huggingface_token.httpx if hasattr(huggingface_token, "httpx") else __import__("httpx"),
        "get",
        get,
    )
    assert huggingface_token.account("hf_good") == "hcX02"
    assert seen == {"url": "https://huggingface.co/api/whoami-v2", "auth": "Bearer hf_good"}
    with pytest.raises(huggingface_token.TokenRejected):
        huggingface_token.account("hf_bad")


def test_batched_results_finish_modals_stream_with_the_last_result():
    finished = []

    def stream():
        yield "a"
        yield "b"
        finished.append(True)

    results = remote_compute._complete(stream(), 2)
    assert next(results) == "a" and not finished
    # The stream ends as the last result is handed over, not at garbage collection.
    assert next(results) == "b" and finished


def test_deploys_replace_old_containers(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="App deployed", stderr=""))
    monkeypatch.setattr(remote_compute.subprocess, "run", run)
    remote_compute.deploy(CONFIG)
    command = run.call_args.args[0]
    assert command[command.index("deploy") :] == [
        "deploy",
        "--strategy",
        "recreate",
        "-m",
        "echora_analysis.modal_app",
    ]


def test_settings_report_a_stored_hugging_face_token_without_decrypting_it(monkeypatch):
    from echora_analysis import huggingface_token

    monkeypatch.setattr(
        huggingface_token, "token_or_none", Mock(side_effect=AssertionError("decrypted"))
    )
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = {"stored": True}
    assert external_processing._has_hf_token(db) is True
    assert "IS NOT NULL" in db.execute.call_args.args[0]
    db.execute.return_value.fetchone.return_value = None
    assert external_processing._has_hf_token(db) is False
