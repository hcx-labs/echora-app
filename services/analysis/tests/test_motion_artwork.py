"""Motion artwork: recipe, graph, post-processing, settings, GPU handoff and album de-duplication."""

import json
import os
import runpy
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
import numpy as np
import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row
from pydantic import ValidationError

from echora_analysis import motion_artwork, motion_artwork_jobs
from echora_analysis.comfyui import ComfyUIUnavailable, launch
from echora_analysis.motion_artwork import MotionArtworkSettings, artwork_path
from echora_analysis.motion_artwork_prompts import CAMERA_RETRY, I2V_SYSTEM_PROMPT
from echora_analysis.motion_artwork_render import (
    CAMERA_LORA,
    CAMERA_LORA_STRENGTH,
    DEFAULT_INSTRUCTIONS,
    NEGATIVE,
    REFINE_SIGMAS,
    UPSCALER,
    Recipe,
    build_encode_graph,
    build_graph,
    build_prompt_graph,
    clean_prompt,
    encoding_file,
    finish,
    zooms,
)

VERSIONS = Path(__file__).resolve().parents[1] / "alembic/versions"


def test_recipe_hash_tracks_only_settings_that_change_the_output():
    base = Recipe()
    assert base.hash() == Recipe().hash()
    assert Recipe(resolution=512).hash() != base.hash()
    assert Recipe(upscale=False).hash() != base.hash()
    assert Recipe(camera_lock=0.7).hash() != base.hash()
    # Instructions are unused in fixed mode and the fixed prompt is unused in auto mode.
    assert Recipe(fixed_prompt="ignored").hash() == base.hash()
    fixed = Recipe(prompt_mode="fixed", fixed_prompt="Rain falls.")
    assert (
        Recipe(prompt_mode="fixed", fixed_prompt="Rain falls.", instructions="other").hash()
        == fixed.hash()
    )


def test_prompt_graph_writes_a_description_from_the_cover():
    graph = build_prompt_graph(Recipe(), "cover.jpg", 7)
    writer = graph["31_write_prompt"]["inputs"]
    assert writer["image"] == ["1_cover", 0]
    assert graph["31_write_prompt"]["class_type"] == "TextGenerate"
    assert not writer["use_default_template"]
    assert writer["prompt"].startswith("<|turn>system\n" + I2V_SYSTEM_PROMPT)
    assert "<|turn>user\n<|image><|image|><image|>" in writer["prompt"]
    assert DEFAULT_INSTRUCTIONS in writer["prompt"]
    assert writer["prompt"].endswith("<|turn>model\n<|channel>final\n")
    assert writer["sampling_mode"] == "on" and writer["sampling_mode.seed"] == 7
    assert graph["32_show_prompt"]["class_type"] == "PreviewAny"
    assert "hidden for the whole clip" in I2V_SYSTEM_PROMPT


def _references_resolve(graph: dict) -> None:
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in graph


def test_video_graph_loops_on_the_cover_with_the_given_prompt():
    graph = build_graph(
        Recipe(upscale=False), "cover.jpg", 7, "prefix", "Wind moves through the grass."
    )
    assert graph["5_positive"]["inputs"]["text"] == "Wind moves through the grass."
    assert graph["8_latent"]["inputs"] == {
        "width": 768,
        "height": 768,
        "length": 241,
        "batch_size": 1,
    }
    # The cover is the first frame and, fully, the last, so the clip loops on it.
    assert graph["10_first_frame"]["inputs"]["image"] == ["9_preprocess", 0]
    end = graph["12_end_frame"]["inputs"]
    assert (end["image"], end["frame_idx"], end["strength"]) == (["1_cover", 0], -1, 1.0)
    assert end["latent"] == ["10_first_frame", 0]
    assert graph["14_guider"]["inputs"]["positive"] == ["12_end_frame", 0]
    assert graph["17_sample"]["inputs"]["latent_image"] == ["12_end_frame", 2]
    assert graph["19_decode"]["inputs"]["samples"] == ["18_crop_guides", 2]
    # The static-camera LoRA drives sampling.
    lora = graph["2b_camera_lora"]["inputs"]
    assert (
        lora["lora_name"] == CAMERA_LORA and lora["strength_model"] == CAMERA_LORA_STRENGTH == 0.5
    )
    firmer = build_graph(Recipe(camera_lock=0.7), "cover.jpg", 7, "prefix", "text")
    assert firmer["2b_camera_lora"]["inputs"]["strength_model"] == 0.7
    assert graph["14_guider"]["inputs"]["model"] == ["2b_camera_lora", 0]
    assert graph["13_noise"]["inputs"]["noise_seed"] == 7
    assert "31_write_prompt" not in graph and "23_upscale" not in graph
    _references_resolve(graph)


def test_upscaling_refines_a_doubled_latent_with_the_cover_pinned_again():
    graph = build_graph(Recipe(), "cover.jpg", 7, "prefix", "text")
    assert graph["8_latent"]["inputs"]["width"] == 768 and Recipe().output_resolution == 1536
    assert graph["23_upscale"]["inputs"]["samples"] == ["18_crop_guides", 2]
    assert graph["22_upscale_model"]["inputs"]["model_name"] == UPSCALER
    assert graph["24_first_frame"]["inputs"]["latent"] == ["23_upscale", 0]
    end = graph["25_end_frame"]["inputs"]
    assert (end["frame_idx"], end["strength"], end["latent"]) == (-1, 1.0, ["24_first_frame", 0])
    refine = graph["29_refine"]["inputs"]
    assert refine["latent_image"] == ["25_end_frame", 2]
    assert graph["28_sigmas"]["inputs"]["sigmas"] == REFINE_SIGMAS
    assert graph["26_noise"]["inputs"]["noise_seed"] == 8
    assert graph["27_guider"]["inputs"]["model"] == ["2b_camera_lora", 0]
    assert graph["19_decode"]["inputs"]["samples"] == ["30_crop_guides", 2]
    _references_resolve(graph)


def test_prompts_are_encoded_ahead_so_rendering_skips_the_text_encoder():
    encode = build_encode_graph("Wind moves through the grass.", "abc123")
    assert encode["6_save"]["inputs"]["filename_prefix"] == "conditioning/abc123"
    assert encoding_file("abc123") == "abc123_00001_.safetensors"
    assert {node["class_type"] for node in encode.values()} == {
        "CLIPLoader",
        "CLIPTextEncode",
        "SaveConditioning",
    }
    graph = build_graph(
        Recipe(),
        "cover.jpg",
        7,
        "prefix",
        encodings=("abc123_00001_.safetensors", "negative.safetensors"),
    )
    assert graph["5_positive"] == {
        "class_type": "ConditioningLoader",
        "inputs": {"conditioning_name": "abc123_00001_.safetensors"},
    }
    assert graph["6_negative"]["inputs"]["conditioning_name"] == "negative.safetensors"
    assert "3_text_encoder" not in graph
    for arguments in ({}, {"prompt": "text", "encodings": ("a", "b")}):
        with pytest.raises(ValueError):
            build_graph(Recipe(), "cover.jpg", 7, "prefix", **arguments)


def test_clean_prompt_cuts_a_repeated_caption():
    caption = (
        "A medium shot frames a young person lying in a meadow of white flowers as the camera "
        "slowly pushes in and returns to its starting framing."
    )
    assert clean_prompt(f"{caption} {caption}") == caption
    assert clean_prompt(f"  {caption}\n\n") == caption
    long = ("A wide shot of a city at night. " * 120).strip()
    cleaned = clean_prompt("Rain falls on the street. " + long, limit=300)
    assert len(cleaned) <= 300 and cleaned.endswith(".")


def test_camera_movement_detection_respects_local_negation():
    for prompt in (
        "the camera executes a very slow, smooth push-in towards the center",
        "as the camera pushes in on her face",
        "a slow zoom toward the sun",
        "the camera pulls back to reveal the city",
        "a gentle dolly forward",
        "the camera moves closer to the flower",
        "The camera does not pan, but slowly zooms in.",
        "The camera does not zoom, then the camera pans right.",
        "No particles appear, and the camera pushes in.",
        "The camera moves gently forward.",
        "The framing slowly tightens around the boat.",
        "The camera does not pan, it zooms in.",
        "The camera holds still, then gently pans left.",
        "The central figure slowly steps forward, then returns.",
        "She leans toward the camera and smiles.",
        "The man walks slowly toward the viewer.",
        "The silhouette moves closer, revealing his face.",
    ):
        assert zooms(prompt), prompt
    for prompt in (
        "a slow sideways slide of the light while confetti drifts",
        "she pushes her hair back",
    ):
        assert not zooms(prompt), prompt
    assert "never zoom" in DEFAULT_INSTRUCTIONS.lower()
    # Any described camera movement counts; negated mentions do not.
    for prompt in (
        "the camera slowly drifts sideways, then eases back without changing distance",
        "the camera pans to the right",
        "as the camera slowly rises over the city",
    ):
        assert zooms(prompt), prompt
    for prompt in (
        "A medium shot with a locked-off camera that does not move.",
        "The camera is locked off and does not move at all: no pan, tilt, drift, zoom or dolly.",
        "The camera stays still, with no pan or push-in.",
        "Clouds drift across the sky.",
        "The camera never zooms.",
        "The camera does not pan, tilt or zoom.",
        "The camera is locked off while leaves move.",
        "Nobody steps forward; the figure sways sideways with feet planted.",
        "He does not lean toward the camera.",
        "Mist drifts toward the foreground while waves roll in.",
    ):
        assert not zooms(prompt), prompt
    assert not zooms(DEFAULT_INSTRUCTIONS)
    assert "ambient motion in the surroundings" in DEFAULT_INSTRUCTIONS


def _clip(path: Path, colors: list[int], size: int = 64) -> None:
    frames = np.stack([np.full((size, size, 3), value, np.uint8) for value in colors])
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{size}x{size}",
            "-r",
            "24",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv444p",
            str(path),
        ],
        input=frames.tobytes(),
        check=True,
    )


def _brightness(path: Path) -> list[float]:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 64 * 64).mean(axis=1).tolist()


def test_finish_drops_the_closing_cover_frame(tmp_path):
    source = tmp_path / "raw.mp4"
    # The last frame repeats the first, as the pinned cover does.
    _clip(source, [20 * index + 20 for index in range(9)] + [20])
    width, height, frames = finish(source, tmp_path / "out" / "loop.mp4")
    assert (width, height, frames) == (64, 64, 9)
    order = [round((value - 20) / 20) for value in _brightness(tmp_path / "out" / "loop.mp4")]
    assert order == list(range(9))


def test_settings_validation_and_recipe_defaults():
    settings = MotionArtworkSettings()
    assert not settings.enabled and settings.resolution == 768
    assert settings.upscale and settings.frames == 241 and settings.recipe().camera_lock == 0.5
    with pytest.raises(ValidationError):
        MotionArtworkSettings(camera_lock=1.5)
    assert settings.recipe().output_resolution == 1536
    assert MotionArtworkSettings(resolution=1536, upscale=False).recipe().output_resolution == 1536
    with pytest.raises(ValidationError):
        MotionArtworkSettings(resolution=1024)  # upscaling is on by default
    with pytest.raises(ValidationError):
        MotionArtworkSettings(frames=97)
    assert settings.recipe().instructions == DEFAULT_INSTRUCTIONS
    assert (
        MotionArtworkSettings(comfyui_url=" http://comfy:8188/ ").comfyui_url == "http://comfy:8188"
    )
    for url in ("ftp://comfy", "http://user:secret@comfy", "http://comfy?x=1", "comfy:8188"):
        with pytest.raises(ValidationError):
            MotionArtworkSettings(comfyui_url=url)
    with pytest.raises(ValidationError):
        MotionArtworkSettings(prompt_mode="fixed", fixed_prompt="  ")
    with pytest.raises(ValidationError):
        MotionArtworkSettings(resolution=2048)


def test_artwork_paths_stay_inside_the_artwork_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHORA_MOTION_ARTWORK_DIR", str(tmp_path))
    assert artwork_path("ab/loop.mp4") == tmp_path / "ab" / "loop.mp4"
    for escape in ("../secret.mp4", "/etc/passwd", "ab/../../x.mp4"):
        with pytest.raises(ValueError):
            artwork_path(escape)


@pytest.fixture
def database(monkeypatch, tmp_path):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL motion artwork tests")
    schema = "test_motion_" + uuid4().hex
    with psycopg.connect(url, autocommit=True) as db:
        db.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

    def connect():
        return psycopg.connect(url, options=f"-c search_path={schema}", row_factory=dict_row)

    monkeypatch.setattr(motion_artwork, "_connect", connect)
    monkeypatch.setattr(motion_artwork_jobs, "_connect", connect)
    monkeypatch.setenv("ECHORA_MOTION_ARTWORK_DIR", str(tmp_path / "artwork"))
    try:
        with connect() as db:
            db.execute("""
                CREATE TABLE tracks (id uuid PRIMARY KEY, title text, album text, artist text);
                CREATE TABLE lyrics (track_id uuid, text text, created_at timestamptz DEFAULT now());
                CREATE TABLE libraries (id uuid PRIMARY KEY, root_path text NOT NULL);
                CREATE TABLE track_sources (library_id uuid, track_id uuid, external_id text,
                                            source_type text, source_data jsonb);
                CREATE TABLE user_track_links (user_id uuid, library_id uuid, track_id uuid, external_id text);
            """)
            with patch("alembic.op.execute", side_effect=db.execute):
                runpy.run_path(str(VERSIONS / "0057_motion_artwork.py"))["upgrade"]()
                runpy.run_path(str(VERSIONS / "0059_motion_artwork_on_modal.py"))["upgrade"]()
                runpy.run_path(str(VERSIONS / "0061_motion_artwork_upscale.py"))["upgrade"]()
        yield connect
    finally:
        with psycopg.connect(url, autocommit=True) as db:
            db.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


CAPTION = (
    "A wide shot of a sunlit meadow with a locked-off camera as the grass sways in a light breeze."
)


class FakeComfyUI:
    rendered: list[str] = []
    encoded: list[str] = []
    prompt_seeds: list[int] = []
    zooming_seeds: set[int] = {42}
    sessions = 0

    def __init__(self, work):
        self.work = work

    def upload_image(self, data, filename, content_type="image/jpeg"):
        return filename

    def queue(self, graph):
        if "31_write_prompt" in graph:
            seed = graph["31_write_prompt"]["inputs"]["sampling_mode.seed"]
            FakeComfyUI.prompt_seeds.append(seed)
            return f"write-prompt:{seed}"
        if "6_save" in graph:
            # Like SaveConditioning: output/conditioning/<prefix>_00001_.safetensors.
            prefix = graph["6_save"]["inputs"]["filename_prefix"].split("/", 1)[1]
            (self.work / "output" / "conditioning").mkdir(parents=True, exist_ok=True)
            (self.work / "output" / "conditioning" / f"{prefix}_00001_.safetensors").write_bytes(
                b"encoding"
            )
            FakeComfyUI.encoded.append(graph["5_encode"]["inputs"]["text"])
            return "encode"
        # Rendering uses the saved encodings, never the text encoder.
        assert (
            "3_text_encoder" not in graph
            and graph["5_positive"]["class_type"] == "ConditioningLoader"
        )
        FakeComfyUI.rendered.append(graph["1_cover"]["inputs"]["image"])
        return "render"

    def wait(self, prompt_id, **_):
        if prompt_id.startswith("write-prompt"):
            seed = int(prompt_id.split(":")[1])
            # The first seed asks for a zoom, which is written again; the small prompt model also
            # sometimes repeats itself, and the encoder gets the caption once.
            if seed in FakeComfyUI.zooming_seeds:
                return {
                    "32_show_prompt": {
                        "text": [
                            "A slow push-in toward the sun over the city at dusk, then back out again."
                        ]
                    }
                }
            return {"32_show_prompt": {"text": [f"{CAPTION} {CAPTION}"]}}
        if prompt_id == "encode":
            return {}
        return {
            "21_save": {"images": [{"filename": "loop.mp4", "subfolder": "", "type": "output"}]}
        }

    def download(self, item, target):
        _clip(target, [90, 90, 90, 90])
        return target


@contextmanager
def fake_phase(settings, work, check):
    FakeComfyUI.sessions += 1
    yield FakeComfyUI(work)


def prompts():
    """Encoded cover prompts, without the shared negative prompt."""
    return [text for text in FakeComfyUI.encoded if text != NEGATIVE]


class FakeNavidrome:
    covers = {"cover-a": b"same image", "cover-b": b"same image", "cover-c": b"other image"}

    def __init__(self, *credentials):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def cover_art(self, cover_id, size):
        return self.covers[cover_id], "image/jpeg"

    def tracks(self, ids):
        # External IDs are "<album id>-<n>"; album "al-a" uses cover "cover-a".
        return [
            SimpleNamespace(
                id=song,
                title=f"Song {song}",
                artist="Artist",
                album=song[3].upper(),
                raw={"coverArt": f"cover-{song[3]}"},
            )
            for song in ids
        ]

    def lyrics(self, song_id):
        return {"text": f"Lyrics for {song_id}"}


def _library(database, owner):
    library, external_ids = uuid4(), []
    with database() as db:
        db.execute("INSERT INTO libraries VALUES (%s, %s)", (library, "http://navidrome"))
        # Album A has two songs; albums A and B share byte-identical art; album C has its own.
        for album_id, album, cover, songs in (
            ("al-a", "A", "cover-a", 2),
            ("al-b", "B", "cover-b", 1),
            ("al-c", "C", "cover-c", 1),
        ):
            for song in range(songs):
                track, external = uuid4(), f"{album_id}-{song}"
                external_ids.append(external)
                db.execute(
                    "INSERT INTO tracks VALUES (%s, %s, %s, 'Artist')",
                    (track, f"Song {external}", album),
                )
                db.execute(
                    "INSERT INTO lyrics (track_id, text) VALUES (%s, %s)",
                    (track, f"Lyrics for {external}"),
                )
                db.execute(
                    "INSERT INTO track_sources VALUES (%s,%s,%s,'subsonic',%s)",
                    (
                        library,
                        track,
                        external,
                        json.dumps({"albumId": album_id, "coverArt": cover}),
                    ),
                )
                db.execute(
                    "INSERT INTO user_track_links VALUES (%s,%s,%s,%s)",
                    (owner, library, track, external),
                )
    return library, external_ids


@pytest.fixture
def batch(database, monkeypatch):
    owner = uuid4()
    library, external_ids = _library(database, owner)
    FakeComfyUI.rendered, FakeComfyUI.encoded, FakeComfyUI.prompt_seeds, FakeComfyUI.sessions = (
        [],
        [],
        [],
        0,
    )
    FakeComfyUI.zooming_seeds = {42}
    monkeypatch.setattr(motion_artwork_jobs, "comfyui_phase", fake_phase)
    monkeypatch.setattr("echora_analysis.navidrome.NavidromeClient", FakeNavidrome)

    def run(**options):
        return motion_artwork_jobs.render_batch(
            ("http://navidrome", "u", "p"),
            owner,
            external_ids,
            progress=lambda update: None,
            check=lambda: None,
            **options,
        )

    return {"owner": owner, "library": library, "external_ids": external_ids, "run": run}


def _enable(database, **settings):
    columns = {"enabled": True, **settings}
    with database() as db:
        db.execute(
            sql.SQL("INSERT INTO motion_artwork_settings ({}) VALUES ({})").format(
                sql.SQL(", ").join(map(sql.Identifier, columns)),
                sql.SQL(", ").join(sql.Placeholder() * len(columns)),
            ),
            list(columns.values()),
        )


def test_each_album_and_each_cover_image_renders_once_in_three_phases(database, batch):
    _enable(database)
    summary = batch["run"]()
    timings = summary.pop("timings")
    assert summary == {"tracks": 4, "rendered": 4, "reused": 0, "errors": 0}
    assert set(timings) == {"prompts_seconds", "encode_seconds", "video_seconds"}
    # One ComfyUI per phase: prompts, encodings, video.
    assert len(FakeComfyUI.rendered) == 4 and FakeComfyUI.sessions == 3
    with database() as db:
        assert db.execute("SELECT count(*) AS n FROM track_motion_artwork").fetchone()["n"] == 4
        rows = db.execute(
            "SELECT status, prompt, frames, path FROM motion_artworks ORDER BY path"
        ).fetchall()
        assert [(row["status"], row["prompt"], row["frames"]) for row in rows] == [
            ("complete", CAPTION, 3)
        ] * 4
        track = db.execute("SELECT track_id FROM track_motion_artwork LIMIT 1").fetchone()[
            "track_id"
        ]
        assert motion_artwork.track_artwork(db.cursor(), batch["owner"], track)["frames"] == 3
        assert motion_artwork.track_artwork(db.cursor(), uuid4(), track) is None
    assert all(artwork_path(row["path"]).is_file() for row in rows)
    assert prompts() == [CAPTION] * 4 and FakeComfyUI.encoded.count(NEGATIVE) == 1
    # Seed 42 described a zoom, so each cover's prompt was written again with seed 43.
    assert FakeComfyUI.prompt_seeds == [42, 43] * 4

    # Nothing left to render: ComfyUI is not started again.
    assert batch["run"]()["rendered"] == 0
    assert len(FakeComfyUI.rendered) == 4 and FakeComfyUI.sessions == 3
    # Re-render all replaces the loops, still once per cover image.
    assert batch["run"](mode="all")["rendered"] == 4
    assert FakeComfyUI.sessions == 6


def test_a_prompt_that_keeps_zooming_is_rejected_after_three_attempts(database, batch):
    _enable(database)
    FakeComfyUI.zooming_seeds = {42, 43, 44}
    result = batch["run"]()
    assert result["rendered"] == 0 and result["errors"] == 4
    assert FakeComfyUI.prompt_seeds == [42, 43, 44] * 4
    assert not FakeComfyUI.rendered and not FakeComfyUI.encoded


def test_fixed_prompt_mode_fills_in_album_details_without_a_prompt_model(database, batch):
    _enable(database, prompt_mode="fixed", fixed_prompt="The {album} cover by {artist} shimmers.")
    assert batch["run"]()["rendered"] == 4
    # One song-guided loop per track, each with its own album's details; album A has two songs.
    assert sorted(prompts()) == ["The A cover by Artist shimmers."] * 2 + [
        "The B cover by Artist shimmers.",
        "The C cover by Artist shimmers.",
    ]
    # No prompt phase: only the encoding and video ComfyUI processes start.
    assert FakeComfyUI.prompt_seeds == [] and FakeComfyUI.sessions == 2


def test_clear_loops_removes_rows_mappings_and_files_but_not_during_a_sync(database, batch):
    _enable(database)
    batch["run"]()
    with database() as db:
        files = [
            artwork_path(row["path"])
            for row in db.execute("SELECT path FROM motion_artworks").fetchall()
        ]
        db.execute(
            """CREATE TABLE jobs (id uuid PRIMARY KEY, kind text, status text, payload jsonb DEFAULT '{}')"""
        )
        db.execute("INSERT INTO jobs VALUES (%s, 'navidrome_sync', 'waiting', '{}')", (uuid4(),))
    assert files and all(path.is_file() for path in files)
    with pytest.raises(RuntimeError):
        motion_artwork.clear_loops()
    with database() as db:
        db.execute("UPDATE jobs SET status='complete'")
    assert motion_artwork.clear_loops() == {"deleted": 4, "files_removed": 4}
    with database() as db:
        assert db.execute("SELECT count(*) AS n FROM motion_artworks").fetchone()["n"] == 0
        assert db.execute("SELECT count(*) AS n FROM track_motion_artwork").fetchone()["n"] == 0
    assert not any(path.exists() for path in files)
    assert not any(Path(os.environ["ECHORA_MOTION_ARTWORK_DIR"]).iterdir())


def test_selection_finds_only_songs_whose_cover_has_no_loop(database, batch):
    _enable(database)
    recipe = MotionArtworkSettings().recipe()
    with database() as db:
        assert motion_artwork.missing_external_ids(
            db.cursor(), batch["library"], batch["external_ids"], recipe
        ) == set(batch["external_ids"])
        assert motion_artwork.sync_selection(db, batch["library"], batch["external_ids"]) == set(
            batch["external_ids"]
        )
    batch["run"]()
    with database() as db:
        assert (
            motion_artwork.missing_external_ids(
                db.cursor(), batch["library"], batch["external_ids"], recipe
            )
            == set()
        )
        other = MotionArtworkSettings(resolution=512).recipe()
        # Loops made with other settings are kept by default...
        assert (
            motion_artwork.missing_external_ids(
                db.cursor(), batch["library"], batch["external_ids"], other
            )
            == set()
        )
        # ...and need a new loop only when regenerating outdated loops.
        assert motion_artwork.missing_external_ids(
            db.cursor(), batch["library"], batch["external_ids"], other, regenerate_outdated=True
        ) == set(batch["external_ids"])


def test_changed_settings_keep_existing_loops_unless_regenerating(database, batch):
    _enable(database)
    assert batch["run"]()["rendered"] == 4
    with database() as db:
        first = db.execute("SELECT path, recipe FROM motion_artworks").fetchall()
        db.execute("UPDATE motion_artwork_settings SET resolution=512")
    # Changed settings apply to future loops only; a sync keeps the existing ones.
    summary = batch["run"]()
    assert (summary["rendered"], summary["reused"]) == (0, 4)
    with database() as db:
        assert motion_artwork.sync_selection(db, batch["library"], batch["external_ids"]) == set()
        db.execute("UPDATE motion_artwork_settings SET regenerate_outdated=true")
        assert motion_artwork.sync_selection(db, batch["library"], batch["external_ids"]) == set(
            batch["external_ids"]
        )
    assert batch["run"]()["rendered"] == 4
    with database() as db:
        rows = db.execute("SELECT path, recipe FROM motion_artworks").fetchall()
    # Each loop records the settings it was made with; the new loops replace the old ones.
    assert [row["recipe"]["resolution"] for row in first] == [768] * 4
    assert [row["recipe"]["resolution"] for row in rows] == [512] * 4
    assert all(row["recipe"]["output_resolution"] == 1024 for row in rows)
    assert not any(artwork_path(row["path"]).exists() for row in first)
    assert all(artwork_path(row["path"]).is_file() for row in rows)
    # Up to date now: another sync renders nothing.
    assert batch["run"]()["rendered"] == 0


def test_disabled_or_not_during_sync_renders_nothing(database, batch):
    assert batch["run"]() is None
    with database() as db:
        assert motion_artwork.sync_selection(db, batch["library"], batch["external_ids"]) == set()
    _enable(database, generate_during_sync=False)
    assert batch["run"]() is None
    with database() as db:
        assert motion_artwork.sync_selection(db, batch["library"], batch["external_ids"]) == set()
    # Syncs are the only way loops are made, so nothing ever starts ComfyUI here.
    assert FakeComfyUI.sessions == 0


def test_unavailable_comfyui_skips_artwork_without_failing_the_batch(database, batch, monkeypatch):
    _enable(database)

    @contextmanager
    def unavailable(settings, work, check):
        raise ComfyUIUnavailable("ComfyUI is not installed in this image")
        yield

    monkeypatch.setattr(motion_artwork_jobs, "comfyui_phase", unavailable)
    summary = batch["run"]()
    assert (
        summary["rendered"] == 0
        and summary["unavailable"] == "ComfyUI is not installed in this image"
    )
    with database() as db:
        assert db.execute("SELECT count(*) AS n FROM motion_artworks").fetchone()["n"] == 0


FAKE_COMFYUI = """
import argparse, http.server, sys
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int)
arguments, _ = parser.parse_known_args()
if "--fail" in open("mode").read():
    sys.exit(3)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/system_stats" else 404)
        self.end_headers()
        self.wfile.write(b"{}")
    def log_message(self, *_):
        pass
http.server.HTTPServer(("127.0.0.1", arguments.port), Handler).serve_forever()
"""


def test_launch_starts_comfyui_on_demand_and_stops_it(tmp_path):
    (tmp_path / "main.py").write_text(FAKE_COMFYUI)
    (tmp_path / "mode").write_text("serve")
    with launch(
        str(tmp_path), sys.executable, startup_timeout_seconds=30, check=lambda: None
    ) as url:
        assert url.startswith("http://127.0.0.1:")
        assert httpx.get(f"{url}/system_stats").status_code == 200
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{url}/system_stats", timeout=2)
    # Phases of one batch share a work directory, including the saved encodings.
    work = tmp_path / "batch"
    for _ in range(2):
        with launch(
            str(tmp_path),
            sys.executable,
            startup_timeout_seconds=30,
            check=lambda: None,
            work_dir=work,
        ):
            pass
    assert (work / "output" / "conditioning").is_dir()
    assert f"base_path: {work / 'output'}" in (work / "batch_model_paths.yaml").read_text()
    assert "embeddings: conditioning" in (work / "batch_model_paths.yaml").read_text()
    (tmp_path / "mode").write_text("--fail")
    with pytest.raises(ComfyUIUnavailable, match="exited during startup"):
        with launch(str(tmp_path), sys.executable, startup_timeout_seconds=30, check=lambda: None):
            pass
    with pytest.raises(ComfyUIUnavailable, match="not installed"):
        with launch(
            str(tmp_path / "missing"),
            sys.executable,
            startup_timeout_seconds=30,
            check=lambda: None,
        ):
            pass


def _chat(content):
    return httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "vision",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
        },
    )


def _external_ai():
    from echora_analysis.external_ai import ExternalAISettings

    return ExternalAISettings(enabled=True, url="http://ai.test/v1", model="vision")


def test_external_writer_sends_the_cover_and_returns_prompt_and_people():
    from echora_analysis.motion_artwork_writer import write_prompt

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return _chat(json.dumps({"prompt": f"  {CAPTION}  ", "people": 2}))

    prompt, people = write_prompt(
        _external_ai(),
        "key",
        b"cover",
        "image/png",
        "Prefer rocking.",
        transport=httpx.MockTransport(handler),
    )
    assert (prompt, people) == (CAPTION, 2)
    body = requests[0]
    assert body["model"] == "vision" and body["response_format"] == {"type": "json_object"}
    assert (
        "locked off on a tripod" in body["messages"][0]["content"]
        and "Prefer rocking." in body["messages"][0]["content"]
    )
    image = body["messages"][1]["content"][1]["image_url"]["url"]
    assert image == "data:image/png;base64,Y292ZXI="


def test_external_writer_rewrites_zooms_and_rejects_bad_answers():
    from echora_analysis.motion_artwork_writer import WriterError, write_prompt

    answers = iter(
        [
            json.dumps(
                {
                    "prompt": "The camera slowly pushes in toward the singer on a quiet stage tonight.",
                    "people": 1,
                }
            ),
            json.dumps({"prompt": CAPTION, "people": 1}),
        ]
    )
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return _chat(next(answers))

    assert (
        write_prompt(
            _external_ai(), None, b"c", "image/jpeg", transport=httpx.MockTransport(handler)
        )[0]
        == CAPTION
    )
    assert len(calls) == 2
    assert calls[1]["messages"][-1] == {"role": "user", "content": CAMERA_RETRY}
    assert "pushes in" in calls[1]["messages"][-2]["content"]
    for response in (
        _chat("not json"),
        _chat(json.dumps({"prompt": "too short", "people": 1})),
        _chat(json.dumps({"prompt": CAPTION, "people": "two"})),
        httpx.Response(500, json={}),
    ):
        with pytest.raises(WriterError):
            write_prompt(
                _external_ai(),
                None,
                b"c",
                "image/jpeg",
                transport=httpx.MockTransport(lambda request, response=response: response),
            )
    from echora_analysis.external_ai import ExternalAISettings

    with pytest.raises(WriterError, match="not enabled"):
        write_prompt(ExternalAISettings(), None, b"c", "image/jpeg")


def test_external_writer_never_returns_a_repeated_camera_move():
    from echora_analysis.motion_artwork_writer import WriterError, write_prompt

    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return _chat(
            json.dumps(
                {
                    "prompt": "The camera slowly zooms in toward the empty boat on the water.",
                    "people": 0,
                }
            )
        )

    with pytest.raises(WriterError, match="camera movement after three attempts"):
        write_prompt(
            _external_ai(), None, b"c", "image/png", transport=httpx.MockTransport(handler)
        )
    assert len(calls) == 3


def test_gemma_retries_with_correction_and_never_encodes_camera_movement():
    class CameraWriter:
        def __init__(self):
            self.requests = []

        def queue(self, graph):
            self.requests.append(graph["31_write_prompt"]["inputs"]["prompt"])
            return "request"

        def wait(self, request, **kwargs):
            return {
                "32_show_prompt": {
                    "text": ["The camera slowly zooms in toward the empty boat on the water."]
                }
            }

    comfy = CameraWriter()
    with pytest.raises(
        motion_artwork_jobs.ComfyUIError, match="camera movement after three attempts"
    ):
        motion_artwork_jobs._write_prompt(comfy, Recipe(), {}, "cover.png", 42, {})
    assert len(comfy.requests) == 3
    assert CAMERA_RETRY not in comfy.requests[0]
    assert CAMERA_RETRY in comfy.requests[1]


def test_external_prompts_skip_the_prompt_comfyui_and_every_loop_is_pinned(
    database, batch, monkeypatch
):
    _enable(database, prompt_mode="external")
    monkeypatch.setattr(
        "echora_analysis.translation_storage.load_settings", lambda: (_external_ai(), None)
    )
    people = {b"same image": 4, b"other image": 1}
    monkeypatch.setattr(
        "echora_analysis.motion_artwork_writer.write_prompt",
        lambda settings, key, data, content_type, instructions, **song: (CAPTION, people[data]),
    )
    graphs = []
    original = FakeComfyUI.queue
    monkeypatch.setattr(
        FakeComfyUI, "queue", lambda self, graph: graphs.append(graph) or original(self, graph)
    )
    assert batch["run"]()["rendered"] == 4
    # No Gemma prompt phase: only the encoding and video ComfyUI processes start.
    assert FakeComfyUI.prompt_seeds == [] and FakeComfyUI.sessions == 2
    videos = [graph for graph in graphs if "17_sample" in graph]
    anchors = sorted(
        graph.get("12_end_frame", {}).get("inputs", {}).get("strength", 0) for graph in videos
    )
    # Every cover, whatever its people count, is pinned fully at the end.
    assert anchors == [1.0, 1.0, 1.0, 1.0]


class FakeModalSession:
    """Renders through the real render_loops, like the Modal Artwork class, and records what it got."""

    def __init__(self):
        self.calls = []

    def motion_artwork(self, settings, recipe, covers):
        self.calls.append((settings, recipe, covers))
        for event in motion_artwork_jobs.render_loops(settings, recipe, covers):
            if "rendered" in event:
                # Like the transfer Volume: the worker receives its own copy of the video.
                copy = Path(event["file"]).with_name(f"received-{event['rendered']}.mp4")
                copy.write_bytes(Path(event["file"]).read_bytes())
                event = {**event, "file": str(copy)}
            yield event

    def close(self):
        pass


def test_modal_syncs_render_only_when_switched_on_for_modal(database, batch):
    from echora_analysis import remote_compute

    _enable(database, generate_during_sync=True, generate_on_modal=False)
    remote = FakeModalSession()
    with (
        remote_compute.computing_on("modal"),
        remote_compute.session("modal", factory=lambda: remote),
    ):
        assert batch["run"]() is None
        with database() as db:
            assert (
                motion_artwork.sync_selection(db, batch["library"], batch["external_ids"]) == set()
            )
    assert remote.calls == [] and FakeComfyUI.sessions == 0


def test_modal_sync_renders_on_modal_and_stores_loops_here(database, batch, monkeypatch):
    from echora_analysis import remote_compute

    _enable(database, generate_during_sync=False, generate_on_modal=True)
    remote = FakeModalSession()
    with (
        remote_compute.computing_on("modal"),
        remote_compute.session("modal", factory=lambda: remote),
    ):
        with database() as db:
            assert motion_artwork.sync_selection(db, batch["library"], batch["external_ids"])
        summary = batch["run"]()
    assert len(remote.calls) == 1 and summary["rendered"] == 4 and summary["errors"] == 0
    with database() as db:
        rows = db.execute("SELECT path, status FROM motion_artworks").fetchall()
    root = Path(os.environ["ECHORA_MOTION_ARTWORK_DIR"])
    assert len(rows) == 4 and all(
        row["status"] == "complete" and (root / row["path"]).is_file() for row in rows
    )
    # A local sync with generation off for this server renders nothing more.
    assert batch["run"]() is None


def test_external_prompts_are_written_before_the_batch_and_used_by_render(
    database, batch, monkeypatch
):
    _enable(database, prompt_mode="external")
    monkeypatch.setattr(
        "echora_analysis.translation_storage.load_settings", lambda: (_external_ai(), None)
    )
    written = []

    def write_prompt(settings, key, data, content_type, instructions, **song):
        written.append(song["title"])
        return CAPTION, 1

    monkeypatch.setattr("echora_analysis.motion_artwork_writer.write_prompt", write_prompt)
    prepare = lambda: motion_artwork_jobs.prepare_prompts(  # noqa: E731
        ("http://navidrome", "u", "p"),
        batch["external_ids"],
        progress=lambda _: None,
        check=lambda: None,
    )
    prompts = prepare()
    assert len(prompts) == 4 and len(written) == 4
    written.clear()
    # The render stage uses the prepared prompts and writes none itself.
    assert batch["run"](prompts=prompts)["rendered"] == 4
    assert written == []
    with database() as db:
        assert {row["prompt"] for row in db.execute("SELECT prompt FROM motion_artworks")} == {
            CAPTION
        }
    # Songs that now have a loop need no prompt in a later batch.
    assert prepare() == {} and written == []


def test_a_prompt_prepared_for_other_settings_is_written_again(database, batch, monkeypatch):
    _enable(database, prompt_mode="external")
    monkeypatch.setattr(
        "echora_analysis.translation_storage.load_settings", lambda: (_external_ai(), None)
    )
    written = []

    def write_prompt(settings, key, data, content_type, instructions, **song):
        written.append(song["title"])
        return CAPTION, 1

    monkeypatch.setattr("echora_analysis.motion_artwork_writer.write_prompt", write_prompt)
    prompts = motion_artwork_jobs.prepare_prompts(
        ("http://navidrome", "u", "p"),
        batch["external_ids"],
        progress=lambda _: None,
        check=lambda: None,
    )
    assert len(written) == 4
    written.clear()
    # The recipe changes between preparing and rendering.
    with database() as db:
        db.execute("UPDATE motion_artwork_settings SET resolution=512")
    assert batch["run"](prompts=prompts)["rendered"] == 4
    assert len(written) == 4
