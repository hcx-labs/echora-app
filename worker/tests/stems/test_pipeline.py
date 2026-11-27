"""Karaoke stems encoding: one four-channel Opus file whose pairs are vocals and instrumental."""

import shutil
import subprocess

import numpy as np
import pytest

import worker.media.audio as audio
import worker.media.preprocessing as preprocessing
from worker.stems import pipeline


def _tone(frequency, seconds=2.0, rate=44100):
    t = np.arange(int(seconds * rate)) / rate
    wave = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    return np.stack([wave, wave], axis=1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg with libopus")
def test_pairs_carry_vocals_then_the_mix_minus_vocals(monkeypatch, tmp_path):
    vocals, backing = _tone(440), _tone(1500)
    monkeypatch.setattr(preprocessing, "vocal_stereo", lambda source, check: vocals)
    monkeypatch.setattr(audio, "decode_audio_channels", lambda *_: vocals + backing)
    content, duration = pipeline.encode_stems(b"song", 128)
    assert duration == pytest.approx(2.0)
    path = tmp_path / "stems.webm"
    path.write_bytes(content)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,channels", "-of",
         "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert probe.stdout.strip() == "opus,4"
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ar", "48000", "pipe:1"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    decoded = np.frombuffer(raw, dtype="<f4").reshape(-1, 4)[4800:-4800]

    def dominant(channel):
        spectrum = np.abs(np.fft.rfft(decoded[:, channel]))
        return np.argmax(spectrum) * 48000 / len(decoded)

    assert dominant(0) == pytest.approx(440, abs=5)
    assert dominant(1) == pytest.approx(440, abs=5)
    assert dominant(2) == pytest.approx(1500, abs=5)
    assert dominant(3) == pytest.approx(1500, abs=5)


def test_the_recipe_names_the_bitrate_and_separator():
    recipe = pipeline.stems_recipe(160)
    assert recipe.startswith("stems-v1:") and recipe.endswith("opus-webm-quad-160x2")
    assert pipeline.stems_recipe(160) != pipeline.stems_recipe(192)


def test_files_stay_inside_the_stems_directory(monkeypatch, tmp_path):
    import uuid

    from core.config import reset_config_cache

    monkeypatch.setenv("ECHORA_KARAOKE_STEMS_DIR", str(tmp_path))
    reset_config_cache()
    try:
        track = uuid.uuid4()
        relative = pipeline._write_file(track, b"data")
        assert (tmp_path / relative).read_bytes() == b"data"
        assert relative.startswith(str(track)[:2] + "/")
        with pytest.raises(ValueError):
            pipeline.stems_path("../outside.webm")
    finally:
        reset_config_cache()
