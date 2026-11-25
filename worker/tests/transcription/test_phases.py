"""Transcription phases: timing repair runs after generation, on the audio the draft kept."""

import io
from types import SimpleNamespace

import numpy as np
import soundfile as sf

import worker.kara.pipeline as karaoke_pipeline
import worker.transcription.timing as transcription_timing
from worker.transcription.song import SongTranscriber, needs_repair


def segments(start):
    return [
        {
            "start_ms": start * 1000 + 100,
            "end_ms": start * 1000 + 200,
            "speaker": "S01",
            "text": f"line {start}",
        }
    ]


def draft_with(windows):
    sr = 100
    candidates = [
        (index, start, start + 10, segments(start), None) for index, start in enumerate(windows)
    ]
    return {
        "ranges": [(start, start + 10) for start in windows],
        "duration": windows[-1] + 10,
        "sr": sr,
        "candidates": candidates,
        "unresolved": [],
        "budget": {},
        "diagnostics": [],
        "repairs": {
            index: np.full(10 * sr, index + 1, dtype=np.float32)
            for index in range(min(2, len(windows)))
        },
    }


def test_repair_aligns_the_kept_audio_and_respects_the_two_repair_budget(monkeypatch):
    aligned_audio = []

    def compressed(found, start, end):
        return None if found and found[0].get("repaired") else {"words_per_second": 9}

    def run_fa_kara(audio, text, language, provided_audio):
        assert provided_audio  # the repaired span, aligned as given
        aligned_audio.append(sf.read(io.BytesIO(audio))[0][0])
        return {"lines": [], "model": "aligner", "model_revision": "r"}

    monkeypatch.setattr(transcription_timing, "compressed_timing", compressed)
    monkeypatch.setattr(
        transcription_timing,
        "apply_aligned_times",
        lambda source, lines, start, end: [{**s, "repaired": True} for s in source],
    )
    monkeypatch.setattr(karaoke_pipeline, "_run_fa_kara", run_fa_kara)
    draft = draft_with([0, 20, 40])
    assert needs_repair(draft)
    result = SongTranscriber("model", "a" * 40).finish(draft)
    repairs = result["transcription"]["timing_repairs"]
    # The first two compressed windows are repaired with their own kept audio; the third is
    # recorded as over budget, as before the phases existed.
    assert [r["status"] for r in repairs] == ["repaired", "repaired"]
    assert aligned_audio == [1.0, 2.0]
    assert [d.get("timing_status") for d in result["transcription"]["windows"]] == [
        "repair_budget_exhausted"
    ]


def test_drafts_without_compressed_windows_need_no_aligner():
    draft = draft_with([0])
    draft["repairs"] = {}
    assert not needs_repair(draft)
    assert SongTranscriber("model", "a" * 40).finish(draft)["text"] == "line 0"


def test_a_stem_cache_miss_moves_moss_off_the_gpu_while_separating(monkeypatch):
    import torch

    import worker.transcription.budget as transcription_budget
    import worker.transcription.song as song_transcription

    moves = []
    model = SimpleNamespace(to=lambda device: moves.append(device))
    processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=100), apply_chat_template=lambda *a, **k: ""
    )

    def vocal_waveform(audio, sample_rate, check, before_separate=None):
        # The stem is not cached, so separation is about to run.
        before_separate()
        moves.append("separated")
        return np.zeros(sample_rate * 10, dtype=np.float32)

    monkeypatch.setattr(song_transcription, "vocal_waveform", vocal_waveform)
    monkeypatch.setattr(transcription_budget, "generate_song", lambda *a, **k: ([], [], {}))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    transcriber = SongTranscriber("model", "a" * 40)
    transcriber._processor, transcriber._model = processor, model
    transcriber._device, transcriber._dtype = "cuda", torch.bfloat16
    transcriber.generate(b"audio")
    # MOSS leaves the GPU before Roformer separates and returns afterwards.
    assert moves == ["cpu", "separated", "cuda"]
