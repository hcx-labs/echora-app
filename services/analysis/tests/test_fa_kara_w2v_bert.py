"""w2v-BERT (Sylla-600M) support in the vendored FA-Kara aligner: the exact frame clock per
backbone, the auxiliary heads and the activity source, with the wav2vec2 path unchanged."""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "fa_kara"
sys.path.insert(0, str(VENDOR))

from ctc_segmentation import (  # noqa: E402
    conv_frame_count,
    frame_count,
    stitched_frame_times,
    w2v_bert_frame_count,
)

SR = 16000


def test_w2v_bert_frame_count_is_half_the_fbank_frames():
    # fbank: 400-sample window every 160 samples; model frames: pairs of fbank frames.
    assert w2v_bert_frame_count(399) == 0 and w2v_bert_frame_count(400) == 0
    assert w2v_bert_frame_count(560) == 1 and w2v_bert_frame_count(879) == 1
    assert w2v_bert_frame_count(880) == 2
    assert w2v_bert_frame_count(SR) == 49 == conv_frame_count(SR)
    assert w2v_bert_frame_count(45 * SR) == 2249 == conv_frame_count(45 * SR)
    # The two front ends do not agree for every length (MMS: floor((N - 400) / 320) + 1).
    assert w2v_bert_frame_count(400 + 320 * 10) == 10 and conv_frame_count(400 + 320 * 10) == 11
    assert frame_count(SR, "wav2vec2-bert") == 49 and frame_count(SR) == conv_frame_count(SR)
    with pytest.raises(ValueError):
        frame_count(SR, "whisper")


def test_exact_clock_per_backbone_keeps_the_stride_and_moves_the_centre():
    legacy = stitched_frame_times(240 * SR)
    assert torch.equal(legacy, stitched_frame_times(240 * SR, backbone="wav2vec2"))
    bert = stitched_frame_times(240 * SR, backbone="wav2vec2-bert")
    assert abs(float(legacy[0]) - 199.5 / SR) < 1e-12
    assert abs(float(bert[0]) - 279.5 / SR) < 1e-12
    assert abs(float(bert[1] - bert[0]) - 0.02) < 1e-12
    # Every 45 s chunk gives 2249 frames on both front ends, so the stitched lattices have the same
    # length and the same 5 ms shift everywhere.
    assert bert.shape == legacy.shape
    assert torch.allclose(bert - legacy, torch.full_like(bert, 80 / SR))
    # A single short chunk follows the backbone's own frame count (10 s + 160 samples: 500 vs 499).
    short = 10 * SR + 160
    assert stitched_frame_times(short).shape[0] == 500 == conv_frame_count(short)
    assert stitched_frame_times(short, backbone="wav2vec2-bert").shape[0] == 499
    assert w2v_bert_frame_count(short) == 499


def test_pitch_from_f0_logits_decodes_voicing_and_the_best_bin():
    from align_yohane import F0_BINS, pitch_from_f0_logits

    logits = torch.full((3, 1 + F0_BINS), -5.0)
    logits[0, 0] = 5.0  # voiced
    logits[:, 1] = 0.0  # bin 1 everywhere
    logits[1, 120] = 3.0  # bin 120 on frame 1
    voicing, hertz = pitch_from_f0_logits(logits)
    assert voicing[0] > 0.99 and voicing[1] < 0.01
    assert abs(float(hertz[0]) - 10 * 2 ** ((20 * 1 + 1997.38) / 1200)) < 1e-3
    assert abs(float(hertz[1]) - 10 * 2 ** ((20 * 120 + 1997.38) / 1200)) < 1e-3
    assert 30 < float(hertz[0]) < 40 and 125 < float(hertz[1]) < 135


def test_aux_heads_load_from_the_checkpoint_directory(tmp_path):
    from safetensors.torch import save_file

    from align_yohane import AUX_HEADS_FILE, _load_aux_heads

    assert _load_aux_heads(str(tmp_path), 8) is None  # no file: no heads, no error
    state = {
        "vad.weight": torch.ones(1, 8),
        "vad.bias": torch.zeros(1),
        "f0.weight": torch.zeros(361, 8),
        "f0.bias": torch.arange(361, dtype=torch.float32),
    }
    save_file(state, str(tmp_path / AUX_HEADS_FILE))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "wav2vec2-bert"}))
    heads = _load_aux_heads(str(tmp_path), 8)
    assert heads is not None and set(heads.keys()) == {"vad", "f0"}
    hidden = torch.ones(1, 4, 8)
    assert torch.allclose(heads["vad"](hidden)[..., 0], torch.full((1, 4), 8.0))
    assert heads["f0"](hidden).shape == (1, 4, 361)
    with pytest.raises(RuntimeError):
        _load_aux_heads(str(tmp_path), 16)  # wrong width


def test_vad_activity_is_scaled_like_the_energy_activity():
    from align_yohane import Wav2Vec2ForcedAligner

    aligner = Wav2Vec2ForcedAligner.__new__(Wav2Vec2ForcedAligner)
    aligner.last_vad_logits = None
    with pytest.raises(RuntimeError):
        aligner.vad_activity()
    aligner.last_vad_logits = torch.linspace(-6, 2, 100)
    raw = aligner.vad_activity(scaled=False)
    scaled = aligner.vad_activity()
    assert torch.allclose(raw, torch.sigmoid(aligner.last_vad_logits))
    assert float(scaled.min()) == 0.0 and float(scaled.max()) == 1.0
    assert torch.all(scaled[1:] >= scaled[:-1])


def test_activity_source_is_validated():
    from align_yohane import VOCAL_ACTIVITY_SOURCES, align_audio_with_timeline

    assert VOCAL_ACTIVITY_SOURCES == ("energy", "vad", "external")
    with pytest.raises(ValueError):
        align_audio_with_timeline(
            torch.zeros(SR).numpy(), [["a"]], sr=SR, vocal_activity_source="loudness"
        )
