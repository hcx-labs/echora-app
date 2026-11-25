import pytest

from worker.transcription.budget import decode_song
from worker.transcription.recovery import WindowDecodeError
from worker.transcription.timing import apply_aligned_times, compressed_timing


def compressed():
    return [
        {
            "text": f"Line {i} has several different words",
            "speaker": "S01",
            "start_ms": 98000 + i * 500,
            "end_ms": 98500 + i * 500,
        }
        for i in range(8)
    ]


def test_requires_sustained_compression_not_one_fast_line():
    assert compressed_timing(compressed(), 96, 128)
    ordinary = [
        {**s, "start_ms": 96000 + i * 3000, "end_ms": 99000 + i * 3000}
        for i, s in enumerate(compressed())
    ]
    assert compressed_timing(ordinary, 96, 128) is None
    assert compressed_timing(compressed()[:2], 96, 128) is None


def test_adlibs_and_repeated_chorus_are_not_compression_evidence():
    repeated = [{**s, "text": "Same repeated phrase"} for s in compressed()]
    assert compressed_timing(repeated, 96, 128) is None
    assert compressed_timing([{**s, "text": "oh"} for s in compressed()], 96, 128) is None


def test_alignment_preserves_all_text_speakers_and_repetitions():
    source = compressed()
    source[-1]["text"] = source[-2]["text"]
    aligned = [
        {"text": s["text"], "start_ms": i * 3000, "end_ms": i * 3000 + 2000}
        for i, s in enumerate(source)
    ]
    result = apply_aligned_times(source, aligned, 96, 128)
    assert [s["text"] for s in result] == [s["text"] for s in source]
    assert all(s["speaker"] == "S01" for s in result)
    assert result[-1]["start_ms"] != result[-2]["start_ms"]
    assert compressed_timing(result, 96, 128) is None


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [{"text": "changed", "start_ms": 0, "end_ms": 1000}] * 8,
        [{"text": s["text"], "start_ms": 0, "end_ms": 0} for s in compressed()],
        [{"text": s["text"], "start_ms": 0, "end_ms": 50000} for s in compressed()],
    ],
)
def test_reject_mismatched_or_invalid_alignment(bad):
    with pytest.raises(ValueError):
        apply_aligned_times(compressed(), bad, 96, 128)


def test_all_generation_finishes_before_reconciliation_and_retry_ownership():
    events = []

    def decode(a, b, limit):
        events.append("decode")
        if limit == 2048:
            raise WindowDecodeError("retry")
        return [{"text": "keep", "start_ms": 0, "end_ms": 1000}], 10

    def reconcile(a, b, segments):
        events.append("align")
        return [{**segments[0], "start_ms": a * 1000 + 3000, "end_ms": a * 1000 + 4000}]

    result, _, _ = decode_song([(0, 60)], decode, lambda a, b: None, reconcile=reconcile)
    assert events == ["decode", "decode", "decode", "align", "align"]
    # The right-half candidate would be discarded at its original 0s timestamp.
    assert len(result[0]) == 2
    assert result[0][1]["start_ms"] == 31000
