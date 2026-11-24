import pytest
from worker.transcription.recovery import recover_window, stalled_segments, WindowDecodeError


def test_detects_stalled_timestamps_not_real_choruses():
    assert stalled_segments("[0.00][S01]Okay.[0.06]" * 4)
    assert not stalled_segments("".join(f"[{i}.00][S01]Okay.[{i + 1}.00]" for i in range(4)))


def test_overlapping_retry_preserves_coverage():
    seen = []

    def decode(a, b):
        seen.append((a, b))
        if b - a > 40:
            raise WindowDecodeError("limit")
        return [{"start_ms": a * 1000, "end_ms": b * 1000, "text": "line"}]

    assert len(recover_window(decode, 48, 108)) == 2
    assert seen == [(48, 108), (48, 80), (76, 108)]


def test_never_returns_partial_song_when_small_window_fails():
    seen = []

    def decode(a, b):
        seen.append((a, b))
        raise WindowDecodeError("invalid")

    with pytest.raises(WindowDecodeError):
        recover_window(decode, 0, 60)
    assert len(seen) == 5


def test_partial_recovery_keeps_other_windows_and_reports_gap():
    gaps = []

    def decode(a, b):
        if a < 30:
            raise WindowDecodeError("repeating output")
        return [{"start_ms": a * 1000, "end_ms": b * 1000, "text": "real lyric"}]

    result = recover_window(decode, 0, 60, on_unresolved=gaps.append)
    assert result
    assert gaps and all(g["status"] == "unresolved" for g in gaps)
    assert all(g["end_ms"] > g["start_ms"] for g in gaps)
    assert all(s["text"] == "real lyric" for s in result)


def test_all_failed_windows_are_not_mistaken_for_lyrics():
    gaps = []

    def decode(a, b):
        raise WindowDecodeError("loop")

    assert recover_window(decode, 0, 6, on_unresolved=gaps.append) == []
    assert gaps == [{"start_ms": 0, "end_ms": 6000, "reason": "loop", "status": "unresolved"}]


def test_unrelated_errors_are_not_retried():
    def decode(a, b):
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError):
        recover_window(decode, 0, 60)


def test_cancellation_is_not_retried():
    def check():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError):
        recover_window(lambda a, b: [], 0, 60, check)
