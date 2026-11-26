from core.lyrics.smoothing import smooth_karaoke


def line(*syllables, **extra):
    return {
        "text": "".join(t for t, _, _ in syllables),
        "syllables": [{"text": t, "start_ms": s, "end_ms": e, **extra} for t, s, e in syllables],
    }


def times(lines):
    return [(s["start_ms"], s["end_ms"]) for s in lines[0]["syllables"]]


def test_all_switches_off_returns_the_lines_unchanged():
    lines = [line(("en", 0, 50), ("e", 300, 320), ("my ", 320, 900))]
    assert smooth_karaoke(lines) is lines


def test_close_gaps_extends_a_syllable_to_the_next_in_its_word_only():
    lines = [line(("hel", 0, 100), ("lo ", 300, 500), ("world", 800, 1200))]
    result = smooth_karaoke(lines, close_gaps=True)
    # The gap between words (500 -> 800) stays.
    assert times(result) == [(0, 300), (300, 500), (800, 1200)]
    assert times(lines) == [(0, 100), (300, 500), (800, 1200)]  # input untouched


def test_even_pacing_shares_an_uneven_word_by_letters_keeping_its_bounds():
    lines = [line(("beau", 0, 900), ("ti", 900, 950), ("ful ", 950, 1000))]
    assert times(smooth_karaoke(lines, even_pacing=True)) == [(0, 444), (444, 667), (667, 1000)]


def test_even_pacing_keeps_a_held_last_syllable():
    lines = [line(("en", 0, 100), ("e", 100, 120), ("my ", 120, 900))]
    assert times(smooth_karaoke(lines, even_pacing=True)) == [(0, 100), (100, 120), (120, 900)]


def test_min_syllable_borrows_from_the_previous_when_the_next_has_nothing_to_spare():
    lines = [line(("a", 0, 300), ("b", 300, 320), ("c ", 320, 360))]
    # "b" takes 80 ms from "a"; "c" stays short: "b" now sits at the minimum, and the word's
    # end never moves.
    assert times(smooth_karaoke(lines, min_syllable=True)) == [(0, 220), (220, 320), (320, 360)]


def test_min_syllable_uses_a_gap_first():
    lines = [line(("a", 0, 40), ("b ", 200, 500))]
    assert times(smooth_karaoke(lines, min_syllable=True)) == [(0, 100), (200, 500)]


def test_cjk_syllables_and_voice_changes_are_separate_words():
    cjk = [line(("君", 0, 10), ("の", 500, 520), ("名", 900, 2000))]
    assert times(smooth_karaoke(cjk, close_gaps=True, min_syllable=True, even_pacing=True)) == [
        (0, 10),
        (500, 520),
        (900, 2000),
    ]
    mixed = {
        "text": "ohyeah",
        "syllables": [
            {"text": "oh", "start_ms": 0, "end_ms": 100},
            {"text": "yeah", "start_ms": 400, "end_ms": 600, "voice": 2},
        ],
    }
    assert times(smooth_karaoke([mixed], close_gaps=True)) == [(0, 100), (400, 600)]


def test_untimed_words_and_lines_without_syllables_are_left_alone():
    lines = [
        {"text": "plain", "start_ms": 0},
        {
            "text": "ab",
            "syllables": [{"text": "a", "start_ms": 0}, {"text": "b", "start_ms": 5, "end_ms": 9}],
        },
    ]
    result = smooth_karaoke(lines, close_gaps=True, min_syllable=True, even_pacing=True)
    assert result[0] is lines[0]
    assert result[1]["syllables"] == lines[1]["syllables"]


def test_an_opening_quote_glued_after_a_space_still_ends_the_word():
    lines = [line(('like, "', 0, 600), ("Hey", 700, 900))]
    assert times(smooth_karaoke(lines, close_gaps=True)) == [(0, 600), (700, 900)]
