"""Versioned lyric entries, independent of duplicated acoustic supervision.

Config contract: top-level ``song_text_version``; absent means legacy version 1.
New training uses version 2. Decoder callers can use canonical_tokens on a
manifest-shaped row. No tensors/dependencies and no mutation of input tokens.
"""

LEGACY_VERSION = 1
CURRENT_VERSION = 2


def canonical_tokens(row: dict, version: int = CURRENT_VERSION) -> list:
    """Return (token, voice) in line/time order (0 lead, 1 background).

    V2 removes a lead copy of the *same occurrence* on the background stream.
    Repeated words at different times remain distinct. Acoustic targets are not
    edited. Ignored entries never become v2 text targets.
    """
    if version not in (LEGACY_VERSION, CURRENT_VERSION):
        raise ValueError(f"Unsupported song_text_version: {version}")
    lead = [(t, 0) for t in row["alignment_tokens"] if t.get("ids") and not t.get("ignore")]
    back = [
        (t, 1)
        for t in row.get("background_tokens", []) or []
        if t.get("ids") and (version == LEGACY_VERSION or not t.get("ignore"))
    ]
    if version == CURRENT_VERSION:

        def key(t):
            # Main-head tails can be trimmed after background targets are copied.
            return (t["line"], t["start"], tuple(t["ids"]))

        def is_copy(main, background):
            roles = (main.get("vocal_role"), background.get("vocal_role"))
            categories = (main.get("background_category"), background.get("background_category"))
            # Explicit policy wins over timing/spelling, including identical words
            # sung by two voices at the same instant.
            if any(role not in (None, "background") for role in roles):
                return False
            if any(category not in (None, "sequential") for category in categories):
                return False
            if main.get("vocal_role") == "background" or "sequential" in categories:
                return True
            # Old fixtures lack policy metadata: preserve the historical exact
            # occurrence rule, rather than guessing across different end times.
            return (
                all(value is None for value in roles + categories)
                and main["end"] == background["end"]
            )

        backing = {}
        for token, _ in back:
            backing.setdefault(key(token), []).append(token)
        remaining = []
        for token, voice in lead:
            candidates = backing.get(key(token), [])
            for index, candidate in enumerate(candidates):
                if is_copy(token, candidate):
                    candidates.pop(index)  # One acoustic copy per occurrence.
                    break
            else:
                remaining.append((token, voice))
        lead = remaining
    return sorted(lead + back, key=lambda item: (item[0]["line"], item[0]["start"]))


def lyric_tokens(token_letters, voices, lines, version=LEGACY_VERSION):
    """Inference lyrics already contain one entry per occurrence, not acoustic copies.

    Never deduplicate by spelling: repeats and simultaneous voices are distinct.
    Version 1 retains historical inference ordering; version 2 uses the same
    canonical occurrence ordering as manifest training (synthetic occurrence
    indexes stand in for unavailable times, and are never used as timing priors).
    """
    if not (len(token_letters) == len(voices) == len(lines)):
        raise ValueError("Text letters, voices and lines must have equal lengths")
    if version not in (LEGACY_VERSION, CURRENT_VERSION):
        raise ValueError(f"Unsupported song_text_version: {version}")
    tokens = [
        (dict(ids=ids, line=line, start=i, end=i + 1), voice)
        for i, (ids, voice, line) in enumerate(zip(token_letters, voices, lines))
    ]
    if any(voice not in (0, 1) for _, voice in tokens):
        raise ValueError("Text voice must be lead (0) or background (1)")
    if version == LEGACY_VERSION:
        return tokens
    return canonical_tokens(
        {
            "alignment_tokens": [t for t, v in tokens if v == 0],
            "background_tokens": [t for t, v in tokens if v == 1],
        },
        version,
    )


def text_features(tokens, max_letters, pad):
    """Shared training/inference text encoding, as dependency-free Python lists."""
    result = {key: [] for key in ("letters", "voice", "line", "position")}
    previous, count = None, 0
    for token, voice in tokens:
        ids = list(token["ids"])[:max_letters]
        result["letters"].append(ids + [pad] * (max_letters - len(ids)))
        result["voice"].append(voice)
        result["line"].append(token["line"])
        count = count + 1 if token["line"] == previous else 0
        previous = token["line"]
        result["position"].append(count)
    return result
