"""Hardware-independent, song-wide recovery limits. First-pass coverage comes first."""

from worker.transcription.recovery import WindowDecodeError


def decode_song(
    ranges,
    decode,
    retry_policy,
    check=lambda: None,
    progress=lambda _: None,
    reconcile=lambda a, b, segments: segments,
):
    candidates, unresolved, budget = generate_song(ranges, decode, retry_policy, check, progress)
    return reconcile_song(ranges, candidates, reconcile, check), unresolved, budget


def generate_song(ranges, decode, retry_policy, check=lambda: None, progress=lambda _: None):
    """All generation for one song: first passes, then budgeted retries of failed windows.

    Returns (candidates, unresolved, budget). Candidates are (window index, start, end,
    segments, ownership) in the order reconcile_song must process them.
    """
    candidates = []
    failed = []
    unresolved = []
    budget = {"calls": 0, "tokens": 0, "max_calls": 4, "max_tokens": 4096, "per_call_tokens": 1024}

    def gap(a, b, reason, evidence=None):
        unresolved.append(
            {
                "start_ms": round(a * 1000),
                "end_ms": round(b * 1000),
                "status": "unresolved",
                "reason": reason,
                **({"retry_evidence": evidence} if evidence else {}),
            }
        )

    for i, (a, b) in enumerate(ranges):
        check()
        progress(f"Window {i + 1} of {len(ranges)} · First pass")
        try:
            segments, _ = decode(a, b, 2048)
            candidates.append((i, a, b, segments, None))
        except WindowDecodeError as error:
            failed.append((i, a, b, str(error)))

    for i, a, b, reason in failed:
        check()
        evidence = retry_policy(a, b)
        if evidence:
            gap(a, b, reason, evidence)
            continue
        middle = (a + b) / 2
        context = min(2.0, (b - a) / 10)
        for start, end, left in [(a, middle + context, True), (middle - context, b, False)]:
            check()
            if budget["calls"] >= budget["max_calls"] or budget["tokens"] >= budget["max_tokens"]:
                gap(start, end, "Retry budget exhausted: " + reason)
                continue
            limit = min(budget["per_call_tokens"], budget["max_tokens"] - budget["tokens"])
            budget["calls"] += 1
            progress(
                f"Window {i + 1} of {len(ranges)} · Retry {budget['calls']} of {budget['max_calls']}"
            )
            try:
                segments, tokens = decode(start, end, limit)
                budget["tokens"] += tokens
                candidates.append((i, start, end, segments, (middle, left)))
            except WindowDecodeError as error:
                budget["tokens"] += error.tokens
                gap(start, end, str(error))
    return candidates, unresolved, budget


def reconcile_song(
    ranges, candidates, reconcile=lambda a, b, segments: segments, check=lambda: None
):
    """Reconcile complete candidates before either retry or outer-window ownership
    can discard words. All generation finishes before this phase starts."""
    results = [[] for _ in ranges]
    for i, a, b, segments, ownership in candidates:
        check()
        segments = reconcile(a, b, segments)
        if ownership is not None:
            middle, left = ownership
            segments = [
                s for s in segments if (((s["start_ms"] + s["end_ms"]) / 2000 < middle) == left)
            ]
        results[i].extend(segments)
    return results
