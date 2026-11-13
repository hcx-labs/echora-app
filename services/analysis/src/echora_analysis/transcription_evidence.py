"""Conservative evidence for declining retries, never for skipping first attempts."""

import math


def retry_evidence(activity, start, end, duration, vocal_rms_dbfs):
    """Require near-complete classifier coverage AND a quiet separated stem.

    Scores are uncalibrated activations, not probabilities. Missing, mismatched,
    sparse or conflicting evidence must leave the retry policy unchanged.
    """
    if (
        not isinstance(activity, dict)
        or activity.get("method") != "discogs-effnet_voice_instrumental"
    ):
        return None
    try:
        if (
            not math.isfinite(vocal_rms_dbfs)
            or abs(float(activity["duration_seconds"]) - duration) > 1
        ):
            return None
        patches = []
        for w in activity["windows"]:
            a, b, v = (
                float(w["start_seconds"]),
                float(w["end_seconds"]),
                float(w["vocal_activation"]),
            )
            if not all(math.isfinite(x) for x in (a, b, v)) or b <= a or not 0 <= v <= 1:
                return None
            a, b = max(start, a), min(end, b)
            if b > a:
                patches.append((a, b, v))
        if not patches or end <= start:
            return None
        patches.sort()
        covered = 0
        edge = start
        weighted = 0
        for a, b, v in patches:
            length = max(0, b - max(edge, a))
            covered += length
            weighted += length * v
            edge = max(edge, b)
        coverage = covered / (end - start)
        mean = weighted / covered if covered else 1
        peak = max(v for _, _, v in patches)
        if coverage < 0.95 or mean > 0.05 or peak > 0.2 or vocal_rms_dbfs > -40:
            return None
        return {
            "decision": "decline_retry",
            "reason": "low_vocal_evidence",
            "coverage": coverage,
            "mean_vocal_activation": mean,
            "peak_vocal_activation": peak,
            "vocal_rms_dbfs": vocal_rms_dbfs,
            "confidence_calibrated": False,
            "thresholds": {
                "coverage_min": 0.95,
                "mean_max": 0.05,
                "peak_max": 0.2,
                "rms_dbfs_max": -40,
            },
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
