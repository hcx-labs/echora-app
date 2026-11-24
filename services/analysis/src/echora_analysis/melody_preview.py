"""Bounded display data from existing hum-search contours, without audio analysis."""

import math
import statistics


def melody_preview(source: str, pitch, voiced, hop_seconds: float, limit: int = 1024):
    if not math.isfinite(hop_seconds) or hop_seconds <= 0 or limit < 1:
        return None
    length = min(len(pitch), len(voiced))
    if not length:
        return None
    count = min(length, limit)
    points = []
    for index in range(count):
        start, end = index * length // count, (index + 1) * length // count
        values = [
            float(pitch[i])
            for i in range(start, end)
            if voiced[i] and math.isfinite(float(pitch[i]))
        ]
        # Preserve gaps instead of joining unrelated melodic phrases.
        value = round(statistics.median(values), 2) if len(values) >= (end - start) / 2 else None
        points.append(
            {"time_seconds": round((start + end - 1) / 2 * hop_seconds, 4), "pitch": value}
        )
    if not any(point["pitch"] is not None for point in points):
        return None
    return {"source": source, "points": points}
