from __future__ import annotations

import numpy as np


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


def spherical_targets(start: np.ndarray, end: np.ndarray, length: int) -> np.ndarray:
    first, last = normalize_rows(np.stack([start, end]))
    cosine = float(np.clip(first @ last, -1, 1))
    angle = float(np.arccos(cosine))
    amounts = np.linspace(0, 1, length)
    if angle < 1e-6 or abs(np.sin(angle)) < 1e-6:
        return normalize_rows(np.stack([(1 - amount) * first + amount * last for amount in amounts]))
    sine = np.sin(angle)
    return np.stack([
        np.sin((1 - amount) * angle) / sine * first + np.sin(amount * angle) / sine * last
        for amount in amounts
    ]).astype(np.float32)


def multi_stop_targets(embeddings: np.ndarray, waypoint_indices: list[int],
                       length: int) -> np.ndarray:
    """Spherical interpolation through ordered waypoints.

    Each leg's share of the playlist tracks is proportional to its angular
    distance, so a long detour between stops gets proportionally more songs.
    """
    matrix = normalize_rows(embeddings)
    points = matrix[waypoint_indices]
    angles = [float(np.arccos(float(np.clip(points[i] @ points[i + 1], -1, 1))))
              for i in range(len(points) - 1)]
    total = sum(angles)
    if total < 1e-9:
        counts = [0] * len(angles)
    else:
        counts = [max(1, round(angle / total * (length - len(points)))) for angle in angles]
    # Fix rounding so intermediate targets plus endpoints equal the length.
    while sum(counts) > length - len(points):
        counts[counts.index(max(counts))] -= 1
    while sum(counts) < length - len(points):
        counts[counts.index(min(counts))] += 1
    targets = [points[0]]
    for i, count in enumerate(counts):
        leg = spherical_targets(points[i], points[i + 1], count + 2)
        targets.extend(leg[1:-1])
    targets.append(points[-1])
    return np.stack(targets).astype(np.float32)


def select_multistop_journey(
    embeddings: np.ndarray,
    waypoint_indices: list[int],
    length: int,
    artists: list[str | None],
    recording_groups: list[str | None],
) -> list[tuple[int, float, float]]:
    """Run the greedy journey selection leg by leg, sharing dedup state.

    Intermediate tracks are split across legs proportionally to each leg's
    angular distance, so a long detour between stops gets proportionally
    more songs. Waypoint tracks themselves always appear, in order.
    """
    matrix = normalize_rows(embeddings)
    points = matrix[waypoint_indices]
    angles = [float(np.arccos(float(np.clip(points[i] @ points[i + 1], -1, 1))))
              for i in range(len(points) - 1)]
    total = sum(angles)
    intermediates = max(length - len(waypoint_indices), 0)
    if total < 1e-9:
        shares = [1] * len(angles)
    else:
        shares = [max(1, round(angle / total * intermediates)) for angle in angles]
    while sum(shares) > intermediates and max(shares) > 1:
        shares[shares.index(max(shares))] -= 1
    while sum(shares) < intermediates:
        shares[shares.index(min(shares))] += 1

    used: set[int] = set()
    artist_counts: dict[str, int] = {}
    used_groups: set[str] = set()
    all_steps: list[tuple[int, float, float]] = []
    for leg, (start_index, end_index) in enumerate(zip(waypoint_indices, waypoint_indices[1:])):
        targets = spherical_targets(matrix[start_index], matrix[end_index], shares[leg] + 2)
        leg_steps = _select_leg(
            matrix, targets, start_index, end_index, artists, recording_groups,
            used, artist_counts, used_groups, is_first=leg == 0,
        )
        all_steps.extend(leg_steps if leg == 0 else leg_steps[1:])
    return all_steps


def _select_leg(
    matrix: np.ndarray,
    targets: np.ndarray,
    start_index: int,
    end_index: int,
    artists: list[str | None],
    recording_groups: list[str | None],
    used: set[int],
    artist_counts: dict[str, int],
    used_groups: set[str],
    is_first: bool,
) -> list[tuple[int, float, float]]:
    if not is_first:
        used.discard(end_index)  # the next leg's start was the previous endpoint
    used.add(start_index)
    used.add(end_index)
    if artists[start_index]:
        artist_counts[artists[start_index].casefold()] = artist_counts.get(artists[start_index].casefold(), 0) + 1
    for group in (recording_groups[start_index], recording_groups[end_index]):
        if group:
            used_groups.add(group)
    previous = start_index
    steps: list[tuple[int, float, float]] = [(start_index, 1.0, 0.0)]
    for position, target in enumerate(targets[1:-1], start=1):
        target_scores = matrix @ target
        previous_scores = matrix @ matrix[previous]
        progress = position / (len(targets) - 1)
        endpoint_axis = matrix @ normalize_rows((matrix[end_index] - matrix[start_index])[None, :])[0]
        candidates = np.argsort(target_scores + 0.12 * previous_scores + 0.04 * progress * endpoint_axis)[::-1]
        choice = None
        for candidate_value in candidates:
            candidate = int(candidate_value)
            if candidate in used:
                continue
            artist = artists[candidate].casefold() if artists[candidate] else None
            if artist and artist_counts.get(artist, 0) >= 2:
                continue
            group = recording_groups[candidate]
            if group and group in used_groups:
                continue
            choice = candidate
            break
        if choice is None:
            continue
        used.add(choice)
        artist = artists[choice].casefold() if artists[choice] else None
        if artist:
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
        if recording_groups[choice]:
            used_groups.add(recording_groups[choice])
        steps.append((choice, float(target_scores[choice]), float(matrix[previous] @ matrix[choice])))
        previous = choice
    steps.append((end_index, 1.0, float(matrix[previous] @ matrix[end_index])))
    if artists[end_index]:
        key = artists[end_index].casefold()
        artist_counts[key] = max(artist_counts.get(key, 1) - 1, 0)
    return steps


def select_journey(
    embeddings: np.ndarray,
    targets: np.ndarray,
    start_index: int,
    end_index: int,
    artists: list[str | None],
    recording_groups: list[str | None],
    max_per_artist: int = 2,
) -> list[tuple[int, float, float]]:
    matrix = normalize_rows(embeddings)
    selected = [start_index]
    used = {start_index, end_index}
    artist_counts: dict[str, int] = {}
    if artists[start_index]:
        artist_counts[artists[start_index].casefold()] = 1
    used_groups = {
        group for group in (recording_groups[start_index], recording_groups[end_index]) if group
    }
    previous = start_index
    steps: list[tuple[int, float, float]] = [(start_index, 1.0, 0.0)]
    for position, target in enumerate(targets[1:-1], start=1):
        target_scores = matrix @ target
        previous_scores = matrix @ matrix[previous]
        progress = position / (len(targets) - 1)
        endpoint_axis = matrix @ normalize_rows((matrix[end_index] - matrix[start_index])[None, :])[0]
        candidates = np.argsort(target_scores + 0.12 * previous_scores + 0.04 * progress * endpoint_axis)[::-1]
        choice = None
        for candidate_value in candidates:
            candidate = int(candidate_value)
            if candidate in used:
                continue
            artist = artists[candidate].casefold() if artists[candidate] else None
            if artist and artist_counts.get(artist, 0) >= max_per_artist:
                continue
            group = recording_groups[candidate]
            if group and group in used_groups:
                continue
            choice = candidate
            break
        if choice is None:
            continue
        used.add(choice); selected.append(choice)
        artist = artists[choice].casefold() if artists[choice] else None
        if artist:
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
        if recording_groups[choice]:
            used_groups.add(recording_groups[choice])
        steps.append((choice, float(target_scores[choice]), float(matrix[previous] @ matrix[choice])))
        previous = choice
    steps.append((end_index, 1.0, float(matrix[previous] @ matrix[end_index])))
    return steps
