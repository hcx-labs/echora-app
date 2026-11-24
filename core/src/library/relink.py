"""Re-link analyzed tracks when the music server renumbers its songs.

Navidrome can change song IDs wholesale (a database upgrade did, in 0.64). The
audio is the same, so the analysis must follow the song instead of the whole
library looking new. A known source whose ID left the catalog is matched to a
catalog song Echora has not seen, by ISRC; recordings that share an ISRC (a
single and its album) and files without one are told apart by path. Only
one-to-one matches are applied.
"""

from __future__ import annotations

from collections import defaultdict

from psycopg.types.json import Jsonb

from core.library.visibility import lock_library, rebuild_links


def _isrcs(raw: dict) -> frozenset[str]:
    value = raw.get("isrc") or []
    values = [value] if isinstance(value, str) else value
    return frozenset(str(v).upper().replace("-", "").strip() for v in values if v)


def _file(raw: dict) -> tuple[str, str] | None:
    path = raw.get("path")
    return (str(path), str(raw.get("size"))) if path else None


def plan_relinks(known: dict[str, dict], catalog: dict[str, dict]) -> dict[str, str]:
    """Map vanished source IDs to the unseen catalog IDs that are the same song."""
    stale = {i: raw or {} for i, raw in known.items() if i not in catalog}
    fresh = {i: raw or {} for i, raw in catalog.items() if i not in known}
    if not stale or not fresh:
        return {}
    by_isrc, by_file = defaultdict(set), defaultdict(set)
    for new, raw in fresh.items():
        for code in _isrcs(raw):
            by_isrc[code].add(new)
        if _file(raw):
            by_file[_file(raw)].add(new)
    candidates: dict[str, str] = {}
    for old, raw in stale.items():
        same_recording = set().union(*(by_isrc[c] for c in _isrcs(raw))) if _isrcs(raw) else set()
        same_file = by_file.get(_file(raw), set()) if _file(raw) else set()
        if len(same_recording) == 1:
            choice = same_recording
        elif same_recording:  # the recording is in the library more than once
            choice = same_recording & same_file
        else:
            choice = same_file
        if len(choice) == 1:
            candidates[old] = next(iter(choice))
    claims = defaultdict(list)
    for old, new in candidates.items():
        claims[new].append(old)
    return {olds[0]: new for new, olds in claims.items() if len(olds) == 1}


def relink_sources(connection, library_id, catalog: dict[str, dict]) -> int:
    """Apply the plan for one library. The caller owns the transaction."""
    if not catalog:
        return 0
    lock_library(connection, library_id)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT external_id, source_data FROM track_sources"
            " WHERE library_id=%s AND source_type='subsonic'",
            (library_id,),
        )
        known = {str(row[0]): row[1] for row in cursor.fetchall()}
        plan = plan_relinks(known, catalog)
        if not plan:
            return 0
        olds, news = list(plan), list(plan.values())
        cursor.execute(
            "CREATE TEMP TABLE source_relink (old text PRIMARY KEY, new text UNIQUE, raw jsonb)"
            " ON COMMIT DROP"
        )
        cursor.executemany(
            "INSERT INTO source_relink VALUES (%s,%s,%s)",
            [(old, new, Jsonb(catalog[new])) for old, new in plan.items()],
        )
        cursor.execute(
            """UPDATE track_sources ts SET external_id=r.new,
                   source_data=coalesce(ts.source_data,'{}'::jsonb) || r.raw
               FROM source_relink r
               WHERE ts.library_id=%s AND ts.source_type='subsonic' AND ts.external_id=r.old""",
            (library_id,),
        )
        # The stored originals belong to the analyzed song, so they win over
        # anything already written under the new ID.
        cursor.execute(
            "DELETE FROM navidrome_lyrics_sources WHERE library_id=%s AND external_id=ANY(%s)",
            (library_id, news),
        )
        cursor.execute(
            """UPDATE navidrome_lyrics_sources n SET external_id=r.new
               FROM source_relink r WHERE n.library_id=%s AND n.external_id=r.old""",
            (library_id,),
        )
        cursor.execute(
            """INSERT INTO user_source_memberships(user_id,library_id,external_id)
               SELECT m.user_id,m.library_id,r.new FROM user_source_memberships m
               JOIN source_relink r ON r.old=m.external_id WHERE m.library_id=%s
               ON CONFLICT DO NOTHING""",
            (library_id,),
        )
        cursor.execute(
            "DELETE FROM user_source_memberships WHERE library_id=%s AND external_id=ANY(%s)",
            (library_id, olds),
        )
        cursor.execute(
            "SELECT DISTINCT user_id FROM user_source_memberships"
            " WHERE library_id=%s AND external_id=ANY(%s)",
            (library_id, news),
        )
        owners = [row[0] for row in cursor.fetchall()]
        cursor.execute("DROP TABLE source_relink")
    if owners:
        rebuild_links(connection, library_id, owners)
    return len(plan)
