"""Bound SQL predicates. Match any selected value per group, all active groups."""

from fastapi import HTTPException


def _values(value, *, split_commas=True):
    """Keep legacy comma-separated codes; genre names may themselves contain commas."""
    if not value:
        return []
    values = [value] if isinstance(value, str) else value
    result = list(
        dict.fromkeys(
            part
            for item in values
            for part in (item.split(",") if split_commas else [item])
            if part
        )
    )
    if len(result) > 128:
        raise HTTPException(422, "Too many filter values")
    return result


def predicates(
    *, vocals="", language="", lyrics="", translation="", genre="", year_from=None, year_to=None
):
    clauses, params = [], []

    def add(sql, *values):
        clauses.append(sql)
        params.extend(values)

    if year_from is not None and year_to is not None and year_from > year_to:
        raise HTTPException(422, "Release year range is reversed")
    genres = _values(genre, split_commas=False)
    if len(genres) == 1:
        add("t.genres @> ARRAY[%s]::text[]", genres[0])
    elif genres:
        add("t.genres && %s::text[]", genres)
    if year_from is not None:
        add("t.year >= %s", year_from)
    if year_to is not None:
        add("t.year <= %s", year_to)
    languages = _values(language)
    if languages:
        known = [item for item in languages if item != "unknown"]
        pieces = []
        if known:
            pieces.append(
                "EXISTS (SELECT 1 FROM lyrics bl WHERE bl.track_id=t.id AND bl.language=ANY(%s))"
            )
            params.append(known)
        if "unknown" in languages:
            pieces.append(
                "NOT EXISTS (SELECT 1 FROM lyrics bl WHERE bl.track_id=t.id AND NULLIF(bl.language,'') IS NOT NULL)"
            )
        clauses.append("(" + " OR ".join(pieces) + ")")
    lyric_values = _values(lyrics)
    if lyric_values:
        choices = {
            "available": "EXISTS (SELECT 1 FROM lyrics bl WHERE bl.track_id=t.id AND NULLIF(btrim(bl.text),'') IS NOT NULL)",
            "missing": "NOT EXISTS (SELECT 1 FROM lyrics bl WHERE bl.track_id=t.id AND NULLIF(btrim(bl.text),'') IS NOT NULL)",
            "ai": "EXISTS (SELECT 1 FROM lyrics bl WHERE bl.track_id=t.id AND bl.provenance->>'ai_generated'='true')",
        }
        if any(value not in choices for value in lyric_values):
            raise HTTPException(422, "Unknown lyrics filter")
        clauses.append("(" + " OR ".join(choices[value] for value in lyric_values) + ")")
    translations = _values(translation)
    if len(translations) == 1:
        add(
            "EXISTS (SELECT 1 FROM lyric_translations bt WHERE bt.track_id=t.id AND bt.status='ready' AND bt.target_language=%s)",
            translations[0],
        )
    elif translations:
        add(
            "EXISTS (SELECT 1 FROM lyric_translations bt WHERE bt.track_id=t.id AND bt.status='ready' AND bt.target_language=ANY(%s))",
            translations,
        )
    vocal_values = _values(vocals)
    if vocal_values:
        if any(
            value not in {"instrumental", "vocal", "female", "male", "unknown"}
            for value in vocal_values
        ):
            raise HTTPException(422, "Unknown vocals filter")
        base = "SELECT 1 FROM current_embeddings be WHERE be.track_id=t.id AND be.embedding_type='voice-gender' AND be.window_index IS NULL"
        # Same instrumental threshold as curation; gender is model evidence, not identity.
        score = "(be.embedding::real[])"
        conditions = {
            "instrumental": f"{score}[1] >= 0.5",
            "vocal": f"{score}[1] < 0.5",
            "female": f"{score}[1] < 0.5 AND {score}[2] >= {score}[3]",
            "male": f"{score}[1] < 0.5 AND {score}[3] > {score}[2]",
        }
        pieces = [
            "NOT EXISTS (" + base + ")"
            if value == "unknown"
            else "EXISTS (" + base + " AND " + conditions[value] + ")"
            for value in vocal_values
        ]
        clauses.append("(" + " OR ".join(pieces) + ")")
    return clauses, params
