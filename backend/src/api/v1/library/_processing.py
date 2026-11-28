"""What each pipeline holds for one song, against what the current code would produce."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from core.resync import store as resync
from core.resync.processing import track_outputs

LABELS = {
    "transcription": "AI lyrics",
    "karaoke": "Karaoke alignment",
    "karaoke_stems": "Karaoke stems",
    "translation": "Translations",
    "themes": "Lyric themes",
    "lyrics_embedding": "Lyrics embedding",
    "audio_embedding": "Audio embeddings",
}


def _entry(key, status, stored=None, current=None, detail=None, resyncable=True, **extra):
    return {
        "key": key,
        "label": LABELS[key],
        "status": status,
        "stored_revision": stored,
        "current_revision": current,
        "detail": detail,
        "resyncable": resyncable,
        **extra,
    }


def _state(stored, current) -> str:
    if stored is None:
        return "missing"
    return "current" if stored == current else "outdated"


def processing(connection, track_id: uuid.UUID) -> list[dict[str, Any]]:
    from core.embeddings.config import EmbeddingConfig
    from core.lyrics.translations import load_settings
    from core.revisions import KARAOKE_PIPELINE_REVISION
    from worker.kara.pipeline import _stored_model_revision, current_model_revision
    from worker.stems.pipeline import stems_recipe
    from worker.themes.pipeline import PROMPT_REVISION
    from worker.transcription.song import PIPELINE_REVISION as TRANSCRIPTION_REVISION

    outputs = track_outputs(connection, track_id)
    lyric = outputs["lyrics"]
    provenance = (lyric or {}).get("provenance") or {}
    has_text = bool(lyric and lyric["source"] not in (None, "none"))
    entries = []

    # Transcription: only songs whose lyrics were generated carry a revision.
    if lyric and lyric["source"] == "transcribed":
        stored = (provenance.get("transcription") or {}).get("pipeline_revision")
        entries.append(
            _entry(
                "transcription",
                _state(stored, TRANSCRIPTION_REVISION),
                stored,
                TRANSCRIPTION_REVISION,
                f"Transcribed in {lyric['language'] or 'an identified language'}",
                language=lyric["language"],
            )
        )
    else:
        source = {
            "embedded": "Lyrics from the library",
            "manual": "Lyrics edited by hand",
        }.get((lyric or {}).get("source"), "No lyrics")
        if provenance.get("manual_status") == "instrumental":
            source = "Marked instrumental"
        elif provenance.get("forced_transcription"):
            source = f"Transcription in {provenance.get('transcription_language')} pending"
        entries.append(
            _entry(
                "transcription",
                "pending" if provenance.get("forced_transcription") else "not_applicable",
                current=TRANSCRIPTION_REVISION,
                detail=source,
                language=(lyric or {}).get("language"),
            )
        )

    current = f"{KARAOKE_PIPELINE_REVISION} · {_stored_model_revision(current_model_revision())}"
    variant = outputs["karaoke"]
    if variant and variant["model"] == "navidrome":
        entries.append(
            _entry(
                "karaoke",
                "current",
                "word timing from the lyrics",
                current,
                "Word timing came with the lyrics; it is kept as is",
                resyncable=False,
            )
        )
    else:
        stored = (
            f"{variant['pipeline_revision']} · {variant['model_revision']}" if variant else None
        )
        entries.append(
            _entry(
                "karaoke",
                _state(stored, current) if has_text else "not_applicable",
                stored,
                current,
                None if has_text else "Needs lyrics",
                resyncable=has_text,
                updated_at=variant["created_at"] if variant else None,
            )
        )

    stems = outputs["stems"]
    recipe = stems_recipe()
    entries.append(
        _entry(
            "karaoke_stems",
            "failed"
            if stems and stems["status"] == "failed" and stems["recipe"] == recipe
            else _state(stems and stems["recipe"], recipe),
            stems and stems["recipe"],
            recipe,
            "The last attempt failed" if stems and stems["status"] == "failed" else None,
            resyncable=has_text,
            updated_at=stems["created_at"] if stems else None,
        )
    )

    settings, _ = load_settings()
    prompt_revision = hashlib.sha256(settings.prompt.encode()).hexdigest() if settings else None
    rows = outputs["translations"]
    current_translation = f"{settings.model or '-'} · prompt {(prompt_revision or '')[:12]}"
    translations = [
        {
            "source": row["source_language"],
            "target": row["target_language"],
            "status": row["status"]
            if row["status"] != "ready"
            else (
                "current"
                if row["model"] == settings.model and row["prompt_revision"] == prompt_revision
                else "outdated"
            ),
            "revision": f"{row['model']} · prompt {(row['prompt_revision'] or '')[:12]}",
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]
    statuses = {t["status"] for t in translations}
    entries.append(
        _entry(
            "translation",
            "missing"
            if not translations
            else next((s for s in ("failed", "stale", "outdated") if s in statuses), "current"),
            None,
            current_translation,
            None if settings.enabled else "Translation is off in Settings",
            resyncable=has_text and settings.enabled,
            translations=translations,
        )
    )

    themes = outputs["themes"]
    entries.append(
        _entry(
            "themes",
            "failed"
            if themes and themes["status"] == "failed"
            else _state(themes and str(themes["prompt_revision"]), str(PROMPT_REVISION)),
            themes and str(themes["prompt_revision"]),
            str(PROMPT_REVISION),
            f"Written by {themes['model']}" if themes and themes["model"] else None,
            resyncable=has_text,
            updated_at=themes["updated_at"] if themes else None,
        )
    )

    config = EmbeddingConfig.load()
    runs = outputs["embedding_runs"]
    entries.append(
        _entry(
            "lyrics_embedding",
            _state(runs.get("bge_m3"), config.lyrics_revision) if has_text else "not_applicable",
            runs.get("bge_m3"),
            config.lyrics_revision,
            "Redone with the lyrics",
            resyncable=False,
        )
    )
    audio_current = f"MuQ {config.muq_revision[:12]} · MERT {config.mert_revision[:12]}"
    audio_stored = (
        f"MuQ {(runs.get('muq_mulan') or '-')[:12]} · MERT {(runs.get('mert') or '-')[:12]}"
        if runs.get("muq_mulan") or runs.get("mert")
        else None
    )
    entries.append(
        _entry(
            "audio_embedding",
            _state(audio_stored, audio_current),
            audio_stored,
            audio_current,
            "Kept current by every full sync",
            resyncable=False,
        )
    )

    requests = resync.for_track(connection, track_id)
    for entry in entries:
        found = requests.get(entry["key"])
        entry["requested"] = (
            {"options": found["options"] or {}, "requested_at": found["requested_at"]}
            if found
            else None
        )
    return entries
