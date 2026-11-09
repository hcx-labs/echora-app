"""Read-only native lyrics export. No network calls or analysis during playback."""

from __future__ import annotations

import hashlib
import json
import math
import re
from xml.etree import ElementTree as ET

from .navidrome_integration import connect
from .plugin_ranking import library_namespace
from .navidrome_lyrics_sources import lyrics_paused

TT = "http://www.w3.org/ns/ttml"
TTM = "http://www.w3.org/ns/ttml#metadata"
XML = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("", TT)
ET.register_namespace("ttm", TTM)


def original_lines(row):
    timed = (row.get("provenance") or {}).get("lines") or []
    if timed and all(
        isinstance(line, dict) and isinstance(line.get("text"), str) for line in timed
    ):
        return [line["text"] for line in timed]
    return (row.get("text") or "").splitlines()


def source_checksum(lines):
    return hashlib.sha256(
        json.dumps(lines, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def timing(value):
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        milliseconds = round(value)
        hours, remainder = divmod(milliseconds, 3600000)
        minutes, remainder = divmod(remainder, 60000)
        seconds, milliseconds = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return None


def timed_attributes(line):
    start, end = timing(line.get("start_ms")), timing(line.get("end_ms"))
    attrs = {"begin": start} if start else {}
    if end and (not start or line["end_ms"] >= line["start_ms"]):
        attrs["end"] = end
    return attrs


BACKING_VOICE = 2  # karaoke_pipeline.mark_backing_voice


def _append_syllables(paragraph, syllables):
    """Timed spans for a line's syllables. A run of second-voice syllables (``voice: 2``, the
    backing vocal the aligner placed on its own stream) is wrapped in a ``ttm:role="x-bg"`` span,
    the TTML form Apple Music uses and Navidrome's parser reads as a background agent, so clients
    can show it as backing instead of lead."""
    runs = []
    for syllable in syllables:
        backing = syllable.get("voice") == BACKING_VOICE
        if runs and runs[-1][0] == backing:
            runs[-1][1].append(syllable)
        else:
            runs.append((backing, [syllable]))
    for backing, run in runs:
        parent = paragraph
        if backing:
            starts = [s["start_ms"] for s in run if s.get("start_ms") is not None]
            ends = [s["end_ms"] for s in run if s.get("end_ms") is not None]
            bounds = {
                "start_ms": min(starts) if starts else None,
                "end_ms": max(ends) if ends else None,
            }
            parent = ET.SubElement(
                paragraph, f"{{{TT}}}span", {f"{{{TTM}}}role": "x-bg", **timed_attributes(bounds)}
            )
        for syllable in run:
            span = ET.SubElement(parent, f"{{{TT}}}span", timed_attributes(syllable))
            span.text = syllable.get("text") or ""


def export_lines(row):
    source = original_lines(row)
    karaoke = row.get("karaoke_lines") or []
    provenance = (row.get("provenance") or {}).get("lines") or []
    # Alignment removes empty stanzas and trims line edges. Map original IDs to
    # exported paragraphs explicitly instead of treating those changes as drift.
    if karaoke:
        source_nonempty = [(i, text.strip()) for i, text in enumerate(source) if text.strip()]
        karaoke_nonempty = [
            (i, line.get("text", "").strip())
            for i, line in enumerate(karaoke)
            if line.get("text", "").strip()
        ]
        if len(source_nonempty) == len(karaoke_nonempty):
            return karaoke, {
                source_id: exported_id
                for (source_id, _), (exported_id, _) in zip(source_nonempty, karaoke_nonempty)
            }
        # TTML metadata must reference a real main paragraph. When segmentation
        # differs, retain source paragraphs rather than discarding translations.
        # Reuse karaoke cues where a source paragraph can be matched in order.
        paragraphs = []
        next_karaoke = 0
        for index, text in enumerate(source):
            paragraph = {**(provenance[index] if index < len(provenance) else {}), "text": text}
            match = next(
                (
                    i
                    for i in range(next_karaoke, len(karaoke))
                    if karaoke[i].get("text", "").strip() == text.strip() and text.strip()
                ),
                None,
            )
            if match is not None:
                paragraph = {**karaoke[match], "text": text}
                next_karaoke = match + 1
            paragraphs.append(paragraph)
        return paragraphs, {i: i for i in range(len(source))}
    if provenance:
        return provenance, {i: i for i in range(len(source))}
    return [{"text": line} for line in source], {i: i for i in range(len(source))}


def safe_translations(row, translations, line_mapping):
    source = original_lines(row)
    checksum = source_checksum(source)
    selected = {}
    for item in translations:
        if item["status"] != "ready" or item["source_checksum"] != checksum:
            continue
        lines = item["lines"]
        if len(lines) != len(source) or any(
            not isinstance(line, dict)
            or line.get("id") != index
            or not isinstance(line.get("text"), str)
            for index, line in enumerate(lines)
        ):
            continue
        # Prefer provider translations over AI when both exist in one language.
        language = item["target_language"]
        if language not in selected or item["provenance"] == "provider":
            selected[language] = {
                **item,
                "lines": [
                    {**line, "id": line_mapping[line["id"]]}
                    for line in lines
                    if line["id"] in line_mapping
                    and line["text"].strip() != source[line["id"]].strip()
                ],
            }
    return [selected[key] for key in sorted(selected) if selected[key]["lines"]]


def ttml(row, translations):
    lines, line_mapping = export_lines(row)
    root = ET.Element(f"{{{TT}}}tt", {f"{{{XML}}}lang": row.get("language") or "und"})
    if translations:
        head = ET.SubElement(root, f"{{{TT}}}head")
        metadata = ET.SubElement(head, f"{{{TT}}}metadata")
        for item in safe_translations(row, translations, line_mapping):
            track = ET.SubElement(
                metadata, f"{{{TTM}}}translation", {f"{{{XML}}}lang": item["target_language"]}
            )
            for translated in item["lines"]:
                element = ET.SubElement(
                    track, f"{{{TTM}}}text", {"for": f"line-{translated['id']}"}
                )
                element.text = translated["text"]
    div = ET.SubElement(ET.SubElement(root, f"{{{TT}}}body"), f"{{{TT}}}div")
    for index, line in enumerate(lines):
        attributes = {"key": f"line-{index}", **timed_attributes(line)}
        if line.get("voice") == "background":
            # A whole line the aligner decoded as backing vocals ("(One, two, three)").
            attributes[f"{{{TTM}}}role"] = "x-bg"
        paragraph = ET.SubElement(div, f"{{{TT}}}p", attributes)
        syllables = line.get("syllables") or []
        # Never replace authoritative text with incomplete predicted tokens.
        if syllables and "".join(str(s.get("text") or "") for s in syllables) == line.get(
            "text", ""
        ):
            _append_syllables(paragraph, syllables)
        else:
            paragraph.text = line.get("text") or ""
    return ET.tostring(root, encoding="unicode")


def lrc(lines):
    def stamp(value):
        milliseconds = round(value)
        minutes, remainder = divmod(milliseconds, 60000)
        seconds, milliseconds = divmod(remainder, 1000)
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    result = []
    for line in lines:
        text = line.get("text") or ""
        if timing(line.get("start_ms")):
            text = f"[{stamp(line['start_ms'])}]{text}"
        result.append(text)
    return "\n".join(result)


def lyrics_response(row, translations, profile):
    if not row or not row.get("text") or not profile.serve_lyrics:
        return {"lyrics": []}
    language = row.get("language") or "und"
    # Reject language-tag injection into headers/XML attributes; XML escapes text.
    if not re.fullmatch(r"[a-zA-Z0-9-]{1,35}", language):
        language = "und"
    if profile.lyrics_format == "ttml":
        return {
            "lyrics": [
                {
                    "lang": language,
                    "text": ttml(row, translations if profile.include_translations else []),
                }
            ]
        }
    lines, line_mapping = export_lines(row)
    result = [{"lang": language, "text": lrc(lines)}]
    if profile.include_translations:
        for item in safe_translations(row, translations, line_mapping):
            mapped = [
                {**lines[translated["id"]], "text": translated["text"], "syllables": []}
                for translated in item["lines"]
            ]
            result.append({"lang": item["target_language"], "text": lrc(mapped)})
    return {"lyrics": result}


def load_lyrics(principal, source_id):
    if not principal["profile"].serve_lyrics:
        return {"lyrics": []}
    with connect() as db:
        if lyrics_paused(db, principal["url"]):
            # Empty is Navidrome's normal provider fallback signal. Never change
            # the saved preference or revoke the key to suppress lyrics alone.
            return {"lyrics": []}
        row = db.execute(
            """SELECT t.id,l.text,l.language,l.provenance,k.lines AS karaoke_lines
            FROM user_source_memberships visible
            JOIN libraries lib ON lib.id=visible.library_id AND lib.namespace=%s
            JOIN track_sources ts ON ts.library_id=visible.library_id AND ts.external_id=visible.external_id
                AND ts.source_type='subsonic'
            JOIN tracks t ON t.id=ts.track_id
            LEFT JOIN lyrics l ON l.track_id=t.id
            LEFT JOIN karaoke_lyrics_variants k ON k.track_id=t.id AND k.bounded=false
            WHERE visible.user_id=%s AND visible.external_id=%s LIMIT 1""",
            (library_namespace(principal), principal["user_id"], source_id),
        ).fetchone()
        translations = (
            db.execute(
                """SELECT * FROM lyric_translations
            WHERE track_id=%s AND status='ready' ORDER BY target_language,provenance""",
                (row["id"],),
            ).fetchall()
            if row
            else []
        )
    return lyrics_response(row, translations, principal["profile"])
