"""The karaoke stage: aligns each song's lyrics to its audio and stores the word timing."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import selectors
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any

from core.db.connection import connect
from core.errors import InputUnavailableError, ProcessingError
from core.integrations.navidrome import NavidromeClient
from core.lyrics import store
from core.lyrics.backing import BACKING_VOICE, tag_bracketed_backing
from core.revisions import KARAOKE_PIPELINE_REVISION
from worker.kara.aligner import label_rules
from worker.kara.aligner.request import AlignRequest
from worker.kara.config import KaraConfig
from worker.kara.furigana import ruby_segments
from worker.media.preprocessing import get_check
from worker.planning.audio import plan_karaoke, resolve_library_id
from worker.processing import Item, ItemResult, ProcessingContext, Processor

logger = logging.getLogger(__name__)

FA_KARA_REVISION = "168ca5f01cecaa1290e31c0ce8dc44af8c7451bb"

DEFAULT_MODEL_ID = "hcxlabs/echora-kara-700m-muq"
DEFAULT_MODEL_REVISION = "7c0cda150c8be886ed5ff2f903460f112cf4c7a2"
_KARAOKE_LOCK = threading.Lock()
_FA_KARA_WORKER: subprocess.Popen[str] | None = None
_FA_KARA_WORKER_KEY: tuple[str, ...] | None = None


@functools.lru_cache(maxsize=4)
def _checkpoint_digest(weights: str, size: int, modified_ns: int) -> str:
    """Content digest of a local checkpoint's weights; cached per file version."""
    digest = hashlib.sha256()
    with Path(weights).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def current_model_revision() -> str:
    """The aligner identity that planning, alignment and storage share.

    The pinned revision, or for FA_KARA_MODEL_PATH the checkpoint's own content,
    so replacing a local checkpoint makes existing alignments stale even when
    FA_KARA_REVISION is unchanged.
    """
    path = KaraConfig.load().model_path
    if not path:
        return KaraConfig.load().revision
    candidates = [Path(path) / name for name in ("model.safetensors", "pytorch_model.bin")]
    weights = next((candidate for candidate in candidates if candidate.is_file()), None)
    if weights is None:
        raise RuntimeError(f"FA-Kara checkpoint has no weights file: {path}")
    stat = weights.stat()
    return f"local-{_checkpoint_digest(str(weights), stat.st_size, stat.st_mtime_ns)}"


def _stored_model_revision(model_revision: str) -> str:
    # The suffix names the audio the aligner hears: the full mix, with or without the reading of
    # the separated vocals added. Stored alignments carry it, so changing the setting redoes them.
    audio = "mix+stem" if KaraConfig.load().stem_reading else "mix"
    return f"{FA_KARA_REVISION}:{model_revision}:{audio}"


def _prepare_vocals(source: bytes, check) -> None:
    """Separate a song's vocals into the prepared-audio cache, for the stem reading."""
    from worker.media.preprocessing import prepare_audio

    prepare_audio(source, vocals=True, check=check)


def _write_stem(audio: bytes, path: Path) -> str:
    """Write the song's separated vocals to ``path``; returns the separator's revision.

    Read from the prepared-audio cache when the stage separated them first, separated here
    otherwise.
    """
    import soundfile as sf

    from worker.media.preprocessing import vocal_stereo
    from worker.media.roformer import SEPARATION_REVISION

    sf.write(str(path), vocal_stereo(audio, get_check()), 44100, subtype="FLOAT")
    return SEPARATION_REVISION


def _restore_display_text(
    source_text: str,
    syllables: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Restore punctuation and spacing omitted from acoustic CTC tokens."""
    if not source_text or not syllables:
        return syllables
    positions: list[tuple[int, int]] = []
    cursor = 0
    folded_source = source_text.casefold()
    for syllable in syllables:
        token_text = str(syllable.get("text") or "")
        if not token_text:
            return syllables
        start = source_text.find(token_text, cursor)
        if start < 0:
            start = folded_source.find(token_text.casefold(), cursor)
        if start < 0:
            return syllables
        end = start + len(token_text)
        positions.append((start, end))
        cursor = end
    restored = [dict(syllable) for syllable in syllables]
    for index, syllable in enumerate(restored):
        start = 0 if index == 0 else positions[index][0]
        end = positions[index + 1][0] if index + 1 < len(positions) else len(source_text)
        syllable["text"] = source_text[start:end]
        # Leading text restored before the first token (an opening 「) shifts its ruby offsets.
        shift = positions[index][0] - start
        if shift and syllable.get("ruby"):
            syllable["ruby"] = [
                {
                    **segment,
                    "start": int(segment["start"]) + shift,
                    "end": int(segment["end"]) + shift,
                }
                for segment in syllable["ruby"]
            ]
    return restored


def build_lines_from_alignment_document(document: dict[str, object]) -> list[dict[str, object]]:
    r"""Build karaoke lines from the alignment document's millisecond-precision timing.

    The alignment document stores token onset and offset in integer milliseconds
    derived directly from frame indices, avoiding the centisecond quantization
    inherent in the ASS ``\k`` tag format (10 ms resolution).  Using it as the
    primary timing source eliminates the systematic rounding error that
    accumulates across dense syllable sequences.
    """
    alignment = document.get("alignment")
    if not isinstance(alignment, dict):
        return []
    canonical_lines = alignment.get("lines")
    if not isinstance(canonical_lines, list):
        return []
    diagnostics = document.get("diagnostics")
    # Japanese kanji groups get furigana from their timed morae (furigana.py).
    japanese = isinstance(diagnostics, dict) and str(
        diagnostics.get("normalization_language") or ""
    ).startswith("ja")
    lines: list[dict[str, object]] = []
    for line_record in canonical_lines:
        if not isinstance(line_record, dict):
            continue
        tokens = line_record.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            continue
        syllables: list[dict[str, object]] = []
        morae: list[list[dict[str, object]]] = []
        for token in tokens:
            if not isinstance(token, dict):
                continue
            text = str(token.get("text") or "")
            start_ms = token.get("start_ms")
            end_ms = token.get("end_ms")
            if not isinstance(start_ms, int) or not isinstance(end_ms, int):
                continue
            mora = {
                "unit": str(token.get("acoustic_token") or ""),
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
            if text:
                syllables.append(
                    {
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": text,
                        **(
                            {"stream": SECOND_STREAM}
                            if token.get("stream") == SECOND_STREAM
                            else {}
                        ),
                    }
                )
                morae.append([mora])
            elif syllables:
                # Acoustic continuation of an indivisible glyph group (or ruby).
                # Keep its full duration without repeating display characters.
                syllables[-1]["end_ms"] = max(int(syllables[-1]["end_ms"]), end_ms)
                morae[-1].append(mora)
        if not syllables:
            continue
        source_text = str(line_record.get("text") or "")
        if japanese:
            for syllable, units in zip(syllables, morae):
                ruby = ruby_segments(str(syllable["text"]), units)
                if ruby:
                    syllable["ruby"] = ruby
        syllables = _restore_display_text(source_text, syllables)
        text = source_text or "".join(str(s["text"]) for s in syllables)
        line_start = int(line_record.get("start_ms") or syllables[0]["start_ms"])
        line_end = int(line_record.get("end_ms") or syllables[-1]["end_ms"])
        lines.append(
            {"start_ms": line_start, "end_ms": line_end, "text": text, "syllables": syllables}
        )
    return lines


BACKING_MIN_OVERLAP_MS = 80  # four aligner frames: shorter overlaps are boundary noise
SECOND_STREAM = 2  # syllable["stream"]: the aligner placed it on its second (overlapping) stream
WORD_END = re.compile(r"\s[(\[]*$")


def mark_backing_voice(lines: list[dict[str, object]]) -> list[dict[str, object]]:
    """Tag backing vocals on syllables (``"voice": 2``) and on whole lines (``"background"``).

    Players show ``"voice": 2`` syllables as a second, smaller line; ``"voice": "background"``
    marks a line that is backing as a whole.

    Bracketed words are always backing vocals (``tag_bracketed_backing``), inline or as a whole
    line, wherever the aligner placed them. On top of that, a phrase (consecutive words of a line) that the
    aligner placed on its second stream (``"stream": 2``) is a second voice. With inline backing
    placement on (``fa_kara_inline_backing``) every such phrase beside lead words is one, since the
    aligner then puts each inline bracket phrase on that stream; otherwise only a phrase that
    overlaps the line's other words by at least ``BACKING_MIN_OVERLAP_MS`` qualifies and the rest
    stay on the line. Overlaps between lines are not marked. Stale marks are cleared.
    """
    for line in lines:
        words: list[list[dict[str, object]]] = []
        current: list[dict[str, object]] = []
        for syllable in line.get("syllables") or []:
            syllable.pop("voice", None)
            current.append(syllable)
            # A word ends at whitespace, also when an opening bracket for the next word is
            # glued after it ("won't (" then "but").
            if WORD_END.search(str(syllable.get("text", ""))):
                words.append(current)
                current = []
        if current:
            words.append(current)
        phrases: list[list[list[dict[str, object]]]] = []
        others: list[tuple[int, int]] = []
        previous_second = False
        for word in words:
            second = any(syllable.get("stream") == SECOND_STREAM for syllable in word)
            if second:
                if not previous_second:
                    phrases.append([])
                phrases[-1].append(word)
            else:
                others.append((int(word[0].get("start_ms", 0)), int(word[-1].get("end_ms", 0))))
            previous_second = second
        for phrase in phrases:
            start = int(phrase[0][0].get("start_ms", 0))
            end = max(int(word[-1].get("end_ms", 0)) for word in phrase)
            # With inline backing placement every inline bracket phrase is on the second stream,
            # so it is shown as the second voice whether or not it overlaps the lead; otherwise
            # only a simultaneous phrase qualifies.
            inline = bool(others) and KaraConfig.load().inline_backing
            if inline or any(
                min(end, e) - max(start, s) >= BACKING_MIN_OVERLAP_MS for s, e in others
            ):
                for word in phrase:
                    for syllable in word:
                        syllable["voice"] = BACKING_VOICE
    return tag_bracketed_backing(lines)


def apply_adaptive_line_padding(lines: list[dict[str, object]]) -> list[dict[str, object]]:
    """Pad line containers only inside silence between aligned syllables."""
    padded = [
        {**line, "syllables": [dict(item) for item in line.get("syllables") or []]}
        for line in lines
    ]
    timed: list[tuple[int, int, int, int] | None] = []
    for line in padded:
        syllables = [
            item
            for item in line["syllables"]
            if int(item.get("end_ms", 0)) > int(item.get("start_ms", 0))
        ]
        if not syllables:
            timed.append(None)
            continue
        first = syllables[0]
        last = syllables[-1]
        first_start, first_end = int(first["start_ms"]), int(first["end_ms"])
        last_start, last_end = int(last["start_ms"]), int(last["end_ms"])
        timed.append((first_start, first_end, last_start, last_end))
        line["start_ms"] = max(0, first_start - min(200, max(0, (first_end - first_start) // 4)))
        line["end_ms"] = last_end + min(200, max(0, (last_end - last_start) // 4))

    for index in range(1, len(padded)):
        previous = timed[index - 1]
        current = timed[index]
        if previous is None or current is None:
            continue
        previous_last_start, previous_last_end = previous[2], previous[3]
        current_first_start, current_first_end = current[0], current[1]
        gap = max(0, current_first_start - previous_last_end)
        desired_start_padding = min(200, max(0, (current_first_end - current_first_start) // 4))
        start_padding = min(desired_start_padding, gap)
        remaining_gap = gap - start_padding
        desired_end_padding = min(200, max(0, (previous_last_end - previous_last_start) // 4))
        end_padding = min(desired_end_padding, remaining_gap)
        padded[index]["start_ms"] = current_first_start - start_padding
        padded[index - 1]["end_ms"] = previous_last_end + end_padding
    return padded


def _line_key(text: object) -> str:
    return re.sub(r"[^\w]+", "", str(text or "").casefold())


def _validate_alignment_document(document: object) -> dict[str, object]:
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise RuntimeError("FA-Kara returned an unsupported alignment document")
    alignment = document.get("alignment")
    if not isinstance(alignment, dict) or not isinstance(alignment.get("lines"), list):
        raise RuntimeError("FA-Kara alignment document has no line records")
    # Timing is monotonic per voice: background lines (B0) are decoded on their own path and may
    # overlap the lead, and so are second-stream words (optional inline brackets).
    previous_ends = {"lead": 0, "background": 0, "lead stream 2": 0, "background stream 2": 0}
    token_count = 0
    for line_index, line in enumerate(alignment["lines"]):
        if not isinstance(line, dict) or line.get("source_index") != line_index:
            raise RuntimeError("FA-Kara alignment document has invalid source indexes")
        tokens = line.get("tokens")
        display_only = all(
            c.isspace() or unicodedata.category(c).startswith("P")
            for c in str(line.get("text") or "")
        )
        if tokens == [] and display_only:
            continue
        if not isinstance(tokens, list) or not tokens:
            raise RuntimeError(f"FA-Kara line {line_index} contains no aligned tokens")
        line_voice = "background" if line.get("voice") == "background" else "lead"
        for token in tokens:
            if not isinstance(token, dict):
                raise RuntimeError("FA-Kara alignment document contains an invalid token")
            voice = f"{line_voice} stream 2" if token.get("stream") == SECOND_STREAM else line_voice
            previous_end = previous_ends[voice]
            start = token.get("start_ms")
            end = token.get("end_ms")
            score = token.get("ctc_score")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or end < start
                or start < previous_end
            ):
                raise RuntimeError(
                    "FA-Kara alignment document contains non-monotonic timing"
                    f" (line {line_index}, {voice} voice, start {start} ms after {previous_end} ms)"
                )
            if not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
                raise RuntimeError("FA-Kara alignment document contains an invalid CTC score")
            previous_ends[voice] = end
            token_count += 1
    if token_count == 0:
        raise RuntimeError("FA-Kara alignment document contains no tokens")
    return document


def _fa_kara_worker(model_revision: str) -> subprocess.Popen[str]:
    global _FA_KARA_WORKER, _FA_KARA_WORKER_KEY
    # A worker holds one checkpoint, with or without its text head (loaded for a text bonus).
    key = (model_revision, KaraConfig.load().text_bonus > 0)
    if (
        _FA_KARA_WORKER is not None
        and _FA_KARA_WORKER.poll() is None
        and key == _FA_KARA_WORKER_KEY
    ):
        return _FA_KARA_WORKER
    if _FA_KARA_WORKER is not None and _FA_KARA_WORKER.poll() is None:
        _FA_KARA_WORKER.terminate()
    numba_cache = Path("/tmp/echora-fa-kara-worker-numba")
    numba_cache.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "NUMBA_CACHE_DIR": str(numba_cache),
        # The worker imports this package as the parent process does.
        "PYTHONPATH": os.pathsep.join(filter(None, [*sys.path, os.environ.get("PYTHONPATH")])),
    }
    _FA_KARA_WORKER = subprocess.Popen(
        [sys.executable, "-m", "worker.kara.aligner.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        env=environment,
    )
    _FA_KARA_WORKER_KEY = key
    return _FA_KARA_WORKER


def _stop_fa_kara_worker() -> None:
    global _FA_KARA_WORKER, _FA_KARA_WORKER_KEY
    worker = _FA_KARA_WORKER
    _FA_KARA_WORKER = None
    _FA_KARA_WORKER_KEY = None
    if worker is None or worker.poll() is not None:
        return
    worker.terminate()
    try:
        worker.wait(timeout=KaraConfig.load().shutdown_grace_seconds)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait(timeout=KaraConfig.load().kill_grace_seconds)


def _run_worker_job(
    worker: subprocess.Popen[str], request: AlignRequest, timeout: float | None = None
) -> dict[str, object]:
    if worker.stdin is None or worker.stdout is None:
        raise RuntimeError("FA-Kara worker pipes are unavailable")
    worker.stdin.write(request.model_dump_json() + "\n")
    worker.stdin.flush()
    selector = selectors.DefaultSelector()
    selector.register(worker.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + (
        KaraConfig.load().job_timeout_seconds if timeout is None else timeout
    )
    response_line = bytearray()
    try:
        # Read bytes rather than blocking on readline after a partial response.
        while b"\n" not in response_line:
            get_check()()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("FA-Kara worker timed out")
            if not selector.select(min(1.0, remaining)):
                continue
            chunk = os.read(worker.stdout.fileno(), 65536)
            if not chunk:
                break
            response_line.extend(chunk)
            if len(response_line) > 64 * 1024**2:
                raise RuntimeError("FA-Kara response exceeds 64 MiB")
        get_check()()
    except BaseException:
        if worker.poll() is None:
            worker.kill()
        worker.wait(timeout=KaraConfig.load().kill_grace_seconds)
        raise
    finally:
        selector.close()
    if not response_line:
        raise RuntimeError(f"FA-Kara worker exited unexpectedly with code {worker.poll()}")
    response = json.loads(response_line)
    if not isinstance(response, dict):
        raise RuntimeError("FA-Kara worker returned an invalid response")
    return response


def _source_line_texts(source: list[dict[str, object]]) -> list[str]:
    """Synced lyrics give the line breaks; their times are not used."""
    texts = [str(line.get("text") or "").strip() for line in source]
    return texts if any(texts) else []


# Lyric languages forwarded to the normalizer as ``--lang`` per label rule set (settings
# fa_kara_label_rules). v1: the allowlist every shipped model was trained with (en/es/fr/zh reach
# the normalizer as auto). v2 (2026-10-18 label round) also forwards en/de/fr (pronunciation labels),
# tr (Turkish rules), yue and zh; a v2 model mislabels Latin lines without it, a v1 model with it.
FA_KARA_LANGUAGES_V1 = frozenset({"auto", "ja", "jaen", "zhen", "ko", "ur", "hi", "pa", "indic"})
FA_KARA_LANGUAGES_V2 = FA_KARA_LANGUAGES_V1 | frozenset({"en", "de", "fr", "tr", "yue", "zh"})

# v3 also forwards every language whose own script is not Latin.
FA_KARA_LANGUAGES_V3 = FA_KARA_LANGUAGES_V2 | label_rules.NON_LATIN_SCRIPT_LANGUAGES


# Stored lyric languages that mean "not identified" (ISO 639 und; fastText's placeholder xxx).
UNKNOWN_LANGUAGES = frozenset({"und", "xxx"})


def fa_kara_languages(label_rules: str) -> frozenset[str]:
    return (
        FA_KARA_LANGUAGES_V1
        if label_rules == "v1"
        else FA_KARA_LANGUAGES_V3
        if label_rules in ("v3", "v3-orthographic", "v3-sung")
        else FA_KARA_LANGUAGES_V2
    )


def _run_fa_kara(
    audio: bytes,
    lyrics_text: str,
    language: str | None,
    source_lines: list[dict[str, object]] | None = None,
    provided_audio: bool = False,
) -> dict[str, object]:
    """Align lyrics to the song's full mix, or (``provided_audio``) to audio the caller cut.

    With ``stem_reading`` the song's separated vocals are read as well and added to the mix's
    reading (the mix stays the reference). Audio the caller cut is aligned as given.
    """
    model_id = KaraConfig.load().model_id
    model_revision = current_model_revision()
    snapshot = (
        Path(os.environ.get("HF_HOME", "/models/huggingface"))
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / KaraConfig.load().revision
    )
    if KaraConfig.load().model_path:
        # A local checkpoint; current_model_revision identifies it by its weights.
        snapshot = Path(KaraConfig.load().model_path)
        model_id = str(snapshot)
    if not snapshot.is_dir():
        raise RuntimeError(f"FA-Kara model snapshot is missing: {model_id}@{model_revision}")
    with tempfile.TemporaryDirectory(prefix="echora-fa-kara-") as directory:
        work = Path(directory)
        audio_path = work / "i.audio"
        audio_path.write_bytes(audio)
        line_texts = _source_line_texts(source_lines or [])
        input_text = "\n".join(line_texts) if line_texts else lyrics_text.rstrip("\r\n")
        config = KaraConfig.load()
        languages = fa_kara_languages(config.label_rules)
        stem_path, separator_revision = None, None
        if config.stem_reading and not provided_audio:
            stem_path = str(work / "stem.wav")
            separator_revision = _write_stem(audio, Path(stem_path))
        request = AlignRequest(
            audio_path=str(audio_path),
            stem_path=stem_path,
            text=input_text,
            model_path=str(snapshot),
            language=language if language in languages else "auto",
            # Lyrics stored without a language (ingest-time detection failed or was unsure): the
            # aligner identifies them itself rather than spell every word letter by letter.
            detect_language=tuple(sorted(languages))
            if not language or language in UNKNOWN_LANGUAGES
            else (),
            label_rules=config.label_rules,
            min_letter_ms=config.min_letter_ms,
            min_token_ms=config.min_token_ms,
            timing_offset_ms=config.timing_offset_ms,
            duration_pace_ratio=config.duration_pace_ratio,
            vocal_activity_weight=config.vocal_activity_weight,
            exact_frame_clock=config.exact_frame_clock,
            background_decoding=config.background_decoding,
            blank_free_tokens=config.blank_free_tokens,
            inline_backing=config.inline_backing,
            inline_text_bonus=config.inline_text_bonus,
            gap_onset_pull=config.gap_onset_pull,
            text_bonus=config.text_bonus,
            in_line_gap_cost=config.in_line_gap_cost,
            backing_rest=config.backing_rest,
            fusion=config.fusion,
            audio_speed=config.audio_speed,
        )
        worker = _fa_kara_worker(model_revision)
        response = _run_worker_job(worker, request)
        if not response.get("ok"):
            detail = str(
                response.get("traceback") or response.get("error") or "unknown worker error"
            )
            raise RuntimeError(f"FA-Kara failed: {detail[-4000:]}")
        alignment_document = _validate_alignment_document(response["document"])
        diagnostics = alignment_document.setdefault("diagnostics", {})
        diagnostics["audio_source"] = "provided_audio" if provided_audio else "original_mix"
        diagnostics["separator"] = None
        diagnostics["separator_revision"] = separator_revision
        lines = build_lines_from_alignment_document(alignment_document)
        # Keep the model's token-derived line boundaries. The former display
        # padding expanded every line into nearby silence, which moved otherwise
        # correct starts and ends away from the timed-lyrics reference.
        if not lines:
            raise RuntimeError("FA-Kara produced no aligned lyric lines")
    return {
        "lines": lines,
        "alignment_document": alignment_document,
        "diagnostics": alignment_document.get("diagnostics", {}),
        "model": model_id,
        "model_revision": model_revision,
    }


def _store_supplied_karaoke(connection, track_id, lines, digest) -> None:
    """Word timing that came with the lyrics, stored as the karaoke without alignment.

    TTML or enhanced LRC from Navidrome. Not AI-generated, so players show no AI disclosure for it.
    """
    provenance = {
        "alignment_mode": "supplied",
        "source": "navidrome",
        "source_digest": digest,
        "pipeline_revision": KARAOKE_PIPELINE_REVISION,
        "ai_generated": False,
    }
    store.save_supplied_karaoke(connection, track_id, lines, provenance)
    connection.commit()


class KaraokeItem(Item):
    """A song whose karaoke timing is missing or stale."""

    track_id: uuid.UUID
    external_id: str
    text: str
    language: str | None = None
    # Stored JSON, kept as read: one song's malformed value fails that song in ``process``,
    # not the batch's plan.
    source_lines: Any = None
    supplied: Any = None
    """Word timing the lyrics came with; stored as it is instead of aligning."""
    supplied_digest: str | None = None

    def malformed(self) -> bool:
        """Whether the stored lines or word timing are not lists of objects."""
        return not all(
            value is None
            or (isinstance(value, list) and all(isinstance(entry, dict) for entry in value))
            for value in (self.source_lines, self.supplied)
        )


class KaraokeProcessor(Processor[None, KaraokeItem, dict[str, object]]):
    """Aligns the batch's lyrics to their audio with the Echora Kara aligner.

    Songs whose lyrics came with word timing keep it (``supplied``). The others are prepared
    first, every one of them (into the batch's source cache, or up to Modal); then the aligner
    loads once and aligns them all (on Modal in one call). One batch aligns at a
    time in this worker; the aligner process stops after the stage.
    """

    name = "karaoke"
    summary_key = "karaoke"
    counters = ("aligned",)
    activity = "Aligning karaoke lyrics for"
    current_message = "Karaoke alignment already current"

    def plan(self, context: ProcessingContext) -> list[KaraokeItem]:
        with connect() as connection:
            self.library_id = resolve_library_id(connection, context.library_url)
            model_revision = _stored_model_revision(current_model_revision())
            planned = plan_karaoke(
                connection,
                KARAOKE_PIPELINE_REVISION,
                list(context.track_ids),
                model_revision,
                library_id=self.library_id,
            ).karaoke_external_ids
            if not planned:
                return []
            rows = store.karaoke_sources(connection, list(planned), self.library_id)
        items = []
        for row in rows:
            supplied, digest = row.get("word_timing"), row.get("word_timing_digest")
            items.append(
                KaraokeItem(
                    key=str(row["track_id"]),
                    label=row["title"] or "",
                    track_id=row["track_id"],
                    external_id=str(row["external_id"]),
                    text=row["text"] or "",
                    language=row["language"],
                    source_lines=row["lines"],
                    supplied=supplied if supplied and digest else None,
                    supplied_digest=digest if supplied and digest else None,
                )
            )
        return items

    def open(self, context: ProcessingContext, items) -> None:
        # One batch aligns at a time: the aligner and vocal separation share the GPU.
        while not _KARAOKE_LOCK.acquire(timeout=KaraConfig.load().lock_poll_seconds):
            context.check()
        self.locked = True
        context.check()
        self.connection = connect()
        self.client = NavidromeClient(*context.library)
        self.client.__enter__()

    def process(self, context, items):
        for item in items:
            if item.malformed():
                logger.warning("Stored lyric lines are malformed for %s", item.key)
                yield ItemResult.failed(item, ProcessingError("Stored lyrics are malformed"))
            elif item.supplied is not None:
                yield ItemResult(item=item, outcome="supplied", output={"supplied": True})
        aligning = [item for item in items if item.supplied is None and not item.malformed()]
        if not aligning:
            return
        _stop_fa_kara_worker()
        prepared, failed = self._prepare(context, aligning)
        ready = [item for item in aligning if item.key in prepared]
        if ready:
            context.publish(
                {
                    "phase": "models",
                    "message": "Loading Echora alignment model",
                    "completed": 0,
                    "total": 1,
                    "unit": "models",
                }
            )
        for item in failed:
            yield ItemResult.failed(item, InputUnavailableError("Karaoke audio unavailable"))
        if not ready:
            return
        aligned = None
        if context.remote is not None:
            # Every prepared song aligns in one Modal call; results stream back in order.
            aligned = iter(
                context.remote.karaoke(
                    [prepared[item.key] for item in ready],
                    [item.text for item in ready],
                    [item.language for item in ready],
                    [item.source_lines or [] for item in ready],
                )
            )
        for item in ready:
            context.check()
            try:
                if aligned is None:
                    result = _run_fa_kara(
                        self.client.audio_bytes(item.external_id),
                        item.text,
                        item.language,
                        item.source_lines or [],
                    )
                else:
                    result = next(aligned)
                    if isinstance(result, BaseException):
                        raise result
            except Exception as error:  # noqa: BLE001 - one song's alignment
                logger.exception("FA-Kara alignment failed for %s", item.key)
                failure = ProcessingError("Alignment failed", stage=type(error).__name__)
                yield ItemResult.failed(item, failure)
                continue
            yield ItemResult(item=item, outcome="aligned", output=result)

    def _prepare(self, context, items):
        """Fetch every song (into the batch's source cache, or up to Modal) before the aligner loads.

        With the stem reading, each song's vocals are separated here too (one separator for the
        batch, released before the aligner loads; on Modal, one separation phase).
        Returns the prepared songs' keys (with their Modal upload digest) and the failed songs.
        """
        from worker.media.roformer import separation_phase

        with separation_phase():
            prepared, failed = self._fetch(context, items)
        stem_reading = KaraConfig.load().stem_reading
        if context.remote is not None and stem_reading and prepared:
            ready = [item for item in items if item.key in prepared]
            separated = context.remote.prepare_vocals(
                [prepared[item.key] for item in ready], vocals=True
            )
            for item, outcome in zip(ready, separated):
                if isinstance(outcome, BaseException):
                    logger.error("Vocal separation on Modal failed for %s", item.key)
                    del prepared[item.key]
                    failed.append(item)
        return prepared, failed

    def _fetch(self, context, items):
        stem_reading = KaraConfig.load().stem_reading
        prepared: dict[str, str | None] = {}
        failed = []
        for index, item in enumerate(items):
            update = {
                "phase": "preprocess",
                "message": f"Preparing karaoke audio for {item.label}",
                "total": len(items),
                "unit": "tracks",
            }
            context.publish({**update, "completed": index})
            context.check()
            try:
                source = self.client.audio_bytes(item.external_id)
                digest = None
                if context.remote is not None:
                    digest = context.remote.upload_audio(source)
                elif stem_reading:
                    _prepare_vocals(source, context.check)
                prepared[item.key] = digest
                del source
            except Exception as error:  # noqa: BLE001 - one song's audio
                logger.error(
                    "Karaoke preprocessing failed for %s: %s", item.key, type(error).__name__
                )
                failed.append(item)
            context.publish({**update, "completed": index + 1})
        return prepared, failed

    def persist(self, context, item: KaraokeItem, output: dict[str, object]) -> None:
        try:
            if item.supplied is not None:
                _store_supplied_karaoke(
                    self.connection, item.track_id, item.supplied, item.supplied_digest
                )
            else:
                _store_alignment(self.connection, item.track_id, output)
                self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def close(self) -> None:
        try:
            if hasattr(self, "client"):
                self.client.__exit__(None, None, None)
            if hasattr(self, "connection"):
                self.connection.close()
            _stop_fa_kara_worker()
        finally:
            if getattr(self, "locked", False):
                self.locked = False
                _KARAOKE_LOCK.release()


def _store_alignment(connection, track_id, result: dict[str, object]) -> None:
    """Store an aligned song as its current karaoke variant."""
    # Aligned from the audio alone; the stored path is never bound to source lines.
    karaoke_provenance = {
        "alignment_mode": "global_ctc_audio_only",
        "audio_input": result["diagnostics"]["audio_source"],
        "separator_revision": result["diagnostics"]["separator_revision"],
        "model": result["model"],
        "pipeline_revision": KARAOKE_PIPELINE_REVISION,
        "inference_passes": result["diagnostics"].get("inference_passes"),
        "korean_labels": "pronunciation_v1",
        "syllable_duration_source": "predicted_token_end",
        "bound_to_synced_lines": False,
    }
    lines = mark_backing_voice(result["lines"])
    revision = _stored_model_revision(str(result["model_revision"]))
    store.save_aligned_karaoke(
        connection,
        track_id,
        lines,
        revision,
        karaoke_provenance,
        result["alignment_document"],
        result["diagnostics"],
    )
