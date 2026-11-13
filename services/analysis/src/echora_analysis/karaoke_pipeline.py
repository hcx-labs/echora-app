from __future__ import annotations

from .settings import get_settings

from collections.abc import Callable
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid

import psycopg
from psycopg.types.json import Jsonb

from .preprocessing import get_check, prepare_audio, vocal_audio_bytes
from .remote_compute import current as current_remote
from .roformer import separation_phase
from .roformer import SEPARATION_REVISION
from .navidrome import NavidromeClient
from .backing_vocals import BACKING_VOICE, tag_bracketed_backing
from .furigana import ruby_segments
from .processing_plan import plan_karaoke, resolve_library_id

logger = logging.getLogger(__name__)

FA_KARA_REVISION = "168ca5f01cecaa1290e31c0ce8dc44af8c7451bb"
# v5: the Echora Kara model on the exact frame clock, with blank-free syllables, backing lines on
# their own head, the in-line gap cost and the text head's bonus (v4: audio-only alignment with
# minimum syllable durations and a +45 ms calibration).
# v8: fast lead lines are no longer re-placed as backing lines (they overlapped their neighbours),
# and backing lines before the first lead line are placed on the lead path.
KARAOKE_PIPELINE_REVISION = "v8-echora-kara-mix"
DEFAULT_MODEL_ID = "hcxlabs/echora-kara-300m"
DEFAULT_MODEL_REVISION = "64087e1078b1aaf39cc1e6a9fc93874ca2f60e22"
_DIALOGUE = re.compile(r"^Dialogue: [^,]*,([^,]+),([^,]+),(?:[^,]*,){6}(.*)$")
_KARAOKE_TAG = re.compile(r"\{\\k(\d+)\}")
_ASS_TAG = re.compile(r"\{[^}]*\}")
_KARAOKE_LOCK = threading.Lock()
_FA_KARA_WORKER: subprocess.Popen[str] | None = None
_FA_KARA_WORKER_KEY: tuple[str, ...] | None = None


@functools.lru_cache(maxsize=4)
def _checkpoint_digest(weights: str, size: int, modified_ns: int) -> str:
    """Content digest of a local checkpoint's weights; cached per file version."""
    digest = hashlib.sha256()
    with open(weights, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def current_model_revision() -> str:
    """The aligner identity that planning, alignment and storage share.

    The pinned revision, or for FA_KARA_MODEL_PATH the checkpoint's own content,
    so replacing a local checkpoint makes existing alignments stale even when
    FA_KARA_REVISION is unchanged.
    """
    path = get_settings().fa_kara_model_path
    if not path:
        return get_settings().fa_kara_revision
    candidates = [Path(path) / name for name in ("model.safetensors", "pytorch_model.bin")]
    weights = next((candidate for candidate in candidates if candidate.is_file()), None)
    if weights is None:
        raise RuntimeError(f"FA-Kara checkpoint has no weights file: {path}")
    stat = weights.stat()
    return f"local-{_checkpoint_digest(str(weights), stat.st_size, stat.st_mtime_ns)}"


def _separates_vocals() -> bool:
    return get_settings().fa_kara_audio_input == "vocals"


def _stored_model_revision(model_revision: str) -> str:
    # The audio input is part of the identity, so switching it makes stored alignments stale.
    if _separates_vocals():
        return f"{FA_KARA_REVISION}:{model_revision}:roformer:{SEPARATION_REVISION}"
    return f"{FA_KARA_REVISION}:{model_revision}:mix"


def _ass_time_ms(value: str) -> int:
    hours, minutes, seconds = value.split(":")
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def parse_ass_karaoke(ass: str) -> list[dict[str, object]]:
    """Convert FA-Kara ASS dialogue into line and syllable timing for the API."""
    lines: list[dict[str, object]] = []
    for raw_line in ass.splitlines():
        match = _DIALOGUE.match(raw_line)
        if not match:
            continue
        start_ms = _ass_time_ms(match.group(1))
        end_ms = _ass_time_ms(match.group(2))
        body = match.group(3)
        cursor_ms = start_ms
        syllables: list[dict[str, object]] = []
        parts = _KARAOKE_TAG.split(body)
        prefix = _ASS_TAG.sub("", parts[0])
        text_parts = [prefix] if prefix else []
        for index in range(1, len(parts), 2):
            duration_ms = int(parts[index]) * 10
            text = _ASS_TAG.sub("", parts[index + 1] if index + 1 < len(parts) else "")
            if text:
                syllables.append(
                    {"start_ms": cursor_ms, "end_ms": cursor_ms + duration_ms, "text": text}
                )
                text_parts.append(text)
            cursor_ms += duration_ms
        text = "".join(text_parts).replace("#|", "").replace("|<", "")
        lines.append({"start_ms": start_ms, "end_ms": end_ms, "text": text, "syllables": syllables})
    return lines


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
    """Build karaoke lines from the alignment document's millisecond-precision timing.

    The alignment document stores token onset and offset in integer milliseconds
    derived directly from frame indices, avoiding the centisecond quantization
    inherent in the ASS ``\\k`` tag format (10 ms resolution).  Using it as the
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
    """Tag backing vocals: ``"voice": 2`` on syllables (players show them as a second, smaller
    line) and ``"voice": "background"`` on a line that is backing as a whole.

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
            inline = bool(others) and get_settings().fa_kara_inline_backing
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


def _fa_kara_worker(vendor: Path, model_revision: str) -> subprocess.Popen[str]:
    global _FA_KARA_WORKER, _FA_KARA_WORKER_KEY
    # The vendored decoder reads these two from its environment (the text head is loaded only
    # when the bonus is on), so they belong to the worker's identity.
    decoder_environment = {
        "FA_KARA_TEXT_BONUS": str(get_settings().fa_kara_text_bonus),
        "FA_KARA_IN_LINE_GAP_COST": str(get_settings().fa_kara_in_line_gap_cost),
    }
    key = (str(vendor), model_revision, *decoder_environment.values())
    if (
        _FA_KARA_WORKER is not None
        and _FA_KARA_WORKER.poll() is None
        and _FA_KARA_WORKER_KEY == key
    ):
        return _FA_KARA_WORKER
    if _FA_KARA_WORKER is not None and _FA_KARA_WORKER.poll() is None:
        _FA_KARA_WORKER.terminate()
    numba_cache = Path("/tmp/echora-fa-kara-worker-numba")
    numba_cache.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "NUMBA_CACHE_DIR": str(numba_cache),
        "PYTHONPATH": f"{vendor}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        **decoder_environment,
    }
    _FA_KARA_WORKER = subprocess.Popen(
        [sys.executable, str(vendor / "worker.py")],
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
        worker.wait(timeout=get_settings().karaoke_shutdown_grace_seconds)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait(timeout=get_settings().karaoke_kill_grace_seconds)


def _run_worker_job(
    worker: subprocess.Popen[str], argv: list[str], timeout: float | None = None
) -> dict[str, object]:
    if worker.stdin is None or worker.stdout is None:
        raise RuntimeError("FA-Kara worker pipes are unavailable")
    worker.stdin.write(json.dumps({"argv": argv}, ensure_ascii=False) + "\n")
    worker.stdin.flush()
    selector = selectors.DefaultSelector()
    selector.register(worker.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + (
        get_settings().karaoke_job_timeout_seconds if timeout is None else timeout
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
        worker.wait(timeout=get_settings().karaoke_kill_grace_seconds)
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
# data/finetune-curation/echora_normalization.py mirrors both sets (test_alignment_fixes checks).
FA_KARA_LANGUAGES_V1 = frozenset({"auto", "ja", "jaen", "zhen", "ko", "ur", "hi", "pa", "indic"})
FA_KARA_LANGUAGES_V2 = FA_KARA_LANGUAGES_V1 | frozenset({"en", "de", "fr", "tr", "yue", "zh"})

# Keep in sync with label_rules.NON_LATIN_SCRIPT_LANGUAGES (vendor runs in a subprocess).
FA_KARA_LANGUAGES_V3 = FA_KARA_LANGUAGES_V2 | frozenset(
    {
        "ko",
        "ja",
        "zh",
        "yue",
        "th",
        "ar",
        "fa",
        "ur",
        "he",
        "hi",
        "bn",
        "ta",
        "te",
        "mr",
        "gu",
        "kn",
        "ml",
        "pa",
        "ru",
        "uk",
        "bg",
        "sr",
        "mk",
        "be",
        "kk",
        "el",
        "ka",
        "hy",
        "km",
        "lo",
        "my",
    }
)


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
    separate_vocals: bool | None = None,
) -> dict[str, object]:
    vendor = Path(__file__).resolve().parents[2] / "vendor" / "fa_kara"
    model_id = get_settings().fa_kara_model_id
    model_revision = current_model_revision()
    snapshot = (
        Path(os.environ.get("HF_HOME", "/models/huggingface"))
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / get_settings().fa_kara_revision
    )
    if get_settings().fa_kara_model_path:
        # A local checkpoint; current_model_revision identifies it by its weights.
        snapshot = Path(get_settings().fa_kara_model_path)
        model_id = str(snapshot)
    if not snapshot.is_dir():
        raise RuntimeError(f"FA-Kara model snapshot is missing: {model_id}@{model_revision}")
    with tempfile.TemporaryDirectory(prefix="echora-fa-kara-") as directory:
        work = Path(directory)
        audio_path = work / "i.audio"
        # None follows FA_KARA_AUDIO_INPUT; a caller that brings its own audio passes False.
        original_mix = separate_vocals is None and not _separates_vocals()
        separate_vocals = _separates_vocals() if separate_vocals is None else separate_vocals
        if separate_vocals:
            # Keep the aligner resident on cache hits. If eviction caused a miss,
            # release it under the cache's producer lock before separator loading.
            prepared = vocal_audio_bytes(
                audio, check=get_check(), before_separate=_stop_fa_kara_worker
            )
            audio_path.write_bytes(prepared)
        else:
            # The song as stored, or timing repair's own audio, which must not be separated again.
            audio_path.write_bytes(audio)
        line_texts = _source_line_texts(source_lines or [])
        input_text = "\n".join(line_texts) if line_texts else lyrics_text.rstrip("\r\n")
        (work / "i.txt").write_text(input_text + "\n", encoding="utf-8")
        aligner = get_settings().fa_kara_aligner.lower()
        if aligner not in {"yohane", "mms"}:
            raise ValueError("FA_KARA_ALIGNER must be 'yohane' or 'mms'")
        command = [
            "--path_io",
            str(work),
            "--input_audio",
            str(audio_path),
            "--input_text",
            "i.txt",
            "--model",
            aligner,
            "--lang",
            language
            if language in fa_kara_languages(get_settings().fa_kara_label_rules)
            else "auto",
        ]
        if not language or language in UNKNOWN_LANGUAGES:
            # Lyrics stored without a language (ingest-time detection failed or was unsure): the
            # aligner identifies them itself rather than spell every word letter by letter.
            command.extend(
                [
                    "--detect_language",
                    ",".join(sorted(fa_kara_languages(get_settings().fa_kara_label_rules))),
                ]
            )
        if get_settings().fa_kara_label_rules != "v1":
            command.extend(["--label_rules", get_settings().fa_kara_label_rules])
        if aligner == "yohane":
            command.extend(["--hf_model_path", str(snapshot)])
        command.extend(["--head_correct", "0", "--tail_correct", "0"])
        command.extend(["--min_letter_ms", str(get_settings().fa_kara_min_letter_ms)])
        command.extend(["--min_token_ms", str(get_settings().fa_kara_min_token_ms)])
        command.extend(["--timing_offset_ms", str(get_settings().fa_kara_timing_offset_ms)])
        command.extend(["--duration_pace_ratio", str(get_settings().fa_kara_duration_pace_ratio)])
        command.extend(
            ["--vocal_activity_weight", str(get_settings().fa_kara_vocal_activity_weight)]
        )
        if get_settings().fa_kara_exact_frame_clock:
            command.append("--exact_frame_clock")
        if get_settings().fa_kara_background_decoding:
            command.append("--background_decoding")
        if get_settings().fa_kara_blank_free_tokens:
            command.append("--blank_free_tokens")
        if get_settings().fa_kara_inline_backing:
            command.append("--inline_backing")
        if get_settings().fa_kara_gap_onset_pull:
            command.append("--gap_onset_pull")
        if os.environ.get("FA_KARA_OPTIONAL_BRACKETS") == "1":
            # Experiment, off: inline bracketed words may leave the lead path. It did not help.
            command.append("--optional_brackets")
        external_activity = os.environ.get("FA_KARA_VOCAL_ACTIVITY_FILE")
        if external_activity:
            # Listening tests: activity from another model on the same mix (see listen_align_mix.py).
            command.extend(
                ["--vocal_activity_source", "external", "--vocal_activity_file", external_activity]
            )
        if get_settings().fa_kara_fusion != "off":
            command.extend(["--fusion", get_settings().fa_kara_fusion])
        audio_speed = float(get_settings().fa_kara_audio_speed)
        if not 0.5 <= audio_speed <= 1.5:
            raise ValueError("FA_KARA_AUDIO_SPEED must be between 0.5 and 1.5")
        if audio_speed != 1.0:
            command.extend(["--audio_speedx", str(audio_speed)])
        worker = _fa_kara_worker(vendor, model_revision)
        response = _run_worker_job(worker, command)
        if not response.get("ok"):
            detail = str(
                response.get("traceback") or response.get("error") or "unknown worker error"
            )
            raise RuntimeError(f"FA-Kara failed: {detail[-4000:]}")
        ass = (work / "o.ass").read_text(encoding="utf-8")
        lrc = (work / "o_ruby.lrc").read_text(encoding="utf-8")
        alignment_document = _validate_alignment_document(
            json.loads((work / "o.alignment.json").read_text(encoding="utf-8"))
        )
        diagnostics = alignment_document.setdefault("diagnostics", {})
        diagnostics["audio_source"] = (
            "roformer_vocals"
            if separate_vocals
            else "original_mix"
            if original_mix
            else "provided_audio"
        )
        diagnostics["separator"] = "roformer" if separate_vocals else None
        diagnostics["separator_revision"] = SEPARATION_REVISION if separate_vocals else None
        lines = build_lines_from_alignment_document(alignment_document)
        if not lines:
            lines = parse_ass_karaoke(ass)
        # Keep the model's token-derived line boundaries. The former display
        # padding expanded every line into nearby silence, which moved otherwise
        # correct starts and ends away from the timed-lyrics reference.
        if not lines:
            raise RuntimeError("FA-Kara produced no aligned lyric lines")
    return {
        "ass": ass,
        "lrc": lrc,
        "lines": lines,
        "alignment_document": alignment_document,
        "diagnostics": alignment_document.get("diagnostics", {}),
        "model": model_id,
        "model_revision": model_revision,
    }


def backfill_karaoke(
    url: str,
    username: str,
    password: str,
    progress: Callable[[dict[str, object]], None] | None = None,
    external_ids: list[str] | None = None,
) -> dict[str, int]:
    """Serialize alignment and release its resident model after the phase."""
    while not _KARAOKE_LOCK.acquire(timeout=get_settings().karaoke_lock_poll_seconds):
        get_check()()
    try:
        get_check()()
        return _backfill_karaoke(url, username, password, progress, external_ids)
    finally:
        try:
            _stop_fa_kara_worker()
        finally:
            _KARAOKE_LOCK.release()


def _store_supplied_karaoke(connection, track_id, lines, digest) -> None:
    """Word timing that came with the lyrics (TTML or enhanced LRC from Navidrome), stored as the
    karaoke without alignment. Not AI-generated, so players show no AI disclosure for it."""
    provenance = {
        "alignment_mode": "supplied",
        "source": "navidrome",
        "source_digest": digest,
        "pipeline_revision": KARAOKE_PIPELINE_REVISION,
        "ai_generated": False,
    }
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO karaoke_lyrics_variants
                 (track_id, bounded, lines, ass, lrc, model, model_revision, provenance,
                  alignment_document, diagnostics, created_at)
               VALUES (%s,false,%s,NULL,NULL,'navidrome','supplied',%s,NULL,'{}',now())
               ON CONFLICT (track_id, bounded) DO UPDATE SET lines=EXCLUDED.lines,
                 ass=NULL, lrc=NULL, model=EXCLUDED.model, model_revision=EXCLUDED.model_revision,
                 provenance=EXCLUDED.provenance, alignment_document=NULL, diagnostics='{}',
                 created_at=now()""",
            (track_id, Jsonb(lines), Jsonb(provenance)),
        )
        cursor.execute(
            """UPDATE lyrics SET karaoke_lines=%s, karaoke_ass=NULL, karaoke_lrc=NULL,
                      karaoke_model='navidrome', karaoke_model_revision='supplied',
                      karaoke_bounded=false, karaoke_created_at=now(),
                      provenance=provenance || %s
               WHERE track_id=%s""",
            (Jsonb(lines), Jsonb({"karaoke": provenance}), track_id),
        )
    connection.commit()


def _backfill_karaoke(
    url: str,
    username: str,
    password: str,
    progress: Callable[[dict[str, object]], None] | None = None,
    external_ids: list[str] | None = None,
) -> dict[str, int]:
    """Align every planned song's lyrics from its audio."""
    report = progress or (lambda _: None)
    summary = {"total": 0, "aligned": 0, "failed": 0}
    with (
        psycopg.connect(get_settings().database_url) as connection,
        NavidromeClient(url, username, password) as client,
    ):
        library_id = resolve_library_id(connection, url)
        model_revision = _stored_model_revision(current_model_revision())
        planned = plan_karaoke(
            connection,
            KARAOKE_PIPELINE_REVISION,
            external_ids,
            model_revision,
            library_id=library_id,
        ).karaoke_external_ids
        if not planned:
            report(
                {
                    "phase": "planning",
                    "message": "Karaoke alignment already current",
                    "completed": 0,
                    "total": 0,
                    "unit": "tracks",
                }
            )
            return summary
        with connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT DISTINCT ON (ts.track_id) ts.track_id, ts.external_id, t.title, l.text, l.language, l.provenance->'lines',
                           l.provenance->'word_timing', l.provenance->>'word_timing_digest'
                    FROM track_sources ts JOIN tracks t ON t.id=ts.track_id JOIN lyrics l ON l.track_id=ts.track_id
                    WHERE ts.source_type='subsonic' AND ts.external_id=ANY(%s) AND ts.library_id=%s
                    ORDER BY ts.track_id, ts.id""",
                (list(planned), library_id),
            )
            rows = cursor.fetchall()
        summary["total"] = len(rows)
        tracks = []
        for row in rows:
            track_id, external_id, title, text, language, lines = row[:6]
            supplied, digest = (row[6], row[7]) if len(row) > 7 else (None, None)
            if supplied and digest:
                try:
                    _store_supplied_karaoke(connection, track_id, supplied, digest)
                    summary["supplied"] = summary.get("supplied", 0) + 1
                except Exception:
                    connection.rollback()
                    summary["failed"] += 1
                    logger.exception("Storing supplied word timing failed for %s", track_id)
            else:
                tracks.append((track_id, external_id, title, text, language, lines))
        if not tracks:
            return summary
        _stop_fa_kara_worker()
        remote = current_remote()
        prepared_sources = set()
        digests: dict[object, str] = {}
        separate = _separates_vocals()
        # Separation phase: one Roformer for every selected song, before the aligner loads.
        # On the original mix this only fetches each song into the batch's source cache.
        with separation_phase():
            for index, (track_id, external_id, title, *_) in enumerate(tracks):
                report(
                    {
                        "phase": "preprocess",
                        "message": f"Preparing karaoke audio for {title}",
                        "completed": index,
                        "total": len(tracks),
                        "unit": "tracks",
                    }
                )
                get_check()()
                try:
                    source = client.audio_bytes(external_id)
                    if remote is not None:
                        # Vocals are separated on Modal; only the source audio goes up.
                        digests[track_id] = remote.upload_audio(source)
                    elif separate:
                        prepare_audio(source, vocals=True, check=get_check())
                    prepared_sources.add(track_id)
                    del source
                except Exception:
                    summary["failed"] += 1
                    logger.exception("Karaoke preprocessing failed for %s", track_id)
                report(
                    {
                        "phase": "preprocess",
                        "message": f"Preparing karaoke audio for {title}",
                        "completed": index + 1,
                        "total": len(tracks),
                        "unit": "tracks",
                        "summary": summary,
                    }
                )
        if remote is not None and separate and prepared_sources:
            # Like local preparation: separate the vocals of exactly the songs selected for
            # alignment first, so the aligner then stays loaded on Modal instead of being
            # stopped and reloaded around each song's separation.
            separating = [row for row in tracks if row[0] in prepared_sources]
            outcomes = remote.prepare_vocals([digests[row[0]] for row in separating], vocals=True)
            for index, (row, outcome) in enumerate(zip(separating, outcomes)):
                if isinstance(outcome, BaseException):
                    summary["failed"] += 1
                    prepared_sources.discard(row[0])
                    logger.error("Vocal separation on Modal failed for %s", row[0])
                report(
                    {
                        "phase": "preprocess",
                        "message": f"Separating vocals for {row[2]}",
                        "completed": index + 1,
                        "total": len(separating),
                        "unit": "tracks",
                        "summary": summary,
                    }
                )
        if prepared_sources:
            report(
                {
                    "phase": "models",
                    "message": "Loading Echora alignment model",
                    "completed": 0,
                    "total": 1,
                    "unit": "models",
                }
            )
        aligned = None
        if remote is not None:
            # Every prepared track aligns in one Modal call; results stream back in order.
            batch = [row for row in tracks if row[0] in prepared_sources]
            aligned = iter(
                remote.karaoke(
                    [digests[row[0]] for row in batch],
                    [row[3] for row in batch],
                    [row[4] for row in batch],
                    [row[5] or [] for row in batch],
                )
            )
        for index, (track_id, external_id, title, text, language, source_lines) in enumerate(
            tracks
        ):
            if track_id not in prepared_sources:
                continue
            try:
                if aligned is None:
                    result = _run_fa_kara(
                        client.audio_bytes(external_id), text, language, source_lines or []
                    )
                else:
                    result = next(aligned)
                    if isinstance(result, BaseException):
                        raise result
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
                result["lines"] = mark_backing_voice(result["lines"])
                with connection.cursor() as cursor:
                    cursor.execute(
                        """INSERT INTO karaoke_lyrics_variants
                             (track_id, bounded, lines, ass, lrc, model, model_revision, provenance,
                              alignment_document, diagnostics, created_at)
                           VALUES (%s,%s,%s,%s,%s,'fa_kara',%s,%s,%s,%s,now())
                           ON CONFLICT (track_id, bounded) DO UPDATE SET lines=EXCLUDED.lines,
                             ass=EXCLUDED.ass, lrc=EXCLUDED.lrc, model=EXCLUDED.model,
                             model_revision=EXCLUDED.model_revision, provenance=EXCLUDED.provenance,
                             alignment_document=EXCLUDED.alignment_document,
                             diagnostics=EXCLUDED.diagnostics, created_at=now()""",
                        (
                            track_id,
                            False,
                            Jsonb(result["lines"]),
                            result["ass"],
                            result["lrc"],
                            _stored_model_revision(str(result["model_revision"])),
                            Jsonb(karaoke_provenance),
                            Jsonb(result["alignment_document"]),
                            Jsonb(result["diagnostics"]),
                        ),
                    )
                    cursor.execute(
                        """UPDATE lyrics SET karaoke_lines=%s, karaoke_ass=%s, karaoke_lrc=%s,
                                  karaoke_model='fa_kara', karaoke_model_revision=%s, karaoke_bounded=%s,
                                  karaoke_created_at=now(),
                                  provenance=provenance || %s
                           WHERE track_id=%s""",
                        (
                            Jsonb(result["lines"]),
                            result["ass"],
                            result["lrc"],
                            _stored_model_revision(str(result["model_revision"])),
                            False,
                            Jsonb({"karaoke": karaoke_provenance}),
                            track_id,
                        ),
                    )
                connection.commit()
                summary["aligned"] += 1
            except Exception:
                connection.rollback()
                summary["failed"] += 1
                logger.exception("FA-Kara alignment failed for %s", track_id)
            report(
                {
                    "phase": "karaoke",
                    "message": f"Aligning karaoke lyrics for {title}",
                    "completed": index + 1,
                    "total": len(tracks),
                    "unit": "tracks",
                    "summary": summary,
                }
            )
    return summary
