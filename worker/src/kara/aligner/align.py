"""Align lyrics to a song: normalise the lines into acoustic tokens, decode, build the document.

One call aligns one song. ``align`` returns the alignment document the karaoke pipeline stores
(``schema_version`` 1): every source line with its language and timed tokens, and diagnostics.
The worker process (``worker.py``) keeps the decoder's models resident between calls.
"""

from __future__ import annotations

import hashlib
import io
import re
import subprocess
import time
import unicodedata
from pathlib import Path

import soundfile as sf

from worker.kara.aligner import korean as korean_pronunciation, label_rules, normalize as hn
from worker.kara.aligner.request import AlignRequest

# Japanese syllable splitting of the normaliser, as every shipped model was trained with.
SOKUON_SPLIT = 0
HATSUON_SPLIT = 1


def file_identity(path):
    import hashlib
    from pathlib import Path

    path = Path(path)
    # Only inference artifacts: never traverse optimizer/trainer checkpoints.
    patterns = (
        "*.safetensors",
        "pytorch_model*.bin",
        "*config.json",
        "*token*.json",
        "vocab.json",
        "*.model",
        "*.index.json",
        "special_tokens_map.json",
        "added_tokens.json",
    )
    files = (
        sorted({p for pattern in patterns for p in path.glob(pattern) if p.is_file()})
        if path.is_dir()
        else [path]
    )
    hashes = {}
    for file in files:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(file.relative_to(path)) if path.is_dir() else file.name] = digest.hexdigest()
    return {"path": str(path.resolve()), "sha256": hashes}


def normalize_source_lines(
    lines, language, sokuon_split, hatsuon_split, label_rules="v1", line_languages=None
):
    """Split source boundaries before normalization; never normalize a newline.

    ``label_rules`` selects the shared normalizer's rule set (label_rules.RULES); "v1" is the
    behaviour of every shipped model, "v2" must only be used with a model trained on v2 labels.
    ``line_languages``, when a list, receives the language each source line was normalised with.
    """
    records, source_texts = [], []
    for index, line in enumerate(lines):
        text = line.rstrip("\r\n")
        source_texts.append(text)
        if line_languages is not None:
            line_languages.append(hn.line_language(text, language, label_rules))
        # Punctuation-only separators are display lines, not acoustic tokens.
        # Keep their source index/text and boundary without inventing speech.
        display_only = all(c.isspace() or unicodedata.category(c).startswith("P") for c in text)
        items = (
            hn.process_haruhi_line(text, language, sokuon_split, hatsuon_split, rules=label_rules)
            if not display_only
            else []
        )
        # Hangul is labelled by pronunciation (같이 → ga+chi), as in training.
        items = korean_pronunciation.apply_to_items(items, v2=label_rules != "v1")
        if not display_only and not any(item.get("pron") for item in items):
            raise ValueError(f"Unsupported display line at source index {index}: {text!r}")
        records.extend(items)
        records.append({"orig": "\n", "type": 0, "pron": ""})
    return records, source_texts


def source_diagnostic_indexes(value, indexes, key=""):
    """Translate compressed acoustic-line diagnostics into source coordinates."""
    if isinstance(value, dict):
        return {k: source_diagnostic_indexes(v, indexes, k) for k, v in value.items()}
    if isinstance(value, list):
        if key == "lines" or key.endswith("_lines") and key != "interpolated_source_lines":
            return [indexes[v] for v in value]
        if key == "vocal_focus_attempted_runs":
            return [[indexes[v] for v in run] for run in value]
        return [source_diagnostic_indexes(v, indexes) for v in value]
    if key == "line" and isinstance(value, int):
        return indexes[value]
    return value


def _read_audio(path: str):
    try:
        return sf.read(path, dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        decoded = str(Path(path).parent / "decoded.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                path,
                "-vn",
                "-acodec",
                "pcm_f32le",
                decoded,
            ],
            check=True,
        )
        return sf.read(decoded, dtype="float32", always_2d=True)


def _time_stretch(audio, speed: float):
    if speed == 1:
        return audio
    import librosa

    return librosa.effects.time_stretch(audio, rate=speed)


def align(request: AlignRequest) -> dict:
    """The alignment document of one song."""
    start_time = time.time()
    hf_model_path = request.model_path
    language = request.language.lower()
    # Lines as a text file reads them: universal newlines, each line with its newline.
    lines = list(io.StringIO(request.text + "\n", newline=None))
    language_source = "given"
    if request.detect_language and language == "auto":
        language_source = "undetected"
        try:
            detected = label_rules.document_language(
                [str(line) for line in lines], frozenset(request.detect_language) - {"auto"}
            )
        except RuntimeError as error:  # no lid.176.ftz / fasttext: keep the generic path
            print(f"Language detection unavailable: {error}")
            detected = None
        if detected:
            language, language_source = detected, "detected"
            print(f"Lyrics stored without a language; identified as {detected}")
    line_languages: list[str] = []
    result_list, source_texts = normalize_source_lines(
        lines, language, SOKUON_SPLIT, HATSUON_SPLIT, request.label_rules, line_languages
    )
    if not any(item.get("pron") for item in result_list):
        raise ValueError("Lyrics contain no alignable text")

    alignment_tokens = []
    alignment_token_lines = [[]]
    token_to_index_map = {}
    bracketed = []  # per acoustic token: inside a bracket of its line
    depth = 0
    for i, item in enumerate(result_list):
        if item.get("orig") == "\n":
            alignment_token_lines.append([])
            depth = 0
        elif "pron" in item and item["pron"]:
            alignment_tokens.append(item["pron"])
            alignment_token_lines[-1].append(item["pron"])
            token_to_index_map[len(alignment_tokens) - 1] = i
            bracketed.append(depth > 0)
        else:
            for char in str(item.get("orig") or ""):
                depth = (
                    depth + 1 if char in "(（" else max(0, depth - 1) if char in ")）" else depth
                )
    if alignment_token_lines and not alignment_token_lines[-1]:
        alignment_token_lines.pop()
    for item in alignment_tokens:
        if not hn.is_english(item):
            print(f"Unexpected acoustic token {item!r}")
    print("Lyrics text analysis executed in", round(time.time() - start_time, 3), "seconds")

    audio_channels, sr = _read_audio(request.audio_path)
    y_processed = _time_stretch(audio_channels.mean(axis=1), request.audio_speed)
    print("Aligning...")
    from worker.kara.aligner.decoder import MAX_BLANK_HOLD_SECONDS, align_audio_with_timeline

    opening_backing_lines = set()
    background_lines = None
    if request.background_decoding:
        # Lines that are one bracketed span are background vocals, decoded on the background head
        # ("(Three) stop your staring at my- (hey!)" is a lead line with inline brackets).
        background_lines = [
            bool(re.fullmatch(r"\s*[(（][^()（）]*[)）]\s*", str(source_texts[i])))
            for i, tokens in enumerate(alignment_token_lines)
            if tokens
        ]
        # Backing lines written before the first lead line are sung alone, so the lead head
        # hears them and the background head does not ("Ah-ah" over an intro): placed on the
        # background head they were squeezed against the first lead line, whose first word took
        # the intro instead. Keep them on the lead path, in order; they stay backing lines for
        # display.
        if not all(background_lines):
            with_tokens = [i for i, tokens in enumerate(alignment_token_lines) if tokens]
            for position, flag in enumerate(background_lines):
                if not flag:
                    break
                background_lines[position] = False
                opening_backing_lines.add(with_tokens[position])

    def inline_flags():
        # Inline brackets ("I know ('til the day that I die)"); a line that is one bracketed span
        # stays whole, and "(oh) (yeah)" keeps a lead word.
        flags_out, cursor = [], 0
        for tokens in alignment_token_lines:
            flags = bracketed[cursor : cursor + len(tokens)]
            flags_out.extend(flag and not all(flags) for flag in flags)
            cursor += len(tokens)
        return flags_out

    alignment_results = align_audio_with_timeline(
        y_processed,
        [tokens for tokens in alignment_token_lines if tokens],
        sr=sr,
        speed=request.audio_speed,
        use_gpu=True,
        hf_model_id=hf_model_path,
        min_letter_ms=request.min_letter_ms,
        min_token_ms=request.min_token_ms,
        timing_offset_ms=request.timing_offset_ms,
        duration_pace_ratio=request.duration_pace_ratio,
        vocal_activity_weight=request.vocal_activity_weight,
        exact_frame_clock=request.exact_frame_clock,
        vocal_activity_source=request.vocal_activity_source,
        vocal_activity_file=request.vocal_activity_file,
        blank_free_tokens=request.blank_free_tokens,
        background_lines=background_lines,
        optional_tokens=inline_flags() if request.optional_brackets else None,
        inline_backing_tokens=inline_flags() if request.inline_backing else None,
        gap_onset_pull=request.gap_onset_pull,
        text_bonus=request.text_bonus,
        in_line_gap_cost=request.in_line_gap_cost,
        fusion=request.fusion,
    )
    if len(alignment_results) != len(alignment_tokens):
        raise RuntimeError(
            f"Alignment returned {len(alignment_results)} tokens for {len(alignment_tokens)} inputs"
        )
    for i, result in enumerate(alignment_results):
        original_index = token_to_index_map[i]
        result_list[original_index]["alignment_score"] = float(result.get("score", 0.0))
        result_list[original_index]["alignment_start"] = float(result.get("original_start", 0.0))
        result_list[original_index]["alignment_end"] = float(result.get("original_end", 0.0))

    canonical_lines = []
    token_offset = 0
    for line_index, line_tokens in enumerate(alignment_token_lines):
        token_records = []
        for token_index, _ in enumerate(line_tokens):
            alignment_index = token_offset + token_index
            original_item = result_list[token_to_index_map[alignment_index]]
            token_records.append(
                {
                    "source_index": token_index,
                    "text": str(original_item.get("orig", "")),
                    "acoustic_token": str(original_item.get("pron", "")),
                    "start_ms": round(float(original_item["alignment_start"]) * 1000),
                    "end_ms": round(float(original_item["alignment_end"]) * 1000),
                    "acoustic_start_ms": round(
                        float(
                            alignment_results[alignment_index].get(
                                "acoustic_start", original_item["alignment_start"]
                            )
                        )
                        * 1000
                    ),
                    "acoustic_end_ms": round(
                        float(
                            alignment_results[alignment_index].get(
                                "acoustic_end", original_item["alignment_end"]
                            )
                        )
                        * 1000
                    ),
                    "ctc_score": float(original_item["alignment_score"]),
                    **(
                        {"stream": 2}
                        if alignment_results[alignment_index].get("stream") == 2
                        else {}
                    ),
                }
            )
        line_is_background = line_index in opening_backing_lines or any(
            alignment_results[token_offset + i].get("voice") == "background"
            for i in range(len(line_tokens))
        )
        token_offset += len(line_tokens)
        canonical_lines.append(
            {
                "source_index": line_index,
                "language": line_languages[line_index],
                "text": re.sub(
                    r"\{([^{}|]*)\|[^{}]*\}|\[([^\[\]|]*)\|[^\[\]]*\]",
                    lambda match: match.group(1) if match.group(1) is not None else match.group(2),
                    str(source_texts[line_index]),
                ),
                # Second-stream words (optional brackets) may end after the line's last lead word.
                "start_ms": min(t["start_ms"] for t in token_records) if token_records else None,
                "end_ms": max(t["end_ms"] for t in token_records) if token_records else None,
                "tokens": token_records,
                **({"voice": "background"} if line_is_background else {}),
            }
        )
    finite_scores = [token["ctc_score"] for line in canonical_lines for token in line["tokens"]]
    source_diagnostics = (
        alignment_results[0].get("source_diagnostics", {}) if alignment_results else {}
    )
    source_diagnostics = source_diagnostic_indexes(
        source_diagnostics, [i for i, tokens in enumerate(alignment_token_lines) if tokens]
    )
    return {
        "schema_version": 1,
        "alignment": {"mode": "global_ctc_audio_only", "lines": canonical_lines},
        "diagnostics": {
            "inference_passes": max(
                (int(result.get("inference_passes", 1)) for result in alignment_results), default=0
            ),
            "token_count": len(finite_scores),
            "mean_ctc_score": sum(finite_scores) / len(finite_scores) if finite_scores else 0.0,
            "minimum_ctc_score": min(finite_scores) if finite_scores else 0.0,
            **source_diagnostics,
            "line_count": len(canonical_lines),
            "acoustic_line_source_indexes": [
                i for i, tokens in enumerate(alignment_token_lines) if tokens
            ],
            "normalizer_identity": file_identity(hn.__file__),
            "label_rules": request.label_rules,
            "source_identity": {"sha256": hashlib.sha256(request.text.encode()).hexdigest()},
            "audio_identity": file_identity(request.audio_path),
            "checkpoint_identity": file_identity(hf_model_path)
            if hf_model_path and Path(hf_model_path).exists()
            else None,
            "max_blank_hold_seconds": MAX_BLANK_HOLD_SECONDS,
            "model_path": hf_model_path,
            "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
            "normalization_language": language,
            "normalization_language_source": language_source,
            # Every line decides its own language (label_rules.line_language); the song's is the default.
            "line_language_counts": {
                language: line_languages.count(language) for language in sorted(set(line_languages))
            },
            "opening_backing_lines_on_lead": len(opening_backing_lines),
            "audio_source": "input_audio",
            "inference_config": request.model_dump(),
        },
    }
