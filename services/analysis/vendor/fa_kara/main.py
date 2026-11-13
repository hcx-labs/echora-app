import argparse
import json
import os
import soundfile as sf
import re
import subprocess
import time
import unicodedata

# import ass2lrc
import haruraw2norm as hn
import korean_pronunciation
import lrcfmt
import norm2ass
from norm2lrc import process_main, process_ruby_V2, process_rlf
from utils_audio import non_silent_recog, time_stretch_audio, func_tail_correct_v250615
from utils_basic import (
    split_long_segments,
    non_silent_head_adjust,
    func_tail_correct_v250611,
)


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


def normalize_source_lines(lines, language, sokuon_split, hatsuon_split, label_rules="v1"):
    """Split source boundaries before normalization; never normalize a newline.

    ``label_rules`` selects the shared normalizer's rule set (label_rules.RULES); "v1" is the
    behaviour of every shipped model, "v2" must only be used with a model trained on v2 labels."""
    records, source_texts = [], []
    for index, line in enumerate(lines):
        text = line.rstrip("\r\n")
        source_texts.append(text)
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


def main(argv=None):
    start_time = time.time()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="可选参数")
    parser.add_argument(
        "-x", "--sokuon_split", type=int, default=0, help="是否将促音与前一字符拆开"
    )
    parser.add_argument(
        "-n", "--hatsuon_split", type=int, default=1, help="是否将拨音与前一字符拆开"
    )
    parser.add_argument("-v", "--audio_speedx", type=float, default=1, help="推理时使用的音频倍速")
    parser.add_argument(
        "-p",
        "--path_io",
        default="",
        help="输入输出文件目录。基于主文件所在目录，支持绝对路径或相对路径",
    )
    parser.add_argument("-ia", "--input_audio", default=None, help="输入音频文件名")
    parser.add_argument("-it", "--input_text", default="i.txt", help="输入歌词文件名")
    parser.add_argument("-t", "--tail_correct", type=int, default=-999, help="尾音拖长选项")
    parser.add_argument(
        "-tl", "--tail_limit_window", type=float, default=0.8, help="全曲静音检测窗口时长，单位：秒"
    )
    parser.add_argument(
        "-tp",
        "--tail_thres_pct",
        type=float,
        default=10,
        help="尾音阈值百分位数，单位：％。以音频能量前“百分位数”的一定比例作为静音检测阈值",
    )
    parser.add_argument(
        "-tr",
        "--tail_thres_ratio",
        type=float,
        default=0.1,
        help="尾音阈值比例。以音频能量前百分位数的一定“比例”作为静音检测阈值",
    )
    parser.add_argument(
        "--offset", type=int, default=-150, help="输出ruby歌词文件中Offset标签的偏移值"
    )
    parser.add_argument("--bpm", type=float, default=60, help="歌曲的BPM，导唱指示灯用")
    parser.add_argument("--bpb", type=int, default=3, help="导唱指示灯的符号个数")
    parser.add_argument("--lang", default="auto", help="歌词语言")
    parser.add_argument(
        "--detect_language",
        default=None,
        help="With --lang auto: comma-separated languages the lyrics may be identified as "
        "(label_rules.document_language, lid.176.ftz). For songs stored without a language, "
        "which would otherwise be spelled letter by letter",
    )
    parser.add_argument(
        "--label_rules",
        choices=("v1", "v2", "v2-orthographic", "v3", "v3-orthographic", "v3-sung"),
        default="v1",
        help="Label rules of the shared normalizer (label_rules.py). v1: every shipped model. "
        "v2: the 2026-10-18 label round (pronunciation labels for en/de/fr, numbers, acronyms, "
        "vocables, Indic/Tamil/Arabic/Japanese fixes); only with a model trained on v2 labels. "
        "v3: v2 plus LABEL_RULES_V3.md (lyric lexicon, homographs, dropped g, spell-outs, per-line "
        "language via fastText lid.176.ftz); only with a model trained on v3 labels",
    )
    parser.add_argument("-f", "--txt_format", default="hrh", help="歌词文本格式")
    parser.add_argument(
        "-cl", "--characters_per_line", type=int, default=0, help="输出文件每行最大字数"
    )
    parser.add_argument(
        "--no-gpu", action="store_false", dest="use_gpu", default=True, help="禁用GPU加速"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str.lower,
        default="mms",
        choices=["mms", "yohane"],
        help="底层模型选择",
    )
    parser.add_argument("-hf", "--hf_model_path", default=None, help="HuggingFace模型ID或本地路径")
    parser.add_argument("--head_correct", type=int, default=-999, help="是否进行静音检测")
    parser.add_argument("--min_letter_ms", type=int, default=0, help="Minimum duration per letter")
    parser.add_argument(
        "--min_token_ms", type=int, default=0, help="Minimum duration per acoustic token"
    )
    parser.add_argument(
        "--timing_offset_ms", type=int, default=0, help="Shift every token time (calibration)"
    )
    parser.add_argument(
        "--vocal_activity_weight",
        type=float,
        default=0.0,
        help="Extra cost per letter frame where the vocal stem is quiet (0: off)",
    )
    parser.add_argument(
        "--vocal_activity_source",
        choices=("energy", "vad", "external"),
        default="energy",
        help="What --vocal_activity_weight penalises against: energy (loudness of the input, "
        "the shipped behaviour), vad (the checkpoint's auxiliary vocal-activity head, "
        "aux_heads.safetensors next to a w2v-BERT checkpoint) or external (--vocal_activity_file)",
    )
    parser.add_argument(
        "--background_decoding",
        action="store_true",
        help="B0: decode whole bracketed lines on the checkpoint's background head (may overlap the lead)",
    )
    parser.add_argument(
        "--inline_backing_anchor",
        type=int,
        default=3,
        help="an inline bracket run may start this many lead words before the run",
    )
    parser.add_argument(
        "--gap_onset_pull",
        action="store_true",
        help="after >=0.6 s of silence, pull a word that the lattice starts >=0.6 s late back to the "
        "first frame with vocal energy (a lone sustained vowel the model barely hears)",
    )
    parser.add_argument(
        "--inline_backing",
        action="store_true",
        help="inline bracketed words stay on the lead path for ordering, then are re-placed on the "
        "background head inside their line's window (2026-11-09)",
    )
    parser.add_argument(
        "--optional_brackets",
        action="store_true",
        help="inline bracketed words may leave the lead path; the ones left out go on a second stream",
    )
    parser.add_argument(
        "--blank_free_tokens",
        action="store_true",
        help="No blank between letters of one token, as in training (held notes stay on their word)",
    )
    parser.add_argument(
        "--vocal_activity_file",
        default=None,
        help="float32 .npy of per-20 ms singing activity (with --vocal_activity_source external)",
    )
    parser.add_argument(
        "--duration_pace_ratio",
        type=float,
        default=0.0,
        help="Scale the minimum durations to each line's pace (0: fixed floor)",
    )
    parser.add_argument(
        "--separate_vocals", action="store_true", help="Align a persistent Demucs vocal stem"
    )
    # Decoder experiments (V1_LARGE_PLAN E3, SYLLABLE_ALIGNMENT_RESEARCH S1/S2); all off by default.
    parser.add_argument(
        "--exact_frame_clock",
        action="store_true",
        help="E3: report each frame at its true receptive-field centre (no duration/frames stretch)",
    )
    parser.add_argument(
        "--subframe_refinement",
        action="store_true",
        help="S2: sub-frame token starts from the first letter's interpolated log-odds",
    )
    parser.add_argument(
        "--onset_offsets",
        default=None,
        help="S2: JSON object {onset class: ms} added to --timing_offset_ms per token "
        "(classes: ctc_segmentation.ONSET_CLASSES)",
    )
    parser.add_argument(
        "--edge_weight",
        type=float,
        default=0.0,
        help="S1: weight of the checkpoint's edge head in the Viterbi transitions (0: off)",
    )
    parser.add_argument(
        "--fusion",
        choices=("off", "on", "gate0"),
        default="off",
        help="§5f MuQ fusion checkpoints: on = load the frozen MuQ branch the checkpoint was "
        "trained with (needs the muq package and OpenMuQ/MuQ-large-msd-iter in HF_HOME); "
        "gate0 = run the checkpoint as the plain model (ablation); off (default) refuses a "
        "fusion checkpoint and changes nothing for the others",
    )
    parser.add_argument(
        "--fusion_model",
        default=None,
        help="Override the MuQ hub id / local snapshot recorded in the fusion checkpoint",
    )
    args = parser.parse_args(argv)
    sokuon_split = args.sokuon_split
    hatsuon_split = args.hatsuon_split
    audio_speed = args.audio_speedx
    user_path = args.path_io
    user_audio_path = args.input_audio
    user_text_path = args.input_text
    tail_correct = args.tail_correct
    head_correct = args.head_correct
    silent_window_s = args.tail_limit_window
    tail_thres_pct = args.tail_thres_pct
    tail_thres_ratio = args.tail_thres_ratio
    ruby_tag_offset = args.offset
    bpm = args.bpm
    beats_per_bar = args.bpb
    lrc_language = args.lang.lower()
    txt_format = args.txt_format.lower()
    output_characters_per_line = args.characters_per_line
    align_use_gpu = True if args.use_gpu else False
    hf_model_path = args.hf_model_path
    if hf_model_path is not None:
        fa_model_select = "yohane_hf"
    elif args.model == "yohane":
        fa_model_select = "yohane_hf"
        hf_model_path = "NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn"
    else:
        fa_model_select = "MMS_FA_torch"
    if fa_model_select == "MMS_FA_torch":
        if tail_correct == -999:
            tail_correct = 3
        if head_correct == -999:
            head_correct = 1
    elif fa_model_select == "yohane_hf":
        if tail_correct == -999:
            tail_correct = 0
        if head_correct == -999:
            head_correct = 0

    real_io_path = (
        os.path.normpath(user_path)
        if os.path.isabs(user_path)
        else os.path.normpath(os.path.join(script_dir, user_path))
    )
    if not os.path.exists(real_io_path):
        os.makedirs(real_io_path)
    input_text_path = os.path.normpath(os.path.join(real_io_path, user_text_path))
    if user_audio_path:
        input_audio_path = os.path.normpath(os.path.join(real_io_path, user_audio_path))
    elif os.path.exists(os.path.normpath(os.path.join(real_io_path, "i.wav"))):
        input_audio_path = os.path.normpath(os.path.join(real_io_path, "i.wav"))
    else:
        input_audio_path = os.path.normpath(os.path.join(real_io_path, "i.mp3"))

    print("Loading files...")
    with open(input_text_path, "r", encoding="utf-8") as file:
        lines = lrcfmt.utat_process(file.read()) if txt_format == "uta" else list(file)
    if txt_format == "moe":
        lines = [lrcfmt.moeg_process_line(line) for line in lines]
    language_source = "given"
    if args.detect_language and lrc_language == "auto":
        import label_rules

        language_source = "undetected"
        try:
            detected = label_rules.document_language(
                [str(line) for line in lines],
                frozenset(args.detect_language.split(",")) - {"auto"},
            )
        except RuntimeError as error:  # no lid.176.ftz / fasttext: keep the generic path
            print(f"Language detection unavailable: {error}")
            detected = None
        if detected:
            lrc_language, language_source = detected, "detected"
            print(f"Lyrics stored without a language; identified as {detected}")
    result_list, source_texts = normalize_source_lines(
        lines, lrc_language, sokuon_split, hatsuon_split, args.label_rules
    )
    if not any(item.get("pron") for item in result_list):
        raise ValueError("Lyrics contain no alignable text")

    if tail_correct in (1, 2):  # 不建议使用
        result_list = func_tail_correct_v250611(result_list, tail_correct)

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
        if hn.is_english(item):
            continue
        else:
            print(f"alignment_tokens可能包含错误数据{item}")

    end_time = time.time()
    print("Lyrics text analysis executed in", round(end_time - start_time, 3), "seconds")

    try:
        audio_channels, sr = sf.read(input_audio_path, dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        decoded_audio_path = os.path.join(os.path.dirname(input_audio_path), "decoded.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                input_audio_path,
                "-vn",
                "-acodec",
                "pcm_f32le",
                decoded_audio_path,
            ],
            check=True,
        )
        audio_channels, sr = sf.read(decoded_audio_path, dtype="float32", always_2d=True)
    input_sr = sr
    if args.separate_vocals:
        if os.environ.get("FA_KARA_SINGLE_VOCAL_CHECKPOINT", "true").lower() == "true":
            from separate_vocals import separate_vocals_single_checkpoint as separate_vocals
        else:
            from separate_vocals import separate_vocals
        audio_file, sr = separate_vocals(audio_channels, sr, use_gpu=align_use_gpu)
    else:
        audio_file = audio_channels.mean(axis=1)
    non_silent_ranges = (
        non_silent_recog(audio_file, sr, silent_window_s, tail_thres_pct, tail_thres_ratio)
        if head_correct
        else []
    )

    def get_align_function(model_name):
        "Select FA model"
        if model_name == "MMS_FA_torch":
            from align import align_audio_with_text

            return align_audio_with_text
        if model_name == "yohane_hf":
            return None  # one global pass below
        raise ValueError(f"Unsupported FA model: '{model_name}'. ")

    align_func = get_align_function(fa_model_select)  # TODO: 面向对象方法实现

    y_processed = time_stretch_audio(audio_file, audio_speed)
    print("Aligning...")
    opening_backing_lines = set()
    if fa_model_select == "yohane_hf":
        # One global CTC pass over the whole song, from the audio alone; the lyric lines
        # give line ranges and each line's pace.
        from align_yohane import align_audio_with_timeline

        background_lines = None
        if args.background_decoding:
            # B0: lines that are one bracketed span are background vocals, decoded on the background
            # head ("(Three) stop your staring at my- (hey!)" is a lead line with inline brackets).
            background_lines = [
                bool(re.fullmatch(r"\s*[(（][^()（）]*[)）]\s*", str(source_texts[i])))
                for i, tokens in enumerate(alignment_token_lines)
                if tokens
            ]
            # Backing lines written before the first lead line are sung alone, so the lead head
            # hears them and the background head does not ("Ah-ah" over an intro): placed on the
            # background head they were squeezed against the first lead line, whose first word took
            # the intro instead. Keep them on the lead path, in order; they stay backing lines for
            # display. FA_KARA_B0_OPENING_ON_LEAD=0 puts them back on the background head.
            if os.environ.get("FA_KARA_B0_OPENING_ON_LEAD", "1") == "1" and not all(
                background_lines
            ):
                with_tokens = [i for i, tokens in enumerate(alignment_token_lines) if tokens]
                for position, flag in enumerate(background_lines):
                    if not flag:
                        break
                    background_lines[position] = False
                    opening_backing_lines.add(with_tokens[position])
        optional_tokens = None
        if args.optional_brackets:
            # Inline brackets ("I know ('til the day that I die)") are backing vocals the lead path
            # may leave out; lines that are one bracketed span stay whole (B0 or the lead).
            optional_tokens, cursor = [], 0
            for i, tokens in enumerate(alignment_token_lines):
                flags = bracketed[cursor : cursor + len(tokens)]
                # A line needs a lead word to stay on the path ("(oh) (yeah)" is kept whole).
                optional_tokens.extend(flag and not all(flags) for flag in flags)
                cursor += len(tokens)
        inline_backing_tokens = None
        if args.inline_backing:
            inline_backing_tokens, cursor = [], 0
            for i, tokens in enumerate(alignment_token_lines):
                flags = bracketed[cursor : cursor + len(tokens)]
                inline_backing_tokens.extend(flag and not all(flags) for flag in flags)
                cursor += len(tokens)

        def run_alignment(background_lines):
            return align_audio_with_timeline(
                y_processed,
                [tokens for tokens in alignment_token_lines if tokens],
                sr=sr,
                speed=audio_speed,
                use_gpu=align_use_gpu,
                hf_model_id=hf_model_path,
                min_letter_ms=args.min_letter_ms,
                min_token_ms=args.min_token_ms,
                timing_offset_ms=args.timing_offset_ms,
                duration_pace_ratio=args.duration_pace_ratio,
                vocal_activity_weight=args.vocal_activity_weight,
                exact_frame_clock=args.exact_frame_clock,
                subframe_refinement=args.subframe_refinement,
                onset_offsets_ms=json.loads(args.onset_offsets) if args.onset_offsets else None,
                edge_weight=args.edge_weight,
                vocal_activity_source=args.vocal_activity_source,
                vocal_activity_file=args.vocal_activity_file,
                blank_free_tokens=args.blank_free_tokens,
                background_lines=background_lines,
                optional_tokens=optional_tokens,
                inline_backing_tokens=inline_backing_tokens,
                inline_backing_anchor_words=args.inline_backing_anchor,
                gap_onset_pull=args.gap_onset_pull,
                fusion=args.fusion,
                fusion_model=args.fusion_model,
            )

        alignment_results = run_alignment(background_lines)
        # Off by default (FA_KARA_CRAMMED_BACKGROUND_MS=120 restores it). The rule took a line the
        # lead path could only fit by cramming its words (mean word shorter than the threshold) for
        # a chant sung under the lead (Boy With Luv's later "Oh-whoa") and decoded it again as a
        # background line, free to overlap its neighbours. Fast rap is naturally that quick, so the
        # rule pulled real lead lines out of order and showed two lead lines at once: with it on,
        # 175 of 1,078 library songs had overlapping lead lines (591 pairs; 42 with it off).
        crammed_ms = float(os.environ.get("FA_KARA_CRAMMED_BACKGROUND_MS", "0"))
        if crammed_ms > 0 and args.background_decoding:
            flags = list(background_lines or [False] * sum(1 for t in alignment_token_lines if t))
            sizes = [len(tokens) for tokens in alignment_token_lines if tokens]
            cursor, crammed = 0, []
            for index, size in enumerate(sizes):
                words = alignment_results[cursor : cursor + size]
                cursor += size
                mean_ms = (
                    1000
                    * sum(
                        float(w.get("original_end", 0)) - float(w.get("original_start", 0))
                        for w in words
                    )
                    / max(1, size)
                )
                if not flags[index] and size >= 4 and mean_ms < crammed_ms:
                    crammed.append(index)
            if crammed and len(crammed) + sum(flags) < len(flags):
                for index in crammed:
                    flags[index] = True
                print(f"Re-aligning {len(crammed)} crammed line(s) as background: {crammed}")
                alignment_results = run_alignment(flags)
    else:
        alignment_results = align_func(
            y_processed, alignment_tokens, non_silent_ranges, sr, audio_speed, use_gpu=align_use_gpu
        )

    if len(alignment_results) != len(alignment_tokens):
        raise RuntimeError(
            f"Alignment returned {len(alignment_results)} tokens for {len(alignment_tokens)} inputs"
        )
    for i, result in enumerate(alignment_results):
        original_index = token_to_index_map[i]
        result_list[original_index]["start"] = result["start"]
        result_list[original_index]["end"] = result["end"]
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
    alignment_document = {
        "schema_version": 1,
        "alignment": {"mode": "global_ctc_audio_only", "lines": canonical_lines},
        "diagnostics": {
            "inference_passes": max(
                (int(result.get("inference_passes", 1)) for result in alignment_results), default=0
            ),
            "line_count": len(canonical_lines),
            "token_count": len(finite_scores),
            "mean_ctc_score": sum(finite_scores) / len(finite_scores) if finite_scores else 0.0,
            "minimum_ctc_score": min(finite_scores) if finite_scores else 0.0,
            **source_diagnostics,
            "line_count": len(canonical_lines),
            "acoustic_line_source_indexes": [
                i for i, tokens in enumerate(alignment_token_lines) if tokens
            ],
            "normalizer_identity": file_identity(hn.__file__),
            "label_rules": args.label_rules,
            "source_identity": file_identity(input_text_path),
            "audio_identity": file_identity(input_audio_path),
            "checkpoint_identity": file_identity(hf_model_path)
            if hf_model_path and os.path.exists(hf_model_path)
            else None,
            "max_blank_hold_seconds": float(
                os.environ.get("FA_KARA_MAX_BLANK_HOLD_SECONDS", "0.2")
            ),
            "model_path": hf_model_path or "torchaudio.pipelines.MMS_FA",
            "device": "cuda"
            if (align_use_gpu or not hf_model_path) and __import__("torch").cuda.is_available()
            else "cpu",
            "normalization_language": lrc_language,
            "normalization_language_source": language_source,
            "opening_backing_lines_on_lead": len(opening_backing_lines),
            "audio_source": "demucs_vocals" if args.separate_vocals else "input_audio",
            "inference_config": vars(args),
        },
    }
    with open(os.path.join(real_io_path, "o.alignment.json"), "w", encoding="utf-8") as f:
        json.dump(alignment_document, f, ensure_ascii=False, separators=(",", ":"))

    result_list = non_silent_head_adjust(result_list, non_silent_ranges)

    if tail_correct == 3:
        result_list = func_tail_correct_v250615(
            result_list, audio_file, sr, tail_thres_pct, tail_thres_ratio
        )

    if output_characters_per_line > 0:
        split_long_segments(result_list, max_length=output_characters_per_line)

    main_output = process_main(result_list, ruby_tag_offset, bpm, beats_per_bar)
    ruby_output = process_ruby_V2(result_list)
    content = f"{main_output}\n{ruby_output}"
    with open(os.path.join(real_io_path, "o_ruby.lrc"), "w", encoding="utf-8") as f:
        f.write(content)
    rlf_output = process_rlf(result_list)
    with open(os.path.join(real_io_path, "o_rlf.lrc"), "w", encoding="utf-8") as f:
        f.write(rlf_output)
    ass_output = norm2ass.process_norm2assV2(result_list)
    ass_head = """[Script Info]
ScriptType: v4.00+
YCbCr Matrix: TV.601
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Source Han Serif,71,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1.99999,1.99999,2,11,11,101,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(os.path.join(real_io_path, "o.ass"), "w", encoding="utf-8") as f:
        f.write(ass_head + ass_output)
    # hrhlrc_output = ''
    # for i in ass_output.splitlines():
    #     hrhlrc_output += ass2lrc.ass2lrc(i, 0)+'\n'
    # with open(os.path.join(real_io_path, 'o_hrh.lrc'), 'w', encoding='utf-8') as f:
    #     f.write(hrhlrc_output)
    print("Success!")


if __name__ == "__main__":
    main()
