"""Lyrics language detection.

Detects the language of track lyrics line by line using fastText lid.176,
with a deterministic Unicode-script fallback for CJK languages and an
aggregated per-track language distribution stored in `lyrics.provenance`.

Single-language labels are wrong for multilingual songs (J-pop and K-pop mix
languages constantly), so the module always produces a distribution:
{"languages": {"ja": 0.62, "en": 0.35}, ...} plus confidence metadata.
"""

from __future__ import annotations

from .settings import get_settings

import hashlib
import logging
import os
import re
import urllib.request
from collections import defaultdict

logger = logging.getLogger(__name__)

# fastText lid.176.bin (CC BY-SA 3.0) from a Hugging Face mirror pinned to a commit; the file is
# byte-identical to Meta's. Meta's own host (dl.fbaipublicfiles.com) served an expired certificate
# from 2026-11-11, and every lyric ingested meanwhile was stored without a language.
MODEL_URL = (
    "https://huggingface.co/julien-c/fasttext-language-id/resolve/"
    "9f2e94d2fd6b2615d26f5ec5b34cf7e2f38cb86e/lid.176.bin"
)
MODEL_SHA256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
MODEL_PATH = get_settings().lid_model_path

# IndicLID-FTR (AI4Bharat): fastText model trained on romanized Indic text.
# Catches Hindi/Urdu/etc. written in Latin script ("Hinglish"), which generic
# LID models misclassify as European languages. Lazy-downloaded on first use.
INDICLID_FTR_URL = "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftr.zip"
INDICLID_FTR_PATH = get_settings().indiclid_ftr_path
INDICLID_FTR_THRESHOLD = 0.6
INDIC_LABELS = {
    "hin": "hi",
    "urd": "ur",
    "pan": "pa",
    "ben": "bn",
    "guj": "gu",
    "mar": "mr",
    "tam": "ta",
    "tel": "te",
    "kan": "kn",
    "mal": "ml",
    "ori": "or",
    "sin": "si",
    "nep": "ne",
    "snd": "sd",
    "asm": "as",
    "kok": "kok",
    "mni": "mni",
    "doi": "doi",
    "kas": "ks",
    "san": "sa",
    "bho": "bho",
    "brx": "brx",
    "mai": "mai",
}

# Predefined confidence thresholds. A line whose fastText probability falls
# below MIN_LINE_CONFIDENCE is discarded. A track whose confident lines cover
# less than MIN_COVERED_WEIGHT of the lyric weight, or whose weighted mean
# confidence falls below MIN_MEAN_CONFIDENCE, is flagged low-confidence and
# treated as unknown by language curations.
MIN_LINE_CONFIDENCE = 0.5
MIN_COVERED_WEIGHT = 0.6
MIN_MEAN_CONFIDENCE = 0.5

# Share of the target language a track needs before a "primarily" language
# curation counts it as a first-tier match.
PRIMARY_SHARE = 0.5

LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Mandarin",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "id": "Indonesian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ar": "Arabic",
    "hi": "Hindi",
    "ur": "Urdu",
    "tr": "Turkish",
}

_SECTION_MARKER = re.compile(r"^\s*[\[(（【].*?[\])）】]\s*$")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

_FASTTEXT_MODEL = None
_INDICLID_FTR = None


def _load_ftr():
    """Lazily download and load the IndicLID-FTR romanized-Indic model."""
    global _INDICLID_FTR
    if _INDICLID_FTR is not None:
        return _INDICLID_FTR
    if _INDICLID_FTR is False:
        return None
    try:
        import fasttext
    except ImportError:
        _INDICLID_FTR = False
        return None
    if not os.path.exists(INDICLID_FTR_PATH):
        os.makedirs(os.path.dirname(INDICLID_FTR_PATH) or ".", exist_ok=True)
        try:
            import io, zipfile

            request = urllib.request.Request(
                INDICLID_FTR_URL, headers={"User-Agent": "Mozilla/5.0"}
            )
            archive = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(request).read()))
            inner = next(
                name for name in archive.namelist() if name.endswith("model_baseline_roman.bin")
            )
            with open(INDICLID_FTR_PATH, "wb") as handle:
                handle.write(archive.read(inner))
        except Exception:
            _INDICLID_FTR = False
            return None
    try:
        model = fasttext.load_model(INDICLID_FTR_PATH)
    except Exception:
        _INDICLID_FTR = False
        return None
    _INDICLID_FTR = model
    return model


def _is_latin(line: str) -> bool:
    counts = _script_counts(line)
    letters = sum(counts.values())
    return letters > 0 and counts["latin"] / letters > 0.5


def _load_fasttext():
    global _FASTTEXT_MODEL
    if _FASTTEXT_MODEL is not None:
        return _FASTTEXT_MODEL
    try:
        import fasttext
    except ImportError:
        return None
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
        # Download beside the target, check it and rename: an interrupted or altered download is
        # never loaded.
        partial = f"{MODEL_PATH}.part"
        try:
            urllib.request.urlretrieve(MODEL_URL, partial)
            with open(partial, "rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != MODEL_SHA256:
                os.remove(partial)
                raise RuntimeError(f"lid.176.bin sha256 is {digest}")
            os.replace(partial, MODEL_PATH)
        except Exception as error:
            # Lyrics are then stored without a language; say so rather than fail silently.
            logger.warning("language model %s unavailable: %s", MODEL_PATH, error)
            return None
    try:
        model = fasttext.load_model(MODEL_PATH)
    except Exception as error:
        logger.warning("language model %s could not be loaded: %s", MODEL_PATH, error)
        return None
    model.get_labels  # touch to fail early on corrupt files
    _FASTTEXT_MODEL = model
    return model


def _script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for character in text:
        code = ord(character)
        if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
            counts["kana"] += 1
        elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            counts["han"] += 1
        elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            counts["hangul"] += 1
        elif character.isalpha():
            counts["latin"] += 1
    return counts


def _script_language(line: str) -> str | None:
    """Deterministic CJK detection; returns None when script is ambiguous."""
    counts = _script_counts(line)
    total = sum(counts.values())
    if not total:
        return None
    if counts["hangul"] / total > 0.15:
        return "ko"
    if counts["kana"] / total > 0.08:
        return "ja"
    if counts["han"] / total > 0.25:
        return "zh"
    return None


def _eligible_lines(text: str) -> list[str]:
    lines = []
    for raw in re.split(r"[\n\r]+", text):
        line = raw.strip()
        if not line or _SECTION_MARKER.match(line):
            continue
        words = _WORD.findall(line)
        if len(words) < 2 or len(line) < 6:
            continue
        lines.append(" ".join(words))
    return lines


def detect_distribution(text: str | None) -> dict[str, object]:
    """Classify lyrics into a per-track language distribution.

    Returns {} when there is no usable text. Otherwise:
    {languages: {code: share}, primary_language, mean_confidence,
     coverage, confident}
    """
    if not text or not text.strip():
        return {}
    lines = _eligible_lines(text)
    if not lines:
        return {}

    model = _load_fasttext()
    ftr = _load_ftr()
    weights: dict[str, float] = defaultdict(float)
    confident_weight = 0.0
    total_weight = 0.0
    confidence_sum = 0.0

    for line in lines:
        weight = float(len(line))
        total_weight += weight
        language: str | None = None
        confidence = 0.0
        script = _script_language(line)
        if model is not None:
            try:
                # fasttext-wheel 0.9.2's predict() is broken under NumPy 2
                # (np.array(..., copy=False)), so use the raw pybind API.
                pairs = model.f.predict(line.lower(), 1, 0.0, "strict")
            except Exception:
                try:
                    labels, probabilities = model.predict(line.lower(), k=1)
                except Exception:
                    labels, probabilities = [], []
                pairs = [(probabilities[index], labels[index]) for index in range(len(labels))]
            if pairs:
                probability, label = pairs[0]
                language = str(label).removeprefix("__label__")[:2]
                confidence = float(probability)
        if script is not None:
            # Script evidence is deterministic and always wins for CJK.
            language = script
            confidence = 0.99
        elif ftr is not None and _is_latin(line):
            # Romanized Indic languages: the specialist model wins when sure.
            try:
                pairs = ftr.f.predict(line.lower(), 1, 0.0, "strict")
            except Exception:
                pairs = []
            if pairs and pairs[0][0] >= INDICLID_FTR_THRESHOLD:
                code = pairs[0][1].replace("__label__", "")
                mapped = INDIC_LABELS.get(code[:3])
                if mapped:
                    language = mapped
                    confidence = float(pairs[0][0])
        if language is None or confidence < MIN_LINE_CONFIDENCE:
            continue
        confident_weight += weight
        confidence_sum += confidence * weight
        weights[language] += weight

    if not confident_weight:
        return {
            "languages": {},
            "primary_language": None,
            "mean_confidence": 0.0,
            "coverage": 0.0,
            "confident": False,
        }

    # Romanized Indic languages (Hindi/Punjabi/Maithili/...) are hard to tell
    # apart line by line, but a full song is decisive. When the track-level
    # FTR prediction is confident, consolidate the whole Indic-family vote
    # behind it so the primary language is stable instead of smeared.
    ftr = _load_ftr()
    if ftr is not None:
        try:
            pairs = ftr.f.predict(" ".join(lines).lower(), 1, 0.0, "strict")
        except Exception:
            pairs = []
        if pairs and pairs[0][0] >= INDICLID_FTR_THRESHOLD:
            mapped = INDIC_LABELS.get(str(pairs[0][1]).replace("__label__", "")[:3])
            if mapped:
                family = set(INDIC_LABELS.values())
                family_weight = sum(weight for code, weight in weights.items() if code in family)
                if family_weight:
                    weights = {
                        code: weight for code, weight in weights.items() if code not in family
                    }
                    weights[mapped] = weights.get(mapped, 0.0) + family_weight

    distribution = {code: round(weight / confident_weight, 4) for code, weight in weights.items()}
    primary = max(distribution, key=distribution.get)
    mean_confidence = confidence_sum / confident_weight
    coverage = confident_weight / total_weight
    return {
        "languages": distribution,
        "primary_language": primary,
        "mean_confidence": round(mean_confidence, 4),
        "coverage": round(coverage, 4),
        "confident": coverage >= MIN_COVERED_WEIGHT and mean_confidence >= MIN_MEAN_CONFIDENCE,
    }


def language_affinity(distribution: dict[str, object] | None, target: str) -> float:
    """Share of the target language for a track; 0 when unknown/unreliable."""
    if not target or not distribution:
        return 0.0
    if not distribution.get("confident", False):
        return 0.0
    languages = distribution.get("languages") or {}
    # Hindi and Urdu are nearly identical in romanized form; either target
    # matches both shares so Hinglish tracks score consistently.
    if target in ("hi", "ur"):
        return float((languages.get("hi") or 0.0) + (languages.get("ur") or 0.0))
    return float((languages.get(target) or 0.0))
