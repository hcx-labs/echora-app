"""Voice evidence with Essentia models: whether a song is sung, and the voice's gender."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import threading
import uuid
from pathlib import Path

import essentia.standard as es
import numpy as np
import onnxruntime as ort
import psycopg
import torch

from core.analysis import store as runs
from core.db.connection import connect
from core.embeddings.representations import configure_representations, voice_config
from core.errors import InputUnavailableError, ProcessingError
from core.integrations.navidrome import NavidromeClient
from core.voice import store
from worker.media.audio import decode_audio
from worker.models.config import ModelsConfig
from worker.planning.audio import resolve_library_id
from worker.processing import Item, ItemResult, ProcessingContext, Processor

logger = logging.getLogger(__name__)

VOICE_EMBEDDING_TYPE = "voice-gender"
VOICE_MODEL_NAME = "mtg-jamendo-voice-gender-v2"

# Essentia MTG-Jamendo classification heads (ONNX): gender (female/male) and
# voice_instrumental, both consuming discogs-effnet frame embeddings computed
# from log-mel patches. All three files are preloaded by
# worker.models.download into ESSENTIA_MODELS_DIR; nothing is
# downloaded at inference time.
_EMBEDDING_MODEL_FILE = "discogs-effnet-bsdynamic-1.onnx"
_GENDER_MODEL_FILE = "gender-discogs-effnet-1.onnx"
_VOICE_INSTRUMENTAL_MODEL_FILE = "voice_instrumental-discogs-effnet-1.onnx"
_PATCH_FRAMES = 128

_model: tuple | None = None
_model_lock = threading.Lock()


def _model_directory() -> str:
    return ModelsConfig.load().essentia_dir


def _require_model_file(filename: str) -> str:
    path = Path(_model_directory()) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Essentia model {filename} is missing from {ModelsConfig.load().essentia_dir}; "
            "run `python -m worker.models.download` on a machine with network access"
        )
    return str(path)


def _aggregate_outputs(gender: np.ndarray, voice: np.ndarray) -> dict[str, float]:
    """Combine [female, male] and [instrumental, voice] patch probabilities."""
    gender = np.asarray(gender, dtype=np.float32)
    voice = np.asarray(voice, dtype=np.float32)
    if gender.shape != voice.shape or gender.ndim != 2 or gender.shape[1] != 2:
        raise ValueError(
            f"Unexpected classifier output shapes: gender={gender.shape}, voice={voice.shape}"
        )
    instrumental = np.clip(voice[:, 0], 0.0, 1.0)
    vocal = np.clip(voice[:, 1], 0.0, 1.0)
    values = {
        "instrumental": float(instrumental.mean()),
        "female": float((vocal * np.clip(gender[:, 0], 0.0, 1.0)).mean()),
        "male": float((vocal * np.clip(gender[:, 1], 0.0, 1.0)).mean()),
    }
    total = sum(values.values()) or 1.0
    return {label: value / total for label, value in values.items()}


class VoiceGenderModel:
    """Voice-activity and vocal-gender classifier over 16 kHz mono audio.

    Returns the mean activation per class across the whole track. These are
    calibrated per track by the caller (library percentile), which is what tag
    matching consumes.
    """

    name = VOICE_MODEL_NAME
    labels = ("instrumental", "female", "male")

    def __init__(self) -> None:
        embedding_graph = _require_model_file(_EMBEDDING_MODEL_FILE)
        gender_graph = _require_model_file(_GENDER_MODEL_FILE)
        voice_graph = _require_model_file(_VOICE_INSTRUMENTAL_MODEL_FILE)
        options = ort.SessionOptions()
        threads = max(4, (os.cpu_count() or 4) // 2)
        options.intra_op_num_threads = options.inter_op_num_threads = threads
        self._embedder = ort.InferenceSession(embedding_graph, options)
        self._gender = ort.InferenceSession(gender_graph, options)
        self._voice = ort.InferenceSession(voice_graph, options)
        # MusiCNN input parameters, hardcoded upstream to match training. Use
        # essentia's own TensorflowInputMusiCNN so the features match the
        # classifier's preprocessing exactly ((mel+1)*10000 -> log10, 96 bands).
        self._input = es.TensorflowInputMusiCNN()

    def _mel_patches(self, waveform: np.ndarray) -> np.ndarray:
        hop = 256
        limit = waveform.size - 512 + 1
        frames = []
        for start in range(0, max(limit, 0), hop):
            frames.append(self._input(waveform[start : start + 512]))
        if len(frames) < _PATCH_FRAMES:
            raise ValueError("Audio is too short for voice classification")
        usable = (len(frames) // _PATCH_FRAMES) * _PATCH_FRAMES
        stacked = np.stack(frames[:usable]).astype(np.float32)
        return stacked.reshape(-1, _PATCH_FRAMES, stacked.shape[-1])

    def classify(self, waveform: np.ndarray) -> dict[str, float]:
        values, _ = self.classify_with_activity(waveform)
        return values

    def classify_with_activity(
        self, waveform: np.ndarray
    ) -> tuple[dict[str, float], dict[str, object]]:
        if waveform.size == 0:
            raise ValueError("Decoded audio is empty")
        patches = self._mel_patches(waveform)
        embeddings = self._embedder.run(
            ["embeddings"], {self._embedder.get_inputs()[0].name: patches}
        )[0]
        gender = np.asarray(
            self._gender.run(None, {self._gender.get_inputs()[0].name: embeddings})[0],
            dtype=np.float32,
        )
        voice = np.asarray(
            self._voice.run(None, {self._voice.get_inputs()[0].name: embeddings})[0],
            dtype=np.float32,
        )
        # Preserve vocal presence in gender evidence. Renormalizing female and
        # male independently lets tiny, arbitrary activations from an
        # instrumental patch look decisive.
        values = _aggregate_outputs(gender, voice)
        duration = waveform.size / 16_000
        windows = [
            {
                "start_seconds": index * _PATCH_FRAMES * 256 / 16_000,
                "end_seconds": min(
                    duration, (index * _PATCH_FRAMES * 256 + 127 * 256 + 512) / 16_000
                ),
                "vocal_activation": float(np.clip(row[1], 0, 1)),
            }
            for index, row in enumerate(voice)
        ]
        activity = {
            "duration_seconds": duration,
            "windows": windows,
            "unanalyzed_tail_seconds": max(0.0, duration - windows[-1]["end_seconds"])
            if windows
            else duration,
            "confidence_calibrated": False,
            "method": "discogs-effnet_voice_instrumental",
        }
        return values, activity


def shared_voice_model() -> VoiceGenderModel:
    global _model
    with _model_lock:
        if _model is None:
            _model = (VoiceGenderModel(),)
        return _model[0]


def _vector_literal(vector) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def _create_run(connection: psycopg.Connection) -> uuid.UUID:
    config = voice_config()
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    environment = {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"}
    return store.voice_run(connection, VOICE_MODEL_NAME, config_hash, config, environment)


def _store_voice(
    connection: psycopg.Connection,
    track_id: uuid.UUID,
    run_id: uuid.UUID,
    values: dict[str, float],
    activity: dict[str, object],
) -> None:
    vector = [values[label] for label in VoiceGenderModel.labels]
    store.save_voice(
        connection,
        track_id,
        run_id,
        VOICE_EMBEDDING_TYPE,
        _vector_literal(vector),
        len(vector),
        activity,
    )


def classify_track(audio: bytes) -> tuple[dict[str, float], dict[str, object]]:
    """Voice evidence for one track. The same code runs locally and on Modal."""
    return shared_voice_model().classify_with_activity(decode_audio(audio, sample_rate=16_000))


def _fetch_stream(client: NavidromeClient, source_id: str) -> np.ndarray:
    return decode_audio(client.audio_bytes(source_id), sample_rate=16_000)


class VoiceItem(Item):
    """A song without voice evidence in the current run."""

    track_id: uuid.UUID
    external_id: str

    def track(self) -> dict[str, object]:
        """The song as the setup screen names it."""
        return {"id": self.external_id, "title": self.label}


Evidence = tuple[dict[str, float], dict[str, object]]
"""Mean activations by label, and the song's vocal activity."""


class VoiceProcessor(Processor[None, VoiceItem, Evidence]):
    """Classifies lead-vocal gender for the batch's songs that lack voice evidence.

    Stores one 3-dimensional vector per song in the embeddings table
    (embedding_type='voice-gender'): [instrumental, female, male] mean activations, normalized to
    sum to one, and the song's vocal activity. On Modal every song is classified in one call.
    """

    name = "voice"
    summary_key = "voice"
    counters = ("classified",)
    activity = "Classifying vocals for"
    current_message = "Voice classification already current"
    announces_items = True

    def __init__(self, limit: int | None = None) -> None:
        super().__init__()
        self.limit = limit

    def plan(self, context: ProcessingContext) -> list[VoiceItem]:
        with connect() as connection:
            library_id = resolve_library_id(connection, context.library_url)
            configure_representations(connection)
            # Get or create: the run of the current classifier settings.
            self.run_id = _create_run(connection)
            rows = store.songs_without_voice(
                connection,
                library_id,
                list(context.track_ids),
                VOICE_EMBEDDING_TYPE,
                self.run_id,
                self.limit,
            )
        return [
            VoiceItem(
                key=str(row["track_id"]),
                label=row["title"] or "",
                track_id=row["track_id"],
                external_id=row["external_id"],
            )
            for row in rows
        ]

    def open(self, context: ProcessingContext, items) -> None:
        self.connection = connect()
        self.client = NavidromeClient(*context.library)
        self.client.__enter__()
        self.model = None
        self.classified = None
        self.unavailable: dict[str, BaseException] = {}
        if context.remote is not None:
            # Upload once, classify every song in one Modal call; results stream back in order.
            digests = []
            for item in items:
                try:
                    digests.append(
                        context.remote.upload_audio(self.client.audio_bytes(item.external_id))
                    )
                except Exception as error:  # noqa: BLE001 - one song's audio
                    self.unavailable[item.key] = error
            self.classified = iter(context.remote.voice(digests))
        context.publish(
            {
                "phase": "models",
                "message": "Loading voice classifier",
                "completed": 0,
                "total": 1,
                "unit": "models",
            }
        )
        if context.remote is None:
            self.model = shared_voice_model()

    def process(self, context, items):
        for item in items:
            context.check()
            if item.key in self.unavailable:
                yield ItemResult.failed(item, InputUnavailableError("Audio unavailable"))
                continue
            try:
                if self.classified is None:
                    evidence = self.model.classify_with_activity(
                        _fetch_stream(self.client, item.external_id)
                    )
                else:
                    evidence = next(self.classified)
                    if isinstance(evidence, BaseException):
                        raise evidence
            except Exception as error:  # noqa: BLE001 - one song's audio or classification
                failure = ProcessingError("Voice classification failed", stage=type(error).__name__)
                yield ItemResult.failed(item, failure)
                continue
            yield ItemResult(item=item, outcome="classified", output=evidence)
        # Every planned song has a result: the run is complete.
        runs.complete_run(self.connection, self.run_id)
        self.connection.commit()

    def persist(self, context, item: VoiceItem, output: Evidence) -> None:
        values, activity = output
        try:
            _store_voice(self.connection, item.track_id, self.run_id, values, activity)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def close(self) -> None:
        global _model
        model = getattr(self, "model", None)
        with _model_lock:
            if model is not None and _model is not None and _model[0] is model:
                _model = None
        self.model = None
        if hasattr(self, "client"):
            self.client.__exit__(None, None, None)
        if hasattr(self, "connection"):
            self.connection.close()
