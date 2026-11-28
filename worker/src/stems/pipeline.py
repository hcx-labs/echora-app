"""Karaoke stems: each song's vocals and instrumental, stored for the Stage player to remix.

The vocals come from the same Roformer separation (and the same prepared-audio cache) as the
rest of analysis; the instrumental is the decoded mix minus the vocals, so the two add back up
to the song. Both are encoded into one four-channel Opus file in WebM (vocals left and right,
then instrumental left and right): one stream keeps the stems sample-locked in the browser.
Unlike prepared audio, the files are durable: they live in ``ECHORA_KARAOKE_STEMS_DIR``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np

from core.db.connection import connect
from core.errors import InputUnavailableError, ProcessingError
from core.features import feature_enabled
from core.integrations.navidrome import NavidromeClient
from core.media.config import KaraokeStemsConfig
from core.resync import store as resync
from core.revisions import SEPARATION_REVISION
from core.stems import store
from core.stems.files import stems_path, stems_root
from worker.planning.audio import resolve_library_id
from worker.processing import Item, ItemResult, ProcessingContext, Processor

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
"""Opus's own rate; the 44.1 kHz stems are resampled once, here, with soxr."""
ENCODE_TIMEOUT_SECONDS = 600
STEMS_REVISION = "stems-v1"


def stems_recipe(bitrate_kbps: int | None = None) -> str:
    """What a stored file was made with; a change makes every song's stems again."""
    rate = bitrate_kbps or KaraokeStemsConfig.load().bitrate_kbps
    return f"{STEMS_REVISION}:{SEPARATION_REVISION}:opus-webm-quad-{rate}x2"


def encode_stems(
    source: bytes, bitrate_kbps: int, check: Callable[[], None] = lambda: None
) -> tuple[bytes, float]:
    """One song's stems file and its duration in seconds. The same code runs on Modal."""
    import soxr

    from worker.media.audio import decode_audio_channels
    from worker.media.preprocessing import vocal_stereo

    vocals = vocal_stereo(source, check)
    mix = decode_audio_channels(source, 44100, 2)
    if vocals.shape != mix.shape or not len(mix):
        raise ValueError("The vocal stem does not match the decoded song")
    check()
    duration = len(mix) / 44100
    # Vocals first, then the instrumental: quad's front and back pairs.
    channels = np.concatenate([vocals, mix - vocals], axis=1)
    del vocals, mix
    channels = np.ascontiguousarray(
        soxr.resample(channels, 44100, SAMPLE_RATE, quality="HQ"), dtype="<f4"
    )
    check()
    with tempfile.TemporaryDirectory(prefix="echora-stems-") as directory:
        output = Path(directory) / "stems.webm"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "4", "-channel_layout", "quad",
            "-i", "pipe:0",
            "-c:a", "libopus", "-b:a", f"{2 * bitrate_kbps}k", "-vbr", "on",
            "-application", "audio", "-mapping_family", "1",
            "-f", "webm", str(output),
        ]  # fmt: skip
        result = subprocess.run(
            command,
            input=channels.tobytes(),
            capture_output=True,
            timeout=ENCODE_TIMEOUT_SECONDS,
            check=False,
        )
        del channels
        if result.returncode != 0 or not output.is_file() or not output.stat().st_size:
            raise RuntimeError("ffmpeg could not encode the karaoke stems")
        check()
        return output.read_bytes(), duration


def _write_file(track_id: uuid.UUID, content: bytes) -> str:
    """Store a stems file atomically; returns its path relative to the stems directory."""
    relative = f"{str(track_id)[:2]}/{track_id}-{uuid.uuid4().hex[:12]}.webm"
    path = stems_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".stems-", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.chmod(0o644)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return relative


def _remove(relative: str | None) -> None:
    if not relative:
        return
    try:
        stems_path(relative).unlink(missing_ok=True)
    except (OSError, ValueError):
        logger.warning("Could not remove replaced karaoke stems")


class StemsItem(Item):
    """A song with lyrics whose karaoke stems are missing or of another recipe."""

    track_id: uuid.UUID
    external_id: str

    def track(self) -> dict[str, object]:
        """The song as the setup screen names it."""
        return {"id": self.external_id, "title": self.label}


Stems = tuple[bytes, float]
"""The encoded file and the song's duration in seconds."""


class KaraokeStemsProcessor(Processor[None, StemsItem, Stems]):
    """Separates and stores the vocals and instrumental of the batch's songs with lyrics.

    Off unless switched on for this location. On Modal every song is separated and encoded in
    one call; only the encoded files come back.
    """

    name = "karaoke_stems"
    summary_key = "karaoke_stems"
    counters = ("stored",)
    activity = "Separating karaoke stems for"
    current_message = "Karaoke stems already current"
    announces_items = True
    # The separation is the costly part: retry the batch rather than drop a file it made.
    storage_failure_fails_job = True

    def __init__(self) -> None:
        super().__init__()
        self.enabled = False

    def plan(self, context: ProcessingContext) -> list[StemsItem]:
        with connect() as connection:
            self.enabled = feature_enabled(connection, "karaoke_stems")
            if not self.enabled or not context.track_ids:
                return []
            library_id = resolve_library_id(connection, context.library_url)
            self.bitrate = KaraokeStemsConfig.load().bitrate_kbps
            self.recipe = stems_recipe(self.bitrate)
            rows = store.songs_without_stems(
                connection, library_id, list(context.track_ids), self.recipe
            )
        return [
            StemsItem(
                key=str(row["track_id"]),
                label=row["title"] or "",
                track_id=row["track_id"],
                external_id=str(row["external_id"]),
            )
            for row in rows
        ]

    def open(self, context: ProcessingContext, items) -> None:
        root = stems_root()
        root.mkdir(parents=True, exist_ok=True)
        if not os.access(root, os.W_OK):
            # Every song would fail the same way; fail the job so it is visible and retried.
            raise PermissionError("The karaoke stems directory is not writable")
        self.connection = connect()
        self.client = NavidromeClient(*context.library)
        self.client.__enter__()
        self.unavailable: set[str] = set()
        self.encoded = None
        if context.remote is not None:
            digests = []
            for item in items:
                try:
                    digests.append(
                        context.remote.upload_audio(self.client.audio_bytes(item.external_id))
                    )
                except Exception:  # noqa: BLE001 - one song's audio
                    self.unavailable.add(item.key)
            self.encoded = iter(context.remote.karaoke_stems(digests, self.bitrate))

    def process(self, context, items):
        from worker.media.roformer import separation_phase

        with separation_phase():
            for item in items:
                context.check()
                if item.key in self.unavailable:
                    yield ItemResult.failed(item, InputUnavailableError("Audio unavailable"))
                    continue
                try:
                    if self.encoded is None:
                        try:
                            source = self.client.audio_bytes(item.external_id)
                        except Exception:  # noqa: BLE001 - one song's audio
                            yield ItemResult.failed(
                                item, InputUnavailableError("Audio unavailable")
                            )
                            continue
                        output = encode_stems(source, self.bitrate, context.check)
                        del source
                    else:
                        output = next(self.encoded)
                        if isinstance(output, BaseException):
                            raise output
                except Exception as error:  # noqa: BLE001 - one song's separation or encoding
                    logger.error("Karaoke stems failed for %s: %s", item.key, type(error).__name__)
                    # Recorded, so later syncs do not separate this song again for this recipe.
                    store.save_stems(
                        self.connection, item.track_id, self.recipe, error=type(error).__name__
                    )
                    # Asked for once: a failure is the answer until it is asked for again.
                    resync.done(self.connection, item.track_id, "karaoke_stems")
                    self.connection.commit()
                    yield ItemResult.failed(
                        item, ProcessingError("Stems failed", stage=type(error).__name__)
                    )
                    continue
                yield ItemResult(item=item, outcome="stored", output=output)

    def persist(self, context, item: StemsItem, output: Stems) -> None:
        content, duration = output
        relative = _write_file(item.track_id, content)
        try:
            replaced = store.save_stems(
                self.connection,
                item.track_id,
                self.recipe,
                path=relative,
                file_bytes=len(content),
                duration_seconds=duration,
            )
            resync.done(self.connection, item.track_id, "karaoke_stems")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            _remove(relative)
            raise
        _remove(replaced)

    def close(self) -> None:
        if hasattr(self, "client"):
            self.client.__exit__(None, None, None)
        if hasattr(self, "connection"):
            self.connection.close()
