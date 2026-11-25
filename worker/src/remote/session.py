"""Run one analysis batch's model computation on Modal (see docs/modal-compute.md).

A batch with `compute: "modal"` opens a session. Stages keep their own
planning, persistence and progress; where they would run a model, they ask the
current session instead. Source audio is uploaded once per batch, named by its
SHA-256, and removed when the batch ends.

Before any computation the session checks the deployment: the remote code must
be this worker's code, on the configured image and GPU, with the model set the
worker's settings require. Anything stale is redeployed or downloaded first.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from worker.remote.constants import APP_NAME, AUDIO_VOLUME

logger = logging.getLogger(__name__)

DEPLOY_TIMEOUT_SECONDS = 30 * 60
ROLLOVER_SECONDS = 120
ROLLOVER_POLL_SECONDS = 10

_current: ContextVar[ModalSession | None] = ContextVar("echora_remote_session", default=None)


# What a cover sends to Modal for rendering: the image, song context, seed and any given prompt.
COVER_FIELDS = (
    "sha256",
    "data",
    "content_type",
    "title",
    "artist",
    "album",
    "lyrics",
    "seed",
    "prompt",
    "failed",
)


def current() -> ModalSession | None:
    """The batch's Modal session, or None when computing locally."""
    return _current.get()


@contextmanager
def session(
    compute: str | None, factory: Callable[[], ModalSession] | None = None
) -> Iterator[ModalSession | None]:
    """Open a Modal session for the duration of a batch when it asks for one."""
    if compute != "modal":
        yield None
        return
    if factory is None:
        from worker.remote.settings import open_session as factory
    remote = factory()
    token = _current.set(remote)
    try:
        yield remote
    finally:
        _current.reset(token)
        remote.close()


def code_identity() -> str:
    """Digest of the core and worker code (with the vendored aligner), alike on both sides."""
    import core
    import worker

    digest = hashlib.sha256()
    for package in (core, worker):
        root = Path(package.__file__).resolve().parent
        for path in sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts):
            digest.update(f"{package.__name__}/{path.relative_to(root)}".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ModalConfig:
    token_id: str
    token_secret: str
    gpu: str
    image: str
    # Hugging Face token for gated models (LTX-2.5). Passed to model downloads only.
    hf_token: str | None = None

    @property
    def deployment(self) -> dict[str, str]:
        return {"image": self.image, "gpu": self.gpu}


@dataclass(frozen=True)
class RemoteModel:
    """Stands in for a local model: records the run, loads nothing locally."""

    name: str
    model_id: str
    revision: str
    device: str = "modal"


def _modal_environment(config: ModalConfig) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("MODAL_")}
    environment.update(
        {
            "MODAL_TOKEN_ID": config.token_id,
            "MODAL_TOKEN_SECRET": config.token_secret,
            "ECHORA_MODAL_IMAGE": config.image,
            "ECHORA_MODAL_GPU": config.gpu,
            "NO_COLOR": "1",
            "TERM": "dumb",
            "COLUMNS": "200",
        }
    )
    return environment


def _scrubbed(text: str, config: ModalConfig) -> str:
    for secret in (config.token_secret, config.token_id, config.hf_token):
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def _modal_cli(config: ModalConfig, *arguments: str, timeout: float) -> str:
    """Run the Modal CLI with this token only, isolated from any local Modal profile."""
    result = subprocess.run(
        [sys.executable, "-m", "modal", *arguments],
        env=_modal_environment(config),
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = _scrubbed(result.stdout + result.stderr, config)
    if result.returncode != 0:
        lines = [line for line in output.splitlines() if line.strip()]
        raise RuntimeError("Modal: " + " ".join(lines[-6:])[-1500:])
    return output


def workspace(config: ModalConfig) -> str:
    """Verify a token and return the name of its Modal workspace."""
    output = _modal_cli(config, "token", "info", timeout=60)
    match = re.search(r"Workspace:\s*(\S+)", output)
    if not match:
        raise RuntimeError("Modal did not report a workspace for this token")
    return match[1]


def deploy(config: ModalConfig) -> None:
    """Deploy this worker's analysis code to the token's workspace."""
    # Recreate, not roll over: old containers would otherwise keep answering (and stay warm
    # while being asked), and Echora only redeploys when the old code is wrong for it.
    _modal_cli(
        config,
        "deploy",
        "--strategy",
        "recreate",
        "-m",
        "worker.remote.modal_app",
        timeout=DEPLOY_TIMEOUT_SECONDS,
    )


def _complete(results, count: int) -> Iterator[object]:
    """Yield a batched call's results, finishing Modal's stream as the last one arrives.

    Stages read exactly one result per input and stop. Left unfinished, Modal's
    stream is closed later by garbage collection, which its client logs as an
    error. Finishing it before handing over the last result avoids that without
    delaying any result.
    """
    iterator = iter(results)
    for index in range(count):
        item = next(iterator)
        if index == count - 1:
            for _ in iterator:
                pass
        yield item


class ModalSession:
    """One batch's connection to the deployed Modal app."""

    def __init__(self, config: ModalConfig, settings: dict[str, object], *, modal_api=None) -> None:
        self.config = config
        self.settings = settings
        if modal_api is None:
            import modal as modal_api
        self._modal = modal_api
        self._client = modal_api.Client.from_credentials(config.token_id, config.token_secret)
        self._volume = modal_api.Volume.from_name(
            AUDIO_VOLUME, create_if_missing=True, client=self._client
        )
        self._analysis = None
        self._uploaded: set[str] = set()

    def _function(self, name: str):
        return self._modal.Function.from_name(APP_NAME, name, client=self._client)

    def expected(self) -> dict[str, object]:
        from worker.models.download import model_manifest

        return {
            "code": code_identity(),
            "deployment": self.config.deployment,
            "models": model_manifest(bool(self.settings.get("motion_artwork_models"))),
        }

    def inspect(self) -> dict[str, object] | None:
        """The deployment's status, or None when Echora is not deployed in this workspace."""
        try:
            return self._function("status").remote()
        except self._modal.exception.NotFoundError:
            return None

    def ensure_ready(
        self,
        progress: Callable[[dict[str, object]], None] = lambda _: None,
        check: Callable[[], None] = lambda: None,
    ) -> dict[str, object]:
        """Deploy and download whatever is stale, then confirm the deployment matches."""
        expected = self.expected()
        progress({"phase": "modal", "message": "Checking Modal deployment"})
        state = self.inspect()
        deployed = False
        if (
            state is None
            or state["code"] != expected["code"]
            or state["deployment"] != expected["deployment"]
        ):
            check()
            progress({"phase": "modal", "message": "Deploying Echora analysis to Modal"})
            deploy(self.config)
            deployed = True
            # Deploys roll over: an old container may still answer for a short while.
            deadline = time.monotonic() + ROLLOVER_SECONDS
            while True:
                state = self.inspect()
                if (
                    state is not None
                    and state["code"] == expected["code"]
                    and state["deployment"] == expected["deployment"]
                ):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "The Modal deployment does not match this Echora version after deploying"
                    )
                check()
                time.sleep(ROLLOVER_POLL_SECONDS)
        downloaded = False
        if state["models"] != expected["models"]:
            check()
            for event in self._function("prepare_models").remote_gen(
                self.settings, self.config.hf_token
            ):
                check()
                if "label" in event:
                    progress(
                        {
                            "phase": "modal",
                            "message": f"Preparing models on Modal: {event['label']}",
                            "completed": event["completed"],
                            "total": event["total"],
                            "unit": "models",
                        }
                    )
            downloaded = True
            state = self.inspect()
            if state is None or state["models"] != expected["models"]:
                raise RuntimeError("Modal model storage is incomplete after preparing it")
        self._analysis = self._modal.Cls.from_name(APP_NAME, "Analysis", client=self._client)()
        return {
            "deployed": deployed,
            "downloaded": downloaded,
            "code": expected["code"],
            "models": expected["models"],
        }

    @property
    def analysis(self):
        if self._analysis is None:
            raise RuntimeError("Check the Modal deployment before computing")
        return self._analysis

    def upload(self, digest: str, audio: bytes) -> None:
        """Make a track's source audio available to Modal, once per batch."""
        if digest in self._uploaded:
            return
        if hashlib.sha256(audio).hexdigest() != digest:
            raise ValueError("Source audio does not match its content hash")
        with self._volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(audio), f"/{digest}")
        self._uploaded.add(digest)

    def upload_audio(self, audio: bytes) -> str:
        """Upload source audio by its content hash and return the hash."""
        digest = hashlib.sha256(audio).hexdigest()
        self.upload(digest, audio)
        return digest

    def _map(self, method, *inputs: list) -> Iterator[object]:
        """Run one call per input in order; yields each result or the exception it raised."""
        if not inputs or not inputs[0]:
            return iter(())
        return _complete(
            method.map(
                *inputs,
                kwargs={"settings": self.settings},
                order_outputs=True,
                return_exceptions=True,
            ),
            len(inputs[0]),
        )

    def embed_audio(self, model: RemoteModel, digests: list[str]) -> Iterator[object]:
        """Audio window embeddings for uploaded tracks."""
        if not digests:
            return iter(())
        return _complete(
            self.analysis.embed_audio.map(
                digests,
                kwargs={
                    "model": (model.name, model.model_id, model.revision),
                    "settings": self.settings,
                },
                order_outputs=True,
                return_exceptions=True,
            ),
            len(digests),
        )

    def melody(self, digests: list[str]) -> Iterator[object]:
        """Usable melody contours per source, for uploaded tracks."""
        return self._map(self.analysis.melody, digests)

    def melody_one(self, audio: bytes):
        return self.analysis.melody.remote(self.upload_audio(audio), settings=self.settings)

    def prepare_vocals(self, digests: list[str], **artifacts: bool) -> Iterator[object]:
        """A separation phase on Modal: one Roformer separates every listed track.

        `artifacts` are prepare_audio's flags (vocals, reference, melody): exactly what
        the stage that asked will read, so its own calls then hit the container's cache.
        """
        if not digests:
            return iter(())
        return _complete(
            self.analysis.prepare_vocals.map(
                digests,
                kwargs={"artifacts": artifacts, "settings": self.settings},
                order_outputs=True,
                return_exceptions=True,
            ),
            len(digests),
        )

    def voice(self, digests: list[str]) -> Iterator[object]:
        """(activations, vocal activity) for uploaded tracks."""
        return self._map(self.analysis.voice, digests)

    def karaoke(
        self,
        digests: list[str],
        texts: list[str],
        languages: list[str | None],
        source_lines: list[list],
    ) -> Iterator[object]:
        """FA-Kara alignment results for uploaded tracks."""
        return self._map(self.analysis.karaoke, digests, texts, languages, source_lines)

    def embed_lyrics(self, texts: list[str]) -> Iterator[object]:
        """BGE-M3 lyrics embeddings."""
        return self._map(self.analysis.embed_lyrics, texts)

    def transcribe_generate(
        self,
        audio: bytes,
        *,
        language: str | None,
        vocal_activity,
        progress: Callable[[str], None],
        diagnostic_sink: Callable[[dict], None],
        check: Callable[[], None],
    ) -> dict[str, object]:
        """MOSS generation for one track (MOSS stays loaded on Modal between tracks).

        Relays remote progress and diagnostics as they happen; returns the draft that
        SongTranscriber.finish completes.
        """
        digest = self.upload_audio(audio)
        draft = None
        for event in self.analysis.transcribe_generate.remote_gen(
            digest, language, vocal_activity, self.settings
        ):
            check()
            if "progress" in event:
                progress(event["progress"])
            elif "diagnostic" in event:
                diagnostic_sink(event["diagnostic"])
            elif "draft" in event:
                draft = event["draft"]
        if draft is None:
            raise RuntimeError("Modal transcription ended without a result")
        return draft

    def transcribe_finish(self, drafts: list[dict]) -> Iterator[object]:
        """The timing repair phase on Modal: drafts finished with one loaded aligner."""
        return self._map(self.analysis.transcribe_finish, drafts)

    def motion_artwork(self, settings, recipe, covers: list[dict]) -> Iterator[dict]:
        """Render covers' loops on Modal; yields render_loops events with local files.

        Finished videos come back through the transfer Volume: each rendered event names a file
        there, which is downloaded to a local temporary file and removed from Modal.
        """
        import tempfile

        artwork = self._modal.Cls.from_name(APP_NAME, "Artwork", client=self._client)()
        # Modal renders with its own private ComfyUI, never an address configured here.
        remote_settings = settings.model_copy(update={"comfyui_url": ""})
        sent = [
            {key: value for key, value in cover.items() if key in COVER_FIELDS} for cover in covers
        ]
        for event in artwork.render.remote_gen(remote_settings, recipe, sent, self.settings):
            if "rendered" in event:
                transfer = event.pop("transfer")
                with tempfile.NamedTemporaryFile(
                    prefix="echora-loop-", suffix=".mp4", delete=False
                ) as local:
                    for chunk in self._volume.read_file(transfer):
                        local.write(chunk)
                try:
                    self._volume.remove_file(transfer)
                except Exception:
                    logger.warning("Could not remove a transferred loop from Modal")
                event["file"] = local.name
            yield event

    def close(self) -> None:
        """Remove this batch's audio from Modal."""
        for digest in self._uploaded:
            try:
                self._volume.remove_file(f"/{digest}")
            except Exception:
                logger.warning("Could not remove uploaded audio %s from Modal", digest[:12])
        self._uploaded.clear()
