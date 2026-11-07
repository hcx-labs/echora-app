"""Echora analysis on Modal (see docs/modal-compute.md).

The image is Echora's published GPU analysis image with the deploying worker's
own analysis source on top, so remote code is exactly the worker's code. Model
weights are not in the image: `prepare_models` fills a persistent Volume, and
a manifest file on it records which pinned model set is present.

Echora deploys this app itself from Settings → External processing. To deploy
by hand, from services/analysis with a Modal token configured:

    ECHORA_MODAL_IMAGE=ghcr.io/hackslashx/echora-analysis-gpu:<version> \\
        PYTHONPATH=src modal deploy -m echora_analysis.modal_app

Every compute call carries the worker's processing settings (model revisions,
aligner options), and the compute functions are the ones the worker runs
locally. Results therefore match local processing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from echora_analysis.modal_constants import APP_NAME, AUDIO_VOLUME, GPU_TYPES, MODELS_VOLUME

# The image keeps an empty /models, which Modal cannot mount over.
MODELS = "/echora-models"
AUDIO = "/echora-audio"
MANIFEST = f"{MODELS}/.echora-model-manifest"

PACKAGE = Path(__file__).resolve().parent
SOURCE = PACKAGE.parent
VENDOR = PACKAGE.parents[1] / "vendor"

IMAGE = os.environ.get("ECHORA_MODAL_IMAGE") or "ghcr.io/hackslashx/echora-analysis-gpu:latest"
GPU = os.environ.get("ECHORA_MODAL_GPU") or "L40S"
if GPU not in GPU_TYPES:
    raise ValueError(f"Unsupported Modal GPU type: {GPU}")
DEPLOYMENT = json.dumps({"image": IMAGE, "gpu": GPU}, sort_keys=True)

image = (
    modal.Image.from_registry(IMAGE)
    .env(
        {
            "HF_HOME": f"{MODELS}/huggingface",
            "TORCH_HOME": f"{MODELS}/torch",
            "HF_HUB_OFFLINE": "0",
            # Some pinned revisions fail over Xet (404 on the read token); plain HTTP works.
            "HF_HUB_DISABLE_XET": "1",
            "ESSENTIA_MODELS_DIR": f"{MODELS}/essentia",
            "ECHORA_PREPROCESS_DIR": "/tmp/echora-preprocessed",
            "ECHORA_MODAL_DEPLOYMENT": DEPLOYMENT,
            # A fresh path, not the image's /service/src, so no file of the image's
            # own source version can mix with the worker's.
            "PYTHONPATH": "/echora/src",
        }
    )
    .add_local_dir(SOURCE, "/echora/src", ignore=["**/__pycache__/**"])
    .add_local_dir(VENDOR, "/echora/vendor", ignore=["**/__pycache__/**"])
)

models_volume = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=True)
# Source audio of running batches, uploaded by the worker and named by SHA-256.
audio_volume = modal.Volume.from_name(AUDIO_VOLUME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _manifest() -> str | None:
    try:
        return Path(MANIFEST).read_text().strip() or None
    except FileNotFoundError:
        return None


@app.function(cpu=1, volumes={MODELS: models_volume}, timeout=120)
def status() -> dict[str, object]:
    """What this deployment runs: code digest, image and GPU, and the model set on the Volume."""
    from echora_analysis.remote_compute import code_identity

    models_volume.reload()
    return {
        "code": code_identity(),
        "deployment": json.loads(os.environ["ECHORA_MODAL_DEPLOYMENT"]),
        "models": _manifest(),
    }


@app.function(cpu=4, memory=8192, volumes={MODELS: models_volume}, timeout=3 * 60 * 60)
def prepare_models(settings: dict[str, object], hf_token: str | None = None):
    """Download the models the worker's settings require, reporting each step.

    Runs on a CPU container so downloads do not bill GPU time. Present files
    are skipped, so an update downloads only what changed. The Hugging Face
    token is used for these downloads only and is not kept on Modal.
    """
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    from echora_analysis.settings import adopt_processing_settings

    adopt_processing_settings(settings)
    from echora_analysis.download_models import analysis_downloads, model_manifest

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    try:
        steps = analysis_downloads()
        for index, (label, download) in enumerate(steps):
            yield {"completed": index, "total": len(steps), "label": label}
            try:
                download()
            except (GatedRepoError, RepositoryNotFoundError):
                raise RuntimeError(
                    f"Hugging Face refused {label}. Accept the model's license on Hugging Face "
                    "with the account of the token in External processing, then prepare again."
                ) from None
            models_volume.commit()
    finally:
        os.environ.pop("HF_TOKEN", None)
    manifest = model_manifest()
    Path(MANIFEST).write_text(manifest)
    models_volume.commit()
    yield {"completed": len(steps), "total": len(steps), "manifest": manifest}


@app.cls(
    gpu=GPU,
    volumes={MODELS: models_volume, AUDIO: audio_volume},
    timeout=60 * 60,
    scaledown_window=300,
    max_containers=1,
)
class Analysis:
    @modal.enter()
    def start(self) -> None:
        self.settings: dict[str, object] | None = None
        # One model family stays on the GPU at a time, as in local processing.
        self.resident: tuple[str, object] | None = None

    def _adopt(self, settings: dict[str, object]) -> None:
        """Use the calling worker's processing settings for this call."""
        if settings == self.settings:
            return
        from echora_analysis.download_models import model_manifest
        from echora_analysis.settings import adopt_processing_settings

        self._release()
        adopt_processing_settings(settings)
        models_volume.reload()
        if _manifest() != model_manifest():
            raise RuntimeError(
                "Modal model storage does not match these settings; prepare Modal again"
            )
        self.settings = dict(settings)

    def _release(self) -> None:
        if self.resident is None:
            return
        kind, model = self.resident
        self.resident = None
        if kind == "fa_kara":
            from echora_analysis.karaoke_pipeline import _stop_fa_kara_worker

            _stop_fa_kara_worker()
        elif kind in {"separator", "moss"}:
            model.close()
        else:
            from echora_analysis.models import release_model

            release_model(model)

    def _resident(self, kind: str, load):
        if self.resident is None or self.resident[0] != kind:
            self._release()
            self.resident = (kind, load())
        return self.resident[1]

    def _audio(self, digest: str) -> bytes:
        """Read uploaded source audio, refreshing the Volume view if it is new."""
        path = Path(AUDIO) / digest
        if not path.exists():
            audio_volume.reload()
        return path.read_bytes()

    @modal.method()
    def embed_audio(self, digest: str, model: tuple[str, str, str], settings: dict[str, object]):
        """Window embeddings for one track, by the local code path."""
        from echora_analysis.ingest import embed_track
        from echora_analysis.models import MertModel, MuQMuLanModel

        self._adopt(settings)
        name, model_id, revision = model
        model_class = {MuQMuLanModel.name: MuQMuLanModel, MertModel.name: MertModel}[name]
        loaded = self._resident(
            f"audio:{name}:{model_id}:{revision}", lambda: model_class(model_id, revision, "cuda")
        )
        return embed_track(self._audio(digest), loaded)

    def _separator(self):
        """The resident Roformer, so a run of separations loads it once."""
        from echora_analysis.roformer import Separator

        return self._resident("separator", Separator)

    @modal.method()
    def prepare_vocals(
        self, digest: str, artifacts: dict[str, bool], settings: dict[str, object]
    ) -> None:
        """One track of a separation phase: prepare what the asking stage reads into the cache."""
        from echora_analysis.preprocessing import prepare_audio
        from echora_analysis.roformer import using

        self._adopt(settings)
        with using(self._separator()):
            prepare_audio(self._audio(digest), **artifacts)

    @modal.method()
    def melody(self, digest: str, settings: dict[str, object]):
        """Melody contours for one track (cached Roformer stems, then Melodia)."""
        from echora_analysis.hum_search import track_contours
        from echora_analysis.roformer import using

        self._adopt(settings)
        with using(self._separator()):
            return track_contours(self._audio(digest))

    def _transcriber(self):
        from echora_analysis.song_transcription import SongTranscriber
        from echora_analysis.transcription_config import transcription_model

        config = transcription_model()
        if config is None:
            raise RuntimeError("AI lyric generation has no MOSS model configured")
        return SongTranscriber(*config)

    @modal.method()
    def transcribe_generate(
        self, digest: str, language: str | None, vocal_activity, settings: dict[str, object]
    ):
        """MOSS generation for one track with the resident MOSS; streams progress, then the draft."""
        import queue
        import threading

        self._adopt(settings)

        def load():
            transcriber = self._transcriber()
            transcriber.open()
            return transcriber

        transcriber = self._resident("moss", load)
        events: queue.Queue = queue.Queue()
        outcome: dict[str, object] = {}

        def run() -> None:
            try:
                outcome["draft"] = transcriber.generate(
                    self._audio(digest),
                    language=language,
                    progress=lambda detail: events.put({"progress": detail}),
                    diagnostic_sink=lambda attempt: events.put({"diagnostic": attempt}),
                    vocal_activity=vocal_activity,
                )
            except BaseException as error:  # Re-raised in the caller's stream.
                outcome["error"] = error
            finally:
                events.put(None)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        while (event := events.get()) is not None:
            yield event
        thread.join()
        if "error" in outcome:
            raise outcome["error"]
        yield {"draft": outcome["draft"]}

    @modal.method()
    def transcribe_finish(self, draft: dict, settings: dict[str, object]):
        """Timing repair for one draft, with the resident karaoke aligner."""
        self._adopt(settings)
        self._resident("fa_kara", lambda: None)
        return self._transcriber().finish(draft)

    @modal.method()
    def embed_lyrics(self, text: str, settings: dict[str, object]):
        """BGE-M3 lyrics embedding, by the local model class."""
        from echora_analysis.lyrics_analysis import LyricsEmbeddingModel
        from echora_analysis.settings import get_settings

        self._adopt(settings)
        model = self._resident(
            "lyrics",
            lambda: LyricsEmbeddingModel(
                get_settings().lyrics_model_id, get_settings().lyrics_revision, "cuda"
            ),
        )
        return model.embed(text)

    @modal.method()
    def karaoke(
        self,
        digest: str,
        text: str,
        language: str | None,
        source_lines,
        settings: dict[str, object],
    ):
        """FA-Kara alignment of one track's lyrics, on the audio FA_KARA_AUDIO_INPUT names."""
        from echora_analysis.karaoke_pipeline import _run_fa_kara

        self._adopt(settings)
        # The aligner runs in a resident worker process; other models leave the GPU first.
        self._resident("fa_kara", lambda: None)
        return _run_fa_kara(self._audio(digest), text, language, source_lines or [])

    @modal.method()
    def voice(self, digest: str, settings: dict[str, object]):
        """Voice and vocal-gender evidence for one track."""
        from echora_analysis.voice_pipeline import classify_track

        self._adopt(settings)
        return classify_track(self._audio(digest))

    @modal.method()
    def health(self) -> dict[str, object]:
        """Report the GPU this deployment runs on."""
        import torch

        return {
            "deployment": json.loads(os.environ["ECHORA_MODAL_DEPLOYMENT"]),
            "torch": str(torch.__version__),
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "models": _manifest(),
        }


@app.cls(
    gpu=GPU,
    volumes={MODELS: models_volume, AUDIO: audio_volume},
    timeout=12 * 60 * 60,
    scaledown_window=120,
    max_containers=1,
)
class Artwork:
    """Motion artwork loops: Echora's own render_loops with a private ComfyUI on this GPU."""

    @modal.method()
    def render(self, artwork_settings, recipe, covers: list[dict], settings: dict[str, object]):
        """Yield render_loops events; each finished video is handed over through the transfer Volume."""
        import shutil
        import uuid

        from echora_analysis.download_models import model_manifest
        from echora_analysis.motion_artwork_jobs import render_loops
        from echora_analysis.settings import adopt_processing_settings

        adopt_processing_settings(settings)
        models_volume.reload()
        if _manifest() != model_manifest():
            raise RuntimeError(
                "Modal model storage does not match these settings; prepare Modal again"
            )
        transfers = Path(AUDIO) / "artwork"
        transfers.mkdir(parents=True, exist_ok=True)
        for event in render_loops(artwork_settings, recipe, covers):
            if "rendered" in event:
                name = f"artwork/{event['rendered']}-{uuid.uuid4().hex}.mp4"
                shutil.copyfile(event.pop("file"), Path(AUDIO) / name)
                audio_volume.commit()
                event["transfer"] = f"/{name}"
            yield event
