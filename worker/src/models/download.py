from core.db.config import DatabaseConfig
from core.media.config import MotionArtworkConfig
from worker.kara.config import KaraConfig
from worker.models.config import ModelsConfig
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

from huggingface_hub import hf_hub_download, scan_cache_dir, snapshot_download

from worker.media.roformer import MODEL_ID as ROFORMER_MODEL_ID, MODEL_REVISION as ROFORMER_REVISION
from worker.recordings.download import download_recording_model
from worker.artwork.render import MODEL_FILES as MOTION_ARTWORK_FILES

ESSENTIA_MODELS = (
    (
        "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx",
        "a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c",
    ),
    (
        "https://essentia.upf.edu/models/classification-heads/gender/gender-discogs-effnet-1.onnx",
        "e3e865d4bf36d4817f32ddab9452b2729f9e33a4d068d1c44ea44972a7999e91",
    ),
    (
        "https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.onnx",
        "20155e4c439714b0c45c08644b73c8e12d9dccb173bd4ab9934bf1e5aee837ca",
    ),
)

MODELS = (
    ("OpenMuQ/MuQ-MuLan-large", "2e01c796b71dca71b45251384c04cd7b237c9020", False),
    ("OpenMuQ/MuQ-large-msd-iter", "0562a57814f6f8bbd9fdea0a25921a2fce1a841a", True),
    ("xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089", True),
    ("m-a-p/MERT-v1-95M", "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5", False),
    ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", False),
)

# The former FA-Kara base snapshot is deliberately managed so the downloader
# can reclaim it after the serving revision moves to Echora's published model.
LEGACY_MANAGED_MODELS = {"NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn"}


def required_models() -> tuple[tuple[str, str, bool], ...]:
    """Return immutable model revisions required by the active service config."""
    fa_model = KaraConfig.load().model_id
    fa_revision = KaraConfig.load().revision
    from worker.transcription.config import transcription_model

    moss = transcription_model()
    return (
        *MODELS,
        (fa_model, fa_revision, False),
        (ROFORMER_MODEL_ID, ROFORMER_REVISION, False),
        *(((*moss, False),) if moss else ()),
    )


def model_manifest(motion_artwork: bool | None = None) -> str:
    """Digest of every pinned model analysis needs; a change means storage must be updated.

    Motion artwork's files count only when its models are wanted (ECHORA_MOTION_ARTWORK_MODELS,
    or for Modal, its motion artwork switch).
    """
    if motion_artwork is None:
        motion_artwork = MotionArtworkConfig.load().models
    content = {
        "snapshots": [list(item) for item in required_models()],
        "essentia": [list(item) for item in ESSENTIA_MODELS],
    }
    if motion_artwork:
        content["motion_artwork"] = [list(item) for item in MOTION_ARTWORK_FILES]
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def analysis_downloads() -> list[tuple[str, object]]:
    """Ordered download steps for the models analysis needs: (label, callable)."""
    motion_artwork = MotionArtworkConfig.load().models
    steps: list[tuple[str, object]] = []
    for model, revision, needs_main_ref in required_models():

        def fetch(model=model, revision=revision, needs_main_ref=needs_main_ref):
            snapshot = snapshot_download(repo_id=model, revision=revision)
            if needs_main_ref:
                _pin_main_ref(snapshot)

        steps.append((f"{model}@{revision[:8]}", fetch))
    if motion_artwork:
        # Only the files ComfyUI loads, never whole repositories. LTX-2.5 is gated.
        for repo, revision, filename in MOTION_ARTWORK_FILES:
            steps.append(
                (
                    f"{repo}/{filename.rsplit('/', 1)[-1]}",
                    lambda repo=repo, revision=revision, filename=filename: hf_hub_download(
                        repo_id=repo, filename=filename, revision=revision
                    ),
                )
            )
    steps.append(("Essentia voice classifiers", _download_essentia))
    keep = required_models() + motion_artwork_snapshots(motion_artwork)
    steps.append(("Removing superseded models", lambda: _prune_huggingface_cache(keep)))
    return steps


def _pin_main_ref(snapshot_path: str) -> None:
    snapshot = Path(snapshot_path)
    refs = snapshot.parent.parent / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(snapshot.name, encoding="utf-8")


def _prune_huggingface_cache(required: tuple[tuple[str, str, bool], ...]) -> None:
    """Remove superseded managed snapshots without touching referenced blobs.

    ``delete_revisions`` is Hugging Face's cache-aware deletion API. It removes
    a snapshot and only then removes blobs no remaining snapshot references.
    Repositories outside Echora's managed model list are left alone.
    """
    cache_dir = Path(os.environ.get("HF_HOME", "/models/huggingface")) / "hub"
    if not cache_dir.is_dir():
        return
    keep = {model: revision for model, revision, _ in required}
    managed = set(keep) | LEGACY_MANAGED_MODELS
    cache = scan_cache_dir(cache_dir)
    stale = [
        revision.commit_hash
        for repo in cache.repos
        if repo.repo_id in managed
        for revision in repo.revisions
        if keep.get(repo.repo_id) != revision.commit_hash
    ]
    if not stale:
        print("Hugging Face cache contains only required managed snapshots.", flush=True)
        return
    strategy = cache.delete_revisions(*stale)
    strategy.execute()
    print(f"Pruned {len(stale)} superseded Hugging Face model snapshot(s).", flush=True)


def _download_essentia() -> None:
    """Fetch the MTG-Jamendo voice/gender classifier files used by voice_pipeline."""
    directory = Path(ModelsConfig.load().essentia_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for url, expected_sha256 in ESSENTIA_MODELS:
        target = directory / url.rsplit("/", 1)[-1]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256:
            print(f"essentia model {target.name} is available.", flush=True)
            continue
        print(f"Downloading {url} into {directory}", flush=True)
        partial = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        actual_sha256 = hashlib.sha256(partial.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            partial.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {target.name}: {actual_sha256}")
        partial.replace(target)
        print(f"essentia model {target.name} is available.", flush=True)


def motion_artwork_snapshots(enabled: bool) -> tuple[tuple[str, str, bool], ...]:
    """Pinned motion artwork repositories, managed (kept and pruned) only once opted in."""
    if not enabled:
        return ()
    return tuple(
        dict.fromkeys((repo, revision, False) for repo, revision, _ in MOTION_ARTWORK_FILES)
    )


def _download_motion_artwork() -> None:
    """Fetch only the LTX-2.5 and Gemma files ComfyUI loads, never whole repositories.

    Lightricks/LTX-2.5 is gated: accept its license on Hugging Face and provide HF_TOKEN.
    """
    for repo, revision, filename in MOTION_ARTWORK_FILES:
        print(f"Downloading {repo}@{revision}/{filename}", flush=True)
        hf_hub_download(repo_id=repo, filename=filename, revision=revision)


def main(*, prune_only: bool = False, motion_artwork: bool = False) -> None:
    if not prune_only and not os.environ.get("HF_TOKEN") and DatabaseConfig.load().database_url:
        # The token saved in Settings → Analysis models, for gated models such as LTX-2.5.
        from worker.models.huggingface_token import token_or_none

        token = token_or_none()
        if token:
            os.environ["HF_TOKEN"] = token
    motion_artwork = motion_artwork or MotionArtworkConfig.load().models
    required = required_models() + motion_artwork_snapshots(motion_artwork)
    if not prune_only:
        download_recording_model()
        if motion_artwork:
            _download_motion_artwork()
        for model, revision, needs_main_ref in required_models():
            print(f"Downloading {model}@{revision}", flush=True)
            snapshot = snapshot_download(repo_id=model, revision=revision)
            if needs_main_ref:
                # MuQ constructs these dependencies without forwarding a revision.
                # Point their local `main` refs at the reviewed commits so offline
                # inference remains pinned.
                _pin_main_ref(snapshot)
    _prune_huggingface_cache(required)
    if not prune_only:
        _download_essentia()
        print("All model snapshots are available.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prune-only",
        action="store_true",
        help="Delete superseded managed snapshots without network downloads.",
    )
    parser.add_argument(
        "--motion-artwork",
        action="store_true",
        help="Also download the motion artwork models (about 44 GB; needs HF_TOKEN).",
    )
    arguments = parser.parse_args()
    main(prune_only=arguments.prune_only, motion_artwork=arguments.motion_artwork)
