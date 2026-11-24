"""Provision a verified recording bundle through Echora's model downloader.

No network access during serving or --prune-only. Source selection is opt-in:
ECHORA_RECORDING_MODEL_ID and ECHORA_RECORDING_MODEL_REVISION must be set together.
ECHORA_RECORDING_MODEL_DIRECTORY defaults to /data/models/recording. Bundles live
under <directory>/<full revision>, leaving other generations untouched.
"""

from collections.abc import Mapping
import errno
import json
from pathlib import Path
import re
import shutil
import tempfile

from huggingface_hub import snapshot_download

from worker.recordings.encoder import load_config
from worker.recordings.search import load_match_policy

BUNDLE_FILES = (
    "encoder.onnx",
    "manifest.json",
    "parity.json",
    "match-policy.json",
    "UPSTREAM-LICENSE",
    "README.md",
)


def validate_bundle(directory: Path) -> None:
    """Reject incomplete, external, corrupt or incompatible bundle contents."""
    root = directory.resolve()
    for name in BUNDLE_FILES:
        path = directory / name
        if not path.is_file() or path.resolve().parent != root:
            raise ValueError(f"Missing or external recording bundle file: {name}")
    # Check the manifest paths before loading any referenced bytes.
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("model_path") != "encoder.onnx"
        or manifest.get("validation_report_path") != "parity.json"
    ):
        raise ValueError("Recording bundle must use its own model and parity files")
    config = load_config(directory / "manifest.json")
    load_match_policy(config, directory / "match-policy.json")


def download_recording_model(environ: Mapping[str, str] | None = None) -> Path | None:
    from core.settings import get_settings

    settings = get_settings() if environ is None else None
    env = environ
    model_id = (
        settings.recording_model_id if settings else env.get("ECHORA_RECORDING_MODEL_ID", "")
    ).strip()
    revision = (
        settings.recording_model_revision
        if settings
        else env.get("ECHORA_RECORDING_MODEL_REVISION", "")
    ).strip()
    if not model_id and not revision:
        return None
    if not model_id or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "Set ECHORA_RECORDING_MODEL_ID and a full immutable ECHORA_RECORDING_MODEL_REVISION together"
        )
    root = Path(
        settings.recording_model_directory
        if settings
        else env.get("ECHORA_RECORDING_MODEL_DIRECTORY", "/data/models/recording")
    )
    if not root.is_absolute():
        raise ValueError("ECHORA_RECORDING_MODEL_DIRECTORY must be an absolute mounted path")
    destination = root / revision
    if destination.exists():
        validate_bundle(destination)
        print("Verified cached recording model and matcher policy.", flush=True)
        return destination

    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".recording-download-", dir=root))
    try:
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=str(staging),
            allow_patterns=list(BUNDLE_FILES),
        )
        validate_bundle(staging)
        # The download init container may run as root; serving runs as UID 1001.
        staging.chmod(0o755)
        for name in BUNDLE_FILES:
            (staging / name).chmod(0o644)
        try:
            staging.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # A concurrent provisioner may have published this same generation.
            # Never overwrite it; validate its completed bundle instead.
            validate_bundle(destination)
        print("Installed verified recording model and matcher policy.", flush=True)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)
