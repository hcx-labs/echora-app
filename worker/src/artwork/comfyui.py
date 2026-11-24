"""ComfyUI for motion artwork: a minimal HTTP client and an on-demand launcher.

The analysis worker normally starts ComfyUI itself, only while a batch renders loops, and
stops it afterwards so its GPU and system memory return to analysis. Graphs are submitted in
API format and outputs are read back through /history and /view.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx


class ComfyUIError(RuntimeError):
    """ComfyUI rejected a graph, failed while running it, or is unreachable."""


class ComfyUIUnavailable(ComfyUIError):
    """ComfyUI is not installed here, could not start, or cannot be reached."""


def model_paths_config() -> str:
    """ComfyUI model paths for the pinned motion artwork snapshots in the Hugging Face cache.

    Derived from HF_HOME at launch, so the same launcher finds the models wherever the cache
    lives: /models on the analysis worker, a Modal Volume on Modal.
    """
    import os

    from worker.artwork.render import CAMERA_LORA_REPOSITORY, CAMERA_LORA_REVISION, GEMMA_REPOSITORY, GEMMA_REVISION, LTX_REPOSITORY, LTX_REVISION

    hub = Path(os.environ.get("HF_HOME", "/models/huggingface")) / "hub"

    def snapshot(repository: str, revision: str) -> Path:
        return hub / f"models--{repository.replace('/', '--')}" / "snapshots" / revision

    return (
        f"echora_ltx:\n  base_path: {snapshot(LTX_REPOSITORY, LTX_REVISION)}\n"
        "  diffusion_models: diffusion_models\n  text_encoders: text_encoders\n  vae: vae\n"
        "  latent_upscale_models: latent_upscale_models\n"
        f"echora_camera_lora:\n  base_path: {snapshot(CAMERA_LORA_REPOSITORY, CAMERA_LORA_REVISION)}\n"
        "  loras: .\n"
        f"echora_gemma:\n  base_path: {snapshot(GEMMA_REPOSITORY, GEMMA_REVISION)}\n"
        "  text_encoders: text_encoders\n"
    )


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextmanager
def launch(
    comfyui_dir: str,
    python: str,
    *,
    startup_timeout_seconds: float,
    check: Callable[[], None],
    work_dir: Path | None = None,
) -> Iterator[str]:
    """Run a private ComfyUI on a free local port and yield its URL; stop it on exit.

    Inputs, outputs and ComfyUI's own state live in `work_dir` (a throwaway directory when not
    given), so several short-lived ComfyUI processes in one batch share uploaded covers and saved
    encodings. Models come from the Hugging Face cache (model_paths_config); embeddings from
    `output/conditioning`.
    """
    root_dir = Path(comfyui_dir)
    if not (root_dir / "main.py").is_file() or not Path(python).is_file():
        raise ComfyUIUnavailable("ComfyUI is not installed in this image")
    with tempfile.TemporaryDirectory(prefix="echora-comfyui-") as directory:
        work = Path(work_dir) if work_dir is not None else Path(directory)
        for name in ("input", "output/conditioning", "temp", "user"):
            (work / name).mkdir(parents=True, exist_ok=True)
        batch_paths = work / "batch_model_paths.yaml"
        batch_paths.write_text(
            f"echora_batch:\n  base_path: {work / 'output'}\n  embeddings: conditioning\n"
        )
        model_paths = work / "echora_model_paths.yaml"
        model_paths.write_text(model_paths_config())
        port = _free_port()
        command = [
            python,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--disable-auto-launch",
            "--extra-model-paths-config",
            str(model_paths),
            str(batch_paths),
            "--reserve-vram",
            "1",
            "--input-directory",
            str(work / "input"),
            "--output-directory",
            str(work / "output"),
            "--temp-directory",
            str(work / "temp"),
            "--user-directory",
            str(work / "user"),
            "--database-url",
            "sqlite:///:memory:",
            "--disable-metadata",
        ]
        with (work / "comfyui.log").open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=root_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            url = f"http://127.0.0.1:{port}"
            try:
                _wait_until_ready(process, url, startup_timeout_seconds, check)
                try:
                    yield url
                except httpx.TransportError as error:
                    # A request failed because ComfyUI itself is gone: say why it stopped.
                    if process.poll() is None:
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            raise error from None
                    raise ComfyUIUnavailable(
                        f"ComfyUI stopped while running (exit code {process.returncode}). "
                        f"Last log lines: {_log_tail(work / 'comfyui.log')}"
                    ) from error
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _log_tail(path: Path, lines: int = 12, limit: int = 1500) -> str:
    """The end of ComfyUI's log, for errors that would otherwise only say "connection refused"."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return "(no log)"
    return " | ".join(line.strip() for line in text.splitlines()[-lines:] if line.strip())[-limit:]


def _wait_until_ready(
    process: subprocess.Popen, url: str, timeout_seconds: float, check: Callable[[], None]
) -> None:
    deadline = time.monotonic() + timeout_seconds
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < deadline:
            check()
            if process.poll() is not None:
                raise ComfyUIUnavailable(
                    f"ComfyUI exited during startup (code {process.returncode})"
                )
            try:
                if client.get(f"{url}/system_stats").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
    raise ComfyUIUnavailable("ComfyUI did not start in time")


class ComfyUI:
    def __init__(self, base_url: str, timeout_seconds: float = 30):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_seconds, follow_redirects=False, trust_env=False)
        self.client_id = str(uuid.uuid4())

    def __enter__(self) -> "ComfyUI":
        return self

    def __exit__(self, *_) -> None:
        self.client.close()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def system_stats(self) -> dict:
        response = self.client.get(self._url("/system_stats"))
        response.raise_for_status()
        return response.json()

    def available(self, node: str, field: str) -> list[str]:
        """Model files ComfyUI can load for one loader input, such as UNETLoader.unet_name."""
        response = self.client.get(self._url(f"/object_info/{node}"))
        response.raise_for_status()
        spec = response.json().get(node, {}).get("input", {})
        for group in ("required", "optional"):
            entry = spec.get(group, {}).get(field)
            if entry and isinstance(entry[0], list):
                return [str(name) for name in entry[0]]
        return []

    def upload_image(self, data: bytes, filename: str, content_type: str = "image/jpeg") -> str:
        response = self.client.post(
            self._url("/upload/image"),
            files={"image": (filename, data, content_type)},
            data={"overwrite": "true"},
        )
        response.raise_for_status()
        return str(response.json()["name"])

    def queue(self, graph: dict) -> str:
        response = self.client.post(
            self._url("/prompt"), json={"prompt": graph, "client_id": self.client_id}
        )
        if response.status_code == 400:
            # ComfyUI explains invalid graphs (missing model files, bad inputs) in the body.
            detail = response.json().get("error", {}).get("message", "invalid graph")
            raise ComfyUIError(f"ComfyUI rejected the graph: {detail}")
        response.raise_for_status()
        return str(response.json()["prompt_id"])

    def wait(
        self,
        prompt_id: str,
        *,
        check: Callable[[], None],
        timeout_seconds: float,
        poll_seconds: float,
    ) -> dict:
        """Poll until the graph finishes. `check` raises to cancel; the render is then stopped."""
        deadline = time.monotonic() + timeout_seconds
        try:
            while time.monotonic() < deadline:
                check()
                response = self.client.get(self._url(f"/history/{prompt_id}"))
                response.raise_for_status()
                job = response.json().get(prompt_id)
                if job:
                    status = job.get("status", {})
                    if status.get("status_str") == "error":
                        raise ComfyUIError(_error_message(status))
                    if status.get("completed") or job.get("outputs"):
                        return job.get("outputs", {})
                time.sleep(poll_seconds)
        except BaseException:
            self.cancel(prompt_id)
            raise
        self.cancel(prompt_id)
        raise ComfyUIError("ComfyUI did not finish the render in time")

    def cancel(self, prompt_id: str) -> None:
        """Remove a queued graph or interrupt it if it is running. Best effort."""
        try:
            self.client.post(self._url("/queue"), json={"delete": [prompt_id]})
            self.client.post(self._url("/interrupt"), json={"prompt_id": prompt_id})
        except httpx.HTTPError:
            pass

    def download(self, item: dict, target: Path) -> Path:
        params = {
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        with self.client.stream("GET", self._url("/view"), params=params) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
        partial.replace(target)
        return target

    def free(self) -> None:
        """Unload models and release VRAM so other GPU work can run."""
        self.client.post(self._url("/free"), json={"unload_models": True, "free_memory": True})


def _error_message(status: dict) -> str:
    for kind, detail in reversed(status.get("messages", [])):
        if kind == "execution_error" and isinstance(detail, dict):
            node = detail.get("node_type", "a node")
            message = str(detail.get("exception_message", "")).strip().splitlines()
            return f"ComfyUI failed in {node}: {message[0] if message else 'unknown error'}"
    return "ComfyUI failed while rendering"


def output_items(outputs: dict, key: str) -> list[dict]:
    """Collect output entries of one kind (images, text) across all output nodes."""
    return [item for node in outputs.values() for item in node.get(key, []) or []]
