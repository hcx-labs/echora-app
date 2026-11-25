#!/usr/bin/env python3
"""Export the pinned NMFP checkpoint with a waveform frontend and measured parity.

Run in scripts/recording-export-requirements.txt's isolated Python 3.11 environment.
The worker needs only ONNX Runtime. Source and checkpoint must be local and pinned;
this command neither downloads nor executes upstream setup/download scripts.
A successful export is NOT calibration of recording identification thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import zipfile

SOURCE_COMMIT = "15c6f3bcdf6a6da1daddfe47a1ffa5a0d22deadc"
CHECKPOINT_RECORD = "https://zenodo.org/records/15719945"
CHECKPOINT_ARCHIVE_MD5 = "ee8a3358fc5e5cdd09d6d2245d395021"
EXPORT_REVISION = "echora-nmfp-waveform-onnx-v1"
# Fixed before validation, never inferred from the observed errors.
FRONTEND_MAX_ERROR = 0.005
EMBEDDING_MAX_ERROR = 0.0005
EMBEDDING_MIN_COSINE = 0.99999


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_checkpoint_archive(archive: Path, checkpoint: Path, files: list[Path]) -> str:
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != CHECKPOINT_ARCHIVE_MD5:
            raise ValueError("Checkpoint archive does not match the published checksum")
    with zipfile.ZipFile(archive) as bundle:
        for path in files:
            relative = path.relative_to(checkpoint).as_posix()
            names = [
                name
                for name in bundle.namelist()
                if name == relative or name.endswith("/" + relative)
            ]
            if len(names) != 1:
                raise ValueError("Checkpoint file is missing or ambiguous in the verified archive")
            with bundle.open(names[0]) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != sha256(path):
                raise ValueError("Local checkpoint file differs from the verified archive")
    return sha256(archive)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load pinned reference implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_source(source: Path):
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError("Reference source must be the clean pinned commit")


def frontend(reference, tf, np):
    """Build equivalent operators, not Python callbacks, for self-contained ONNX.

    Probe the pinned reference's linear mel operator and Hann window, retaining its
    float32 coefficients. Magnitude DFT uses constant real/imaginary matrices to
    avoid runtime-specific FFT/complex operators. Both ends use zero padding;
    Essentia emits 33 centers, including the final center at sample 8192.
    """
    # Keep coefficients as NumPy until tracing. Captured eager Tensor constants
    # otherwise become extra public inputs in tf2onnx's concrete function.
    window = reference.window(np.ones(1024, dtype=np.float32))
    mel_basis = np.stack([reference.mb(row) for row in np.eye(513, dtype=np.float32)])
    phase = 2 * np.pi * np.outer(np.arange(1024), np.arange(513)) / 1024
    real = np.cos(phase).astype(np.float32)
    imaginary = -np.sin(phase).astype(np.float32)
    frames = np.arange(33)[:, None] * 256 + np.arange(1024)[None, :]

    @tf.function(input_signature=[tf.TensorSpec([None, 8000], tf.float32, name="waveform")])
    def compute(waveform):
        framed = tf.gather(tf.pad(waveform, [[0, 0], [512, 704]]), frames, axis=1) * window
        spectrum = tf.sqrt(
            tf.square(tf.matmul(framed, real)) + tf.square(tf.matmul(framed, imaginary))
        )
        mel = tf.maximum(tf.matmul(spectrum, mel_basis), 1e-5)
        relative = mel / tf.reduce_max(mel, axis=[1, 2], keepdims=True)
        decibels = 20 * tf.math.log(relative) / tf.math.log(tf.constant(10.0))
        return tf.transpose(1 + tf.maximum(decibels, -80.0) / 40.0, [0, 2, 1])

    return compute


def decode_fixture(path: Path, np):
    if not path.is_file():
        raise ValueError("Parity audio must be a local file")
    process = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-t",
            "12",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )
    samples = np.frombuffer(process.stdout, dtype="<f4").copy()
    if len(samples) < 64000 or not np.isfinite(samples).all():
        raise ValueError(
            "Each parity recording needs at least eight seconds of finite decoded audio"
        )
    if np.sqrt(np.mean(samples.astype(np.float64) ** 2)) < 1e-5:
        raise ValueError("Parity recording is silent or too quiet")
    return samples


def windows(samples, np):
    return np.stack([samples[i : i + 8000] for i in range(0, len(samples) - 7999, 4000)])


def metric(reference, candidate, np):
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    if reference.shape != candidate.shape or not np.isfinite(candidate).all():
        raise ValueError("Export returned invalid embeddings")
    norms = np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
    return {
        "max_absolute_error": float(np.max(np.abs(reference - candidate))),
        "min_cosine": float(
            np.min(np.sum(reference * candidate, axis=1) / np.maximum(norms, 1e-12))
        ),
    }


def synthetic_fixtures(np):
    rng = np.random.default_rng(20260921)
    t = np.arange(8000) / 8000
    impulse = np.zeros(8000)
    impulse[4000] = 1
    return np.stack(
        [
            np.zeros(8000),
            np.ones(8000),
            impulse,
            np.linspace(-1, 1, 8000),
            np.sin(2 * np.pi * 440 * t),
            np.sin(2 * np.pi * 1000 * t),
            rng.normal(0, 0.1, 8000),
            rng.normal(0, 1e-6, 8000),
        ]
    ).astype(np.float32)


def validate(reference_frontend, graph_frontend, model, session, audio_paths, np):
    reference_batches = []
    converted_batches = []
    fixtures = []
    retrieval = []
    rng = np.random.default_rng(20260921)

    def compare(label, batch):
        start = time.perf_counter()
        mel = reference_frontend.compute_batch(batch)
        our_mel = graph_frontend(batch).numpy()
        mel_error = float(np.max(np.abs(mel - our_mel)))
        reference_embedding = model(mel[..., None], training=False).numpy()
        pieces = []
        # Exercise dynamic batches, including a non-power-of-two tail.
        for begin in range(0, len(batch), 3):
            pieces.append(session.run(["embedding"], {"waveform": batch[begin : begin + 3]})[0])
        converted = np.concatenate(pieces)
        metrics = metric(reference_embedding, converted, np)
        fixtures.append(
            {
                "case": label,
                "windows": len(batch),
                "frontend_max_absolute_error": mel_error,
                **metrics,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        if (
            mel_error > FRONTEND_MAX_ERROR
            or metrics["max_absolute_error"] > EMBEDDING_MAX_ERROR
            or metrics["min_cosine"] < EMBEDDING_MIN_COSINE
        ):
            raise ValueError(f"Parity failed for {label}: {fixtures[-1]}")
        return reference_embedding, converted

    compare("synthetic", synthetic_fixtures(np))
    fingerprints = set()
    recordings = []
    for index, path in enumerate(audio_paths):
        samples = decode_fixture(path, np)
        digest = hashlib.sha256(samples.tobytes()).hexdigest()
        if digest in fingerprints:
            raise ValueError("Parity audio fixtures must be distinct")
        fingerprints.add(digest)
        recordings.append(samples)
        expected, actual = compare(f"recording-{index}-clean", windows(samples, np))
        reference_batches.append(expected)
        converted_batches.append(actual)
        fixtures[-1]["decoded_audio_sha256"] = digest

    def best_alignment(query, corpus):
        candidates = []
        for track_index, track in enumerate(corpus):
            for offset in range(len(track) - len(query) + 1):
                candidates.append(
                    (
                        float(np.sum(query * track[offset : offset + len(query)]) / len(query)),
                        track_index,
                        offset,
                    )
                )
        return max(candidates, key=lambda row: (row[0], -row[1], -row[2]))

    for index, samples in enumerate(recordings):
        clip = samples[8000:40000].copy()
        rms = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2)))
        noise = rng.normal(0, rms / np.sqrt(10), len(clip)).astype(np.float32)
        impulse = np.zeros(1201, dtype=np.float32)
        impulse[[0, 240, 800, 1200]] = [1, 0.25, 0.12, 0.08]
        variants = {
            "clean": clip,
            "quiet": clip * 0.01,
            "noise-10db": clip + noise,
            "room-echo": np.convolve(clip, impulse)[: len(clip)],
        }
        for name, variant in variants.items():
            expected, actual = compare(
                f"recording-{index}-{name}-query", windows(variant.astype(np.float32), np)
            )
            original_best = best_alignment(expected, reference_batches)
            converted_best = best_alignment(actual, converted_batches)
            consistent = original_best[1:] == converted_best[1:]
            retrieval.append(
                {
                    "recording": index,
                    "condition": name,
                    "same_top_track_and_offset": consistent,
                    "reference_top_track": original_best[1],
                    "reference_offset_windows": original_best[2],
                    "reference_score": original_best[0],
                    "export_score": converted_best[0],
                }
            )
            if not consistent:
                raise ValueError("Export changed top recording or alignment in parity fixtures")
    return {
        "fixtures": fixtures,
        "retrieval_parity": retrieval,
        "scope": "Conversion parity on synthetic and local decoded audio; not phone-recording calibration",
        "thresholds": {
            "frontend_max_absolute_error": FRONTEND_MAX_ERROR,
            "embedding_max_absolute_error": EMBEDDING_MAX_ERROR,
            "embedding_min_cosine": EMBEDDING_MIN_COSINE,
        },
    }


def export(args):
    # Set before importing either inference runtime. Export is CPU-only and bounded.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import numpy as np
    import tensorflow as tf
    import onnx
    import onnxruntime as ort
    import tf2onnx
    import yaml

    source, checkpoint, output = (
        args.source.resolve(),
        args.checkpoint.resolve(),
        args.output.resolve(),
    )
    if not args.license_acknowledged:
        raise ValueError("Explicit --license-acknowledged is required")
    if output.exists():
        raise ValueError(
            "Output directory already exists; choose a new immutable artifact directory"
        )
    if len(args.audio) < 3:
        raise ValueError("Provide at least three distinct local music files for conversion parity")
    verify_source(source)
    config_path = checkpoint / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    model_config = config["MODEL"]
    expected_input = {
        "STFT_WIN": 1024,
        "STFT_HOP": 256,
        "F_MIN": 160,
        "F_MAX": 4000,
        "N_MELS": 256,
        "DYNAMIC_RANGE": 80,
        "SCALE": True,
    }
    if (
        model_config["AUDIO"] != {"SEGMENT_DUR": 1.0, "FS": 8000}
        or any(model_config["INPUT"].get(k) != v for k, v in expected_input.items())
        or model_config["ARCHITECTURE"]["EMB_SZ"] != 128
        or model_config["ARCHITECTURE"]["BN"] != "layer_norm2d"
    ):
        raise ValueError("Checkpoint is incompatible with the pinned NMFP frontend")

    reference_frontend = load_module(
        "nmfp_reference_frontend", source / "nmfp/audio_processing/melspectrogram.py"
    ).Melspec_layer(f_min=160)
    model_class = load_module("nmfp_reference_encoder", source / "nmfp/model/nnfp.py").FingerPrinter
    tf.keras.mixed_precision.set_global_policy("float32")
    model = model_class(emb_sz=128, norm="layer_norm2d", mixed_precision=False)
    model(tf.zeros([1, 256, 33, 1], tf.float32), training=False)
    checkpoint_path = tf.train.latest_checkpoint(str(checkpoint))
    if checkpoint_path is None:
        raise ValueError("No TensorFlow checkpoint found")
    checkpoint_files = sorted(Path(checkpoint_path).parent.glob(Path(checkpoint_path).name + ".*"))
    archive_sha256 = verify_checkpoint_archive(
        args.checkpoint_archive, checkpoint, [config_path, *checkpoint_files]
    )
    restore = tf.train.Checkpoint(model=model).restore(checkpoint_path)
    restore.assert_existing_objects_matched()
    restore.expect_partial()  # Optimizer/training variables are intentionally excluded.
    model.trainable = False
    compute_frontend = frontend(reference_frontend, tf, np)

    @tf.function(input_signature=[tf.TensorSpec([None, 8000], tf.float32, name="waveform")])
    def waveform_encoder(waveform):
        return tf.identity(
            model(compute_frontend(waveform)[..., None], training=False), name="embedding"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".recording-export-", dir=output.parent) as staging:
        staging = Path(staging)
        model_path = staging / "encoder.onnx"
        graph, _ = tf2onnx.convert.from_function(
            waveform_encoder,
            input_signature=waveform_encoder.input_signature,
            opset=17,
            output_path=None,
        )
        old_name = graph.graph.output[0].name
        for node in graph.graph.node:
            for names in (node.input, node.output):
                for index, value in enumerate(names):
                    if value == old_name:
                        names[index] = "embedding"
        graph.graph.output[0].name = "embedding"
        onnx.checker.check_model(graph)
        onnx.save_model(graph, str(model_path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        report = validate(reference_frontend, compute_frontend, model, session, args.audio, np)
        report.update(
            {
                "passed": True,
                "export_revision": EXPORT_REVISION,
                "source_commit": SOURCE_COMMIT,
                "source_repository": "https://github.com/raraz15/neural-music-fp",
                "checkpoint_record": CHECKPOINT_RECORD,
                "published_archive_md5": CHECKPOINT_ARCHIVE_MD5,
                "checkpoint_archive_sha256": archive_sha256,
                "checkpoint_files": {file.name: sha256(file) for file in checkpoint_files},
                "checkpoint_config_sha256": sha256(config_path),
                "precision": "float32",
                "export_script_sha256": sha256(Path(__file__)),
                "license_acknowledged_by_operator": True,
                "model_sha256": sha256(model_path),
                "python": platform.python_version(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "tensorflow-cpu",
                        "tf2onnx",
                        "onnx",
                        "onnxruntime",
                        "essentia",
                        "numpy",
                    )
                },
            }
        )
        report_path = staging / "parity.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        manifest = {
            "schema_version": 1,
            "enabled": True,
            "license_acknowledged": True,
            "parity_validated": True,
            "model_path": "encoder.onnx",
            "model_sha256": sha256(model_path),
            "validation_report_path": "parity.json",
            "validation_report_sha256": sha256(report_path),
            "frontend": "embedded-waveform-v1",
            "sample_rate": 8000,
            "window_samples": 8000,
            "hop_samples": 4000,
            "embedding_dim": 128,
            "input_name": "waveform",
            "output_name": "embedding",
            "batch_size": 32,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Preserve upstream license and the exact exporter used by this report.
        (staging / "UPSTREAM-LICENSE").write_bytes((source / "LICENSE").read_bytes())
        (staging / "exporter.py").write_bytes(Path(__file__).read_bytes())
        # TemporaryDirectory is private (0700). The published model bundle must
        # be readable by the separate, non-root worker UID on the shared volume.
        # No source audio is included in this directory.
        for artifact in staging.iterdir():
            artifact.chmod(0o644)
        staging.chmod(0o755)
        os.rename(staging, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "model_sha256": manifest["model_sha256"],
                "parity_passed": True,
                "recognition_calibrated": False,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio", type=Path, action="append", required=True)
    parser.add_argument("--license-acknowledged", action="store_true")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
