"""Inference regression tests; no checkpoints, GPU, network or database required."""

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "services/analysis/vendor/fa_kara"))
import main as vendor
from align_yohane import _display_span_results, _assign_ctc_blank_holds
from torchaudio.functional import TokenSpan
from echora_analysis import karaoke_pipeline as pipeline


class InferenceRegression(unittest.TestCase):
    def test_source_boundaries(self):
        for text in ("one\ntwo", "one\ntwo\n", "\none\n\ntwo\n\n"):
            with self.subTest(text=text):
                records, texts = vendor.normalize_source_lines(io.StringIO(text), "en", 0, 1)
                self.assertEqual(texts, text.splitlines())
                self.assertEqual(sum(r["orig"] == "\n" for r in records), len(texts))
                self.assertFalse(any("\n" in r.get("pron", "") for r in records))

    def test_punctuation_only_lines_preserve_source_boundaries(self):
        for punctuation in ("...", "…", " — ", "！？"):
            with self.subTest(punctuation=punctuation):
                source = ["one", punctuation, "two"]
                records, texts = vendor.normalize_source_lines(source, "en", 0, 1)
                self.assertEqual(texts, source)
                self.assertEqual(sum(r["orig"] == "\n" for r in records), 3)
                lines = [[]]
                for record in records:
                    if record["orig"] == "\n":
                        lines.append([])
                    elif record.get("pron"):
                        lines[-1].append(record)
                self.assertTrue(lines[0])
                self.assertEqual(lines[1], [])
                self.assertTrue(lines[2])
                doc = {
                    "schema_version": 1,
                    "alignment": {
                        "lines": [
                            {
                                "source_index": 0,
                                "text": "one",
                                "tokens": [{"start_ms": 10, "end_ms": 20, "ctc_score": 0.8}],
                            },
                            {"source_index": 1, "text": punctuation, "tokens": []},
                            {
                                "source_index": 2,
                                "text": "two",
                                "tokens": [{"start_ms": 30, "end_ms": 40, "ctc_score": 0.8}],
                            },
                        ]
                    },
                }
                pipeline._validate_alignment_document(doc)
                doc["alignment"]["lines"][1]["text"] = "Ж"
                with self.assertRaises(RuntimeError):
                    pipeline._validate_alignment_document(doc)

    def test_unsupported_line_fails_at_source_index(self):
        with self.assertRaisesRegex(ValueError, "source index 1"):
            vendor.normalize_source_lines(["hello", "Ж"], "en", 0, 1)

    def test_blank_anchors_and_bounds(self):
        source = [
            {"text": "one", "start_ms": 100},
            {"text": "", "start_ms": 500},
            {"text": "two", "start_ms": 900},
        ]
        self.assertEqual(len(pipeline._anchored_source_lines(source)), 3)
        raw = [
            {
                "text": "one",
                "start_ms": 0,
                "end_ms": 800,
                "syllables": [{"start_ms": 0, "end_ms": 800}],
            }
        ]
        bounded = pipeline.bound_to_synced_lines(raw, source)
        self.assertEqual(bounded[0]["syllables"], [{"start_ms": 100, "end_ms": 500}])
        self.assertEqual(raw[0]["end_ms"], 800)

    def test_empty_line_document(self):
        doc = {
            "schema_version": 1,
            "alignment": {
                "lines": [
                    {"source_index": 0, "text": "", "tokens": []},
                    {
                        "source_index": 1,
                        "text": "one",
                        "tokens": [{"start_ms": 10, "end_ms": 20, "ctc_score": 0.8}],
                    },
                    {"source_index": 2, "text": "", "tokens": []},
                ]
            },
        }
        pipeline._validate_alignment_document(doc)
        doc["alignment"]["lines"][0]["text"] = "missing"
        with self.assertRaises(RuntimeError):
            pipeline._validate_alignment_document(doc)

    def test_raw_and_capped_holds(self):
        spans = [
            [TokenSpan(1, 2, 4, 0.8)],
            [TokenSpan(2, 100, 102, 0.9)],
            [TokenSpan(3, 200, 202, 0.9)],
        ]
        for duration in (0.02, 0.04):
            with patch.dict(os.environ, {"FA_KARA_MAX_BLANK_HOLD_SECONDS": "0.2"}):
                result = _display_span_results(["a", "b", "c"], spans, [(0, 2), (2, 3)], duration)
            self.assertAlmostEqual(result[0]["acoustic_end"], 4 * duration)
            self.assertLessEqual(result[0]["original_end"] - result[0]["acoustic_end"], 0.20000001)
            self.assertEqual(result[1]["original_end"], result[1]["acoustic_end"])
            self.assertEqual(spans[0][0].end, 4)
        self.assertEqual(_assign_ctc_blank_holds(spans, [(0, 3)], 0.02, 0), spans)
        for limit in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                _assign_ctc_blank_holds(spans, [(0, 3)], 0.02, limit)

    def test_diagnostics_source_indexes(self):
        value = {"bad_lines": [1], "details": [{"line": 0}], "vocal_focus_attempted_runs": [[0, 1]]}
        self.assertEqual(
            vendor.source_diagnostic_indexes(value, [1, 3]),
            {"bad_lines": [3], "details": [{"line": 1}], "vocal_focus_attempted_runs": [[1, 3]]},
        )

    def test_identity_excludes_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.safetensors", "config.json", "optimizer.pt", "scheduler.pt"):
                (root / name).write_bytes(b"test")
            (root / "checkpoint-1").mkdir()
            (root / "checkpoint-1/model.safetensors").write_bytes(b"ignored")
            identity = vendor.file_identity(root)
            self.assertEqual(set(identity["sha256"]), {"model.safetensors", "config.json"})
            (root / "model.safetensors").write_bytes(b"changed")
            self.assertNotEqual(identity, vendor.file_identity(root))

    def test_actual_benchmark_provenance(self):
        path = ROOT / "data/finetune-curation/quick-tests/benchmark_fa_kara_checkpoint.py"
        spec = importlib.util.spec_from_file_location("benchmark", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = {
            "model": "actual-model",
            "model_revision": "/actual/checkpoint",
            "diagnostics": {
                "device": "cpu",
                "checkpoint_identity": {"sha256": {"model.safetensors": "abc"}},
            },
            "alignment_document": {"alignment": {"mode": "actual-mode"}},
        }
        provenance = module.benchmark_provenance(
            result, b"audio", "one", [], "label-run", "label-only"
        )
        self.assertEqual(provenance["model_revision"], "/actual/checkpoint")
        self.assertEqual(provenance["alignment_mode"], "actual-mode")
        self.assertEqual(provenance["diagnostics"], result["diagnostics"])
        self.assertNotEqual(provenance["downloaded_audio_sha256"], provenance["source_sha256"])


if __name__ == "__main__":
    unittest.main()
