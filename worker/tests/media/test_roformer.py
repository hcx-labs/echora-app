import sys
import types
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
import torch

import worker.media.roformer as r


class Cancelled(Exception):
    pass


class FakeModel(torch.nn.Module):
    def __init__(self, failure=None):
        super().__init__()
        self.failure = failure
        self.calls = 0
        self.released = False

    def forward(self, batch):
        self.calls += 1
        if self.failure == "raise":
            raise RuntimeError("inference failed")
        if self.failure == "shape":
            return batch[:, :1]
        if self.failure == "nan":
            return batch * float("nan")
        return batch.clone()

    def cpu(self):
        self.released = True
        return super().cpu()


class RoformerTests(unittest.TestCase):
    def setUp(self):
        # Exercise real torch inference/CPU plumbing, but tiny chunks and no weights.
        self.model = FakeModel()
        self.patches = [
            patch.object(r, "_CHUNK_SIZE", 80),
            patch.object(r, "_new_model", return_value=self.model),
            patch.object(r, "_load_weights"),
            patch.object(torch.cuda, "is_available", return_value=False),
        ]
        self.mocks = [p.start() for p in self.patches]
        for p in self.patches:
            self.addCleanup(p.stop)

    def test_separation_phase_loads_once_for_many_tracks_and_releases(self):
        audio = np.random.default_rng(1).uniform(-1, 1, (100, 2)).astype("float32")
        new_model = self.mocks[1]
        with r.separation_phase():
            for _ in range(3):
                np.testing.assert_allclose(r.separate_vocals(audio), audio)
            self.assertEqual(new_model.call_count, 1)
            self.assertFalse(self.model.released)
        self.assertTrue(self.model.released)
        # Outside a phase, each call loads its own model again.
        r.separate_vocals(audio)
        self.assertEqual(new_model.call_count, 2)

    def test_separation_phase_loads_nothing_when_no_track_needs_separating(self):
        with r.separation_phase():
            r.separate_vocals(np.empty((0, 2), dtype="float32"))
        self.mocks[1].assert_not_called()

    def test_identity_at_short_chunk_and_reflection_boundaries(self):
        # Cover both the overlap-2 crop threshold at 80 and the old threshold at 140.
        for length in (0, 1, 2, 9, 10, 39, 40, 41, 79, 80, 81, 139, 140, 141, 160, 237):
            with self.subTest(length=length):
                audio = np.random.default_rng(length).uniform(-1, 1, (length, 2)).astype("float32")
                before = audio.copy()
                result = r.separate_vocals(audio)
                self.assertEqual(result.shape, audio.shape)
                self.assertEqual(result.dtype, np.float32)
                np.testing.assert_allclose(result, audio, atol=1e-7, rtol=1e-7)
                np.testing.assert_array_equal(audio, before)
                self.assertFalse(np.shares_memory(result, audio))

    def test_noncontiguous_readonly_input(self):
        audio = np.arange(800, dtype=np.float32).reshape(400, 2)[::-3]
        audio.flags.writeable = False
        np.testing.assert_allclose(r.separate_vocals(audio), audio)

    def test_weighted_boundary_numerics(self):
        # Chunk k predicts constant k+1. Independent scalar reference checks fades,
        # hop spacing, final partial chunks and normalization, not just identity.
        def predict(model, part, device):
            model.calls += 1
            return np.full_like(part, model.calls)

        for overlap, step, offset, total in ((2, 40, 40, 180), (8, 10, 0, 100)):
            with self.subTest(overlap=overlap):
                self.model.calls = 0
                with (
                    patch.object(r, "_OVERLAP", overlap),
                    patch.object(r, "_predict", side_effect=predict),
                ):
                    result = r.separate_vocals(np.zeros((100, 2), dtype=np.float32))
                for sample in range(100):
                    numerator = denominator = 0.0
                    for start in range(0, total, step):
                        index = sample + offset - start
                        if not 0 <= index < 80:
                            continue
                        weight = 1.0
                        if index < 8 and start != 0:
                            weight = index / 7
                        if index >= 72 and start + 80 < total:
                            weight = (79 - index) / 7
                        numerator += (start // step + 1) * weight
                        denominator += weight
                    np.testing.assert_allclose(result[sample], numerator / denominator, rtol=1e-6)

    def test_invalid_inputs_do_not_load(self):
        for audio in (
            [1],
            np.zeros((2, 2)),
            np.zeros(2, dtype="float32"),
            np.zeros((2, 1), dtype="float32"),
            np.full((2, 2), np.nan, dtype="float32"),
            np.full((2, 2), np.inf, dtype="float32"),
        ):
            with self.subTest(audio=audio), self.assertRaises(ValueError):
                r.separate_vocals(audio)
        self.mocks[1].assert_not_called()

    def test_empty_does_not_load(self):
        r.separate_vocals(np.empty((0, 2), dtype="float32"))
        self.mocks[1].assert_not_called()

    def test_cancellation_before_load(self):
        with self.assertRaises(Cancelled):
            r.separate_vocals(np.zeros((1, 2), dtype="float32"), Mock(side_effect=Cancelled))
        self.mocks[1].assert_not_called()

    def test_cancellation_between_chunks_releases_model(self):
        checks = 0

        def check():
            nonlocal checks
            checks += 1
            if self.model.calls == 2:
                raise Cancelled()

        with self.assertRaises(Cancelled):
            r.separate_vocals(np.zeros((100, 2), dtype="float32"), check)
        self.assertEqual(self.model.calls, 2)
        self.assertEqual(checks, 5)  # entry, post-load, then one check per chunk
        self.assertTrue(self.model.released)

    def test_inference_errors_release_model(self):
        for failure in ("shape", "nan", "raise"):
            with self.subTest(failure=failure):
                self.model.failure = failure
                self.model.released = False
                with self.assertRaises((ValueError, RuntimeError)):
                    r.separate_vocals(np.zeros((100, 2), dtype="float32"))
                self.assertTrue(self.model.released)

    def test_load_failure_releases_model(self):
        self.mocks[2].side_effect = RuntimeError("missing checkpoint")
        with self.assertRaisesRegex(RuntimeError, "missing checkpoint"):
            r.separate_vocals(np.zeros((1, 2), dtype="float32"))
        self.assertTrue(self.model.released)

    def test_fresh_model_each_call(self):
        models = []

        def factory():
            model = FakeModel()
            models.append(weakref.ref(model))
            return model

        self.mocks[1].side_effect = factory
        for _ in range(2):
            r.separate_vocals(np.zeros((1, 2), dtype="float32"))
        self.assertEqual(self.mocks[2].call_count, 2)
        # Mock call history retains model arguments; remove it before checking GC.
        self.mocks[2].reset_mock()
        self.assertTrue(all(ref() is None for ref in models))

    def test_cuda_cleanup_without_real_cuda(self):
        for fail in (False, True):
            model = Mock()
            with (
                self.subTest(fail=fail),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "empty_cache") as empty_cache,
                patch.object(r, "_new_model", return_value=model),
                patch.object(
                    r,
                    "_demix",
                    return_value=np.zeros((1, 2), dtype="float32"),
                    side_effect=Cancelled if fail else None,
                ),
            ):
                if fail:
                    with self.assertRaises(Cancelled):
                        r.separate_vocals(np.zeros((1, 2), dtype="float32"))
                else:
                    r.separate_vocals(np.zeros((1, 2), dtype="float32"))
                model.cpu.assert_called_once()
                empty_cache.assert_called_once()

    def test_pinned_local_weights_only_strict_loading(self):
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = Mock(return_value="/cached/model.ckpt")
        model = Mock()
        state = {"weight": torch.ones(1)}
        # Undo only this test's loader patch; the real loader uses a fake HF module.
        self.patches[2].stop()
        with (
            patch.dict(sys.modules, {"huggingface_hub": hub}),
            patch.object(torch, "load", return_value=state) as load,
        ):
            r._load_weights(model)
        hub.hf_hub_download.assert_called_once_with(
            repo_id="KimberleyJSN/melbandroformer",
            revision="ac9b0614ab3cd7f77219e18ba494dfd93956c348",
            filename="MelBandRoformer.ckpt",
            local_files_only=True,
        )
        load.assert_called_once_with("/cached/model.ckpt", map_location="cpu", weights_only=True)
        model.load_state_dict.assert_called_once_with(state, strict=True)

    def test_revision_and_production_geometry(self):
        self.patches[0].stop()
        self.assertEqual(r._CHUNK_SIZE, 352800)
        self.assertEqual(r._CHUNK_SIZE // r._OVERLAP, 176400)
        self.assertEqual(r.SAMPLE_RATE, 44100)
        for value in ("overlap2", "config", "source", r.MODEL_REVISION):
            self.assertIn(value, r.SEPARATION_REVISION)


if __name__ == "__main__":
    unittest.main()
