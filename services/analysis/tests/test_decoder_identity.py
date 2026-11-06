"""CPU tensor regressions; runnable without pytest via unittest."""

import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "fa_kara"
sys.path.insert(0, str(VENDOR))
import ctc_segmentation as ctc


class DecoderIdentityTests(unittest.TestCase):
    def test_pseudo_zero_bonus_equivalence(self):
        for groups in ([[1, 1, 2]], [[1], [1, 2]], [[1, 2], [2, 1]]):
            letters = [x for g in groups for x in g]
            pseudo, offset = [], 0
            for g in groups:
                pseudo.append(list(range(offset, offset + len(g))))
                offset += len(g)
            for blank_free in (False, True):
                for seed in range(8):
                    torch.manual_seed(seed)
                    emission = torch.randn(12, 3).log_softmax(-1)
                    copied = torch.cat((emission[:, letters], emission[:, :1]), 1)
                    kwargs = dict(minimum_frames=[1] * len(letters), blank_free_tokens=blank_free)
                    normal = ctc.align_with_minimum_durations(emission, groups, blank=0, **kwargs)
                    unique = ctc.align_with_minimum_durations(
                        copied, pseudo, blank=len(letters), acoustic_token_groups=groups, **kwargs
                    )
                    self.assertEqual(
                        [(s.start, s.end, s.score) for g in normal for s in g],
                        [(s.start, s.end, s.score) for g in unique for s in g],
                    )

    def test_frame_zero_edge(self):
        # Equal emissions: avoid the much less likely boundary at frame zero.
        spans = ctc.align_with_minimum_durations(
            torch.zeros(2, 2),
            [[1]],
            blank=0,
            minimum_frames=[1],
            edge_logits=torch.tensor([-10.0, -2.0]),
            edge_weight=1.0,
        )
        self.assertEqual(spans[0][0].start, 1)

    def decode_lead(self, **overrides):
        # Execute the production closure without loading model/audio dependencies.
        tree = ast.parse((VENDOR / "align_yohane.py").read_text())
        node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_decode_lead"
        )
        env = dict(
            vars(ctc),
            os=os,
            torch=torch,
            aligner_model=SimpleNamespace(
                blank=0,
                last_edge_logits=torch.zeros(2),
                tokenizer=SimpleNamespace(get_vocab=lambda: {"a": 1}),
            ),
            encoded=[[1, 1]],
            acoustic_tokens=["aa"],
            frame_shift=0.02,
            vocal_activity_weight=0,
            min_letter_ms=0,
            min_token_ms=0,
            duration_pace_ratio=0,
            blank_free_tokens=True,
            edge_weight=0,
            text_share=None,
            optional_share=None,
            text_bonus=0,
            line_acoustic_ranges=[(0, 1)],
            optional_groups=None,
        )
        env.update(overrides)
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]), str(VENDOR / "align_yohane.py"), "exec"
            ),
            env,
        )
        return env["_decode_lead"](torch.zeros(2, 2))

    def test_blank_free_without_floors_or_ordinary_prepass(self):
        result = self.decode_lead()
        self.assertEqual([(s.start, s.end) for s in result[2][0]], [(0, 1), (1, 2)])
        self.assertIsNone(result[3])

    def test_floor_fallback_preserves_topology_and_bonus(self):
        result = self.decode_lead(min_letter_ms=100, text_share=torch.zeros(2, 1), text_bonus=1.0)
        self.assertEqual(result[3], "too_dense_one_frame_fallback")
        self.assertEqual([s.token for s in result[2][0]], [1, 1])

    def test_options_without_floors_reach_decoder(self):
        from unittest.mock import patch

        for option in ("text", "edge", "gap"):
            calls = []

            def decode(*args, **kwargs):
                calls.append(kwargs)
                return ctc.align_with_minimum_durations(*args, **kwargs)

            overrides = dict(align_with_minimum_durations=decode)
            if option == "text":
                overrides.update(text_share=torch.zeros(2, 1), text_bonus=1.0)
            if option == "edge":
                overrides.update(edge_weight=1.0)
            with patch.dict(
                os.environ, {"FA_KARA_IN_LINE_GAP_COST": "1" if option == "gap" else "0"}
            ):
                self.decode_lead(**overrides)
            self.assertEqual(calls[0]["minimum_frames"], [1, 1])
            self.assertTrue(calls[0]["blank_free_tokens"])
            if option == "text":
                self.assertEqual(calls[0]["blank"], 2)
                self.assertEqual(calls[0]["acoustic_token_groups"], [[1, 1]])
            elif option == "edge":
                self.assertEqual(calls[0]["edge_weight"], 1.0)
            else:
                self.assertEqual(calls[0]["in_line_gap_cost"], 1.0)

    def test_edge_without_floors_requires_head(self):
        model = SimpleNamespace(
            blank=0, last_edge_logits=None, tokenizer=SimpleNamespace(get_vocab=lambda: {})
        )
        with self.assertRaisesRegex(RuntimeError, "edge head"):
            self.decode_lead(edge_weight=1.0, aligner_model=model)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
