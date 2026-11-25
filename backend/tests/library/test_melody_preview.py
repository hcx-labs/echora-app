import unittest

from backend.library.melody_preview import melody_preview


class MelodyPreviewTests(unittest.TestCase):
    def test_preserves_pitch_timing_and_gaps(self):
        result = melody_preview("full-mix", [60, 62, 0, 67], [True, True, False, True], 0.1)
        self.assertEqual([p["pitch"] for p in result["points"]], [60, 62, None, 67])
        self.assertEqual(result["points"][-1]["time_seconds"], 0.3)

    def test_bounded_and_not_stretched_to_track_length(self):
        result = melody_preview("vocals", [60] * 10000, [True] * 10000, 0.1, 100)
        self.assertEqual(len(result["points"]), 100)
        self.assertEqual(result["points"][-1]["time_seconds"], 994.95)

    def test_missing_and_invalid_data(self):
        self.assertIsNone(melody_preview("vocals", [], [], 0.1))
        self.assertIsNone(melody_preview("vocals", [60], [False], 0.1))
        self.assertIsNone(melody_preview("vocals", [float("nan")], [True], 0.1))
        self.assertIsNone(melody_preview("vocals", [60], [True], 0))
