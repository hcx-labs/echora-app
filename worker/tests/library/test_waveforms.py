import unittest

import numpy as np

from worker.library.waveforms import waveform_peaks


class WaveformTests(unittest.TestCase):
    def test_peaks_cover_whole_track_and_last_sample(self):
        samples = np.zeros((2051, 2), dtype=np.float32)
        samples[0, 0] = 0.5
        samples[-1, 1] = 1
        peaks = waveform_peaks(samples)
        self.assertEqual(len(peaks), 1024)
        self.assertGreater(peaks[0], 0)
        self.assertLess(peaks[0], 0.5)
        self.assertEqual(peaks[-1], 1)
        self.assertTrue(all(value == 0 for value in peaks[1:-1]))

    def test_antiphase_stereo_does_not_cancel(self):
        samples = np.array([[0.25, -0.25], [0.5, -0.5]], dtype=np.float32)
        self.assertEqual(waveform_peaks(samples, 2), [round(0.5**1.5, 4), 1])

    def test_silence_and_short_clips(self):
        self.assertEqual(waveform_peaks(np.zeros((3, 2))), [0, 0, 0])
        self.assertEqual(waveform_peaks(np.array([0.2])), [1])

    def test_reject_invalid_audio(self):
        for samples in (np.array([]), np.array([np.nan]), np.array([np.inf])):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                waveform_peaks(samples)

    def test_invalid_bucket_count(self):
        with self.assertRaises(ValueError):
            waveform_peaks(np.ones(4), 0)

    def test_equal_peaks_do_not_hide_different_sustained_energy(self):
        quiet = np.full((1000, 2), 0.05)
        loud = np.full((1000, 2), 0.7)
        quiet[0] = loud[0] = 1
        bars = waveform_peaks(np.concatenate([quiet, loud]), 2)
        self.assertLess(bars[0], 0.2)
        self.assertEqual(bars[1], 1)

    def test_constant_energy_is_not_given_fake_variation(self):
        self.assertEqual(waveform_peaks(np.full((100, 2), 0.4), 10), [1] * 10)

    def test_gain_does_not_change_display_shape(self):
        samples = np.arange(1, 101, dtype=float) / 100
        self.assertEqual(waveform_peaks(samples, 10), waveform_peaks(samples * 0.1, 10))
