import zlib

import numpy as np

from worker.library.fingerprints import fingerprint_similarity


def _fingerprint(values: np.ndarray) -> bytes:
    return zlib.compress(values.astype(np.uint32).tobytes())


def test_identical_chromaprints_match():
    values = np.arange(120, dtype=np.uint32) * 2654435761
    assert fingerprint_similarity(_fingerprint(values), _fingerprint(values)) == 1.0


def test_alignment_tolerates_leading_frames():
    values = np.arange(120, dtype=np.uint32) * 2246822519
    shifted = np.concatenate([np.array([9, 12, 15], dtype=np.uint32), values])
    assert fingerprint_similarity(_fingerprint(values), _fingerprint(shifted)) == 1.0


def test_unrelated_chromaprints_have_low_agreement():
    generator = np.random.default_rng(42)
    left = generator.integers(0, 2**32, 120, dtype=np.uint32)
    right = generator.integers(0, 2**32, 120, dtype=np.uint32)
    assert fingerprint_similarity(_fingerprint(left), _fingerprint(right)) < 0.6
