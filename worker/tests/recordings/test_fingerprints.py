"""Recording fingerprints of a whole song: reuse, encoding and the source-identity check."""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

import worker.media.audio as audio
import worker.recordings.encoder as recording_encoder
from worker.recordings import search


def test_full_track_fingerprints_reuse_and_source_identity(monkeypatch):
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    content = b"original source"
    config = SimpleNamespace(representation_id="model")
    samples = np.ones(181 * 8000, dtype=np.float32)
    fingerprints = np.ones((361, 128), dtype=np.float32)
    decoder = Mock(return_value=samples)
    encoder = Mock(return_value=fingerprints)
    monkeypatch.setattr(audio, "decode_audio", decoder)
    monkeypatch.setattr(recording_encoder, "encode", encoder)
    check = Mock()
    # Missing for this encoder, and the bytes are the track's own: encode and store.
    cursor.fetchone.side_effect = [None, {"audio_hash": hashlib.sha256(content).hexdigest()}]
    assert search.needs_fingerprints(connection, "track", content, config)
    encoded = search.encode_fingerprints(content, config, check)
    decoder.assert_called_once_with(content, 8000)
    assert encoder.call_args.args[0] is samples
    cursor.rowcount = 1
    assert search.insert_fingerprints(connection, "track", encoded, config)
    assert cursor.execute.call_args.args[1][2] == 361
    assert len(cursor.execute.call_args.args[1][3]) == 361 * 128 * 4
    # Stored already, or by another worker first: nothing to do.
    cursor.fetchone.side_effect = [{"?column?": 1}]
    assert not search.needs_fingerprints(connection, "track", content, config)
    cursor.rowcount = 0
    assert not search.insert_fingerprints(connection, "track", encoded, config)
    # Other bytes than the track's identity: refused before encoding.
    cursor.fetchone.side_effect = [None, {"audio_hash": "old-source-hash"}]
    with pytest.raises(ValueError, match="source bytes changed"):
        search.needs_fingerprints(connection, "track", content, config)
    assert decoder.call_count == encoder.call_count == 1
