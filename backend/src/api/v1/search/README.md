# Search (`/api/v1/search`)

Finding songs in the user's library by humming, by a recorded clip and by named concepts.
Searches that score against a per-user index run as search services
(`backend/src/search/`): hum search keeps the user's melody contours in memory and the concept
lens reuses the persistent MuQ-MuLan and BGE-M3 corpora (`corpora.py`); both run on the bounded
search pool. Recording search is a job: the endpoint queues the clip and the worker matches it.

| Endpoint | File | |
| --- | --- | --- |
| `POST /search/hum` | `hum.py` | the body is the recording, up to 8 MB |
| `GET /search/hum/index` | `hum_index_status.py` | how many songs have contours |
| `POST /search/hum/index` | `hum_index_build.py` | queue a hum index build (202) |
| `GET /search/recording/status` | `recording_status.py` | recognition and calibration availability |
| `POST /search/recording` | `recording.py` | queue a clip (202); 429 with `Retry-After` when busy |
| `GET /search/recording/{job_id}` | `recording_result.py` | state, then matches |
| `DELETE /search/recording/{job_id}` | `recording_cancel.py` | stop a search |
| `GET /search/recording/calibration/samples` | `calibration_samples.py` | saved calibration clips |
| `POST /search/recording/calibration/samples` | `calibration_sample_save.py` | save a clip, with consent (201) |
| `DELETE /search/recording/calibration/samples/{id}` | `calibration_sample_delete.py` | (204) |
| `POST /search/concepts` | `concepts.py` | concept percentiles per song (the Galaxy's concept lens) |

Uploads (hum, recording, calibration) are the raw request body, not JSON. The recording
endpoints read it as a stream with `ECHORA_RECORDING_*` size and time limits
(`worker/src/recordings/config.py`).
