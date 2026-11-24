"""Revisions of stored results: what each stored artifact was computed with.

The worker stores these with every result and replans a song whose stored revision differs; the
backend reports results as current or stale by the same values. Changing one reprocesses every
song for that result, so it changes only with the computation it names.
"""

# Mel-Band-Roformer vocal separation (worker.media.roformer).
SEPARATOR_MODEL_ID = "KimberleyJSN/melbandroformer"
SEPARATOR_MODEL_REVISION = "ac9b0614ab3cd7f77219e18ba494dfd93956c348"
SEPARATOR_MODEL_FILENAME = "MelBandRoformer.ckpt"
SEPARATOR_SOURCE_REVISION = "25f44ffb55ee3c301281bba21b2d6d311cb69ae2"
SEPARATOR_CHUNK_SIZE = 352800
SEPARATOR_OVERLAP = 2
SEPARATION_REVISION = (
    f"{SEPARATOR_MODEL_ID}@{SEPARATOR_MODEL_REVISION}/{SEPARATOR_MODEL_FILENAME}:"
    f"overlap{SEPARATOR_OVERLAP}:chunk{SEPARATOR_CHUNK_SIZE}:"
    f"config_vocals_mel_band_roformer.yaml@{SEPARATOR_SOURCE_REVISION}:"
    f"source@{SEPARATOR_SOURCE_REVISION}:echora-v1"
)

# Melody index contours (worker.melody).
MELODY_CONTOUR_REVISION = f"multi-source-roformer-residual-v3:{SEPARATION_REVISION}"
MELODY_EXTRACTOR = "roformer-overlap2-residual-plus-essentia-melodia"

# Ingest-time analysis (worker.library).
WAVEFORM_REVISION = "2"
DESCRIPTOR_REVISION = "1"
VISUAL_FEATURE_REVISION = "2"

# Audio profiles (worker.profiles) and the embedding models that have them.
AUDIO_PROFILE_REVISION = "1"
SUPPORTED_PROFILE_MODELS = ("muq_mulan", "mert")

# Karaoke alignment (worker.kara).
KARAOKE_PIPELINE_REVISION = "v9-echora-kara-mix"

# Lyric theme descriptions (worker.themes): the first 16 hex digits of the SHA-256 of the system
# prompt. worker/tests/themes checks it against the prompt, so a prompt change must update it.
LYRIC_THEMES_REVISION = "d1923433205ae09c"
