"""Optional analysis features, switched on per location (see docs/modal-compute.md).

A sync runs either on this server or on Modal. Each optional feature has one
switch per location, so a feature may run only where its hardware is: karaoke
switched on for Modal alone is skipped by local syncs and done by Modal syncs.
Core analysis (audio and lyric embeddings, voice) has no switch.
"""

from __future__ import annotations

from worker.remote.session import location as current_location

# Feature: (local column, Modal column, default before settings are saved).
FEATURES = {
    "transcription": ("transcription_processing_enabled", "transcription_modal_enabled", False),
    "karaoke": ("karaoke_processing_enabled", "karaoke_modal_enabled", True),
    "hum": ("hum_processing_enabled", "hum_modal_enabled", True),
}


def feature_enabled(connection, feature: str, where: str | None = None) -> bool:
    """Whether a feature runs in the current job's location (or the one given)."""
    local, modal, default = FEATURES[feature]
    column = modal if (where or current_location()) == "modal" else local
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT {column} FROM analysis_settings WHERE singleton=true")
        row = cursor.fetchone()
    if row is None:
        return default
    value = row[column] if isinstance(row, dict) else row[0]
    return bool(value)
