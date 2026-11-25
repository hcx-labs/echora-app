"""Recording search configuration: limits, retention, calibration and the model bundle."""

from __future__ import annotations

from core.config import EchoraConfig, setting


class RecordingConfig(EchoraConfig, env_prefix="ECHORA_RECORDING_"):
    """Recording search and calibration. Every name is ECHORA_RECORDING_<FIELD>."""

    # Model bundle.
    model_manifest: str = setting("", description="Local bundle manifest (blank: search off).")
    model_id: str = setting("")
    model_revision: str = setting("")
    model_directory: str = setting("/data/models/recording")
    match_policy: str = setting("", description="Calibrated match policy file.")
    intra_op_threads: int = setting(4, gt=0)
    inter_op_threads: int = setting(1, gt=0)
    # Queries.
    max_upload_bytes: int = setting(8388608, gt=0)
    max_active: int = setting(32, gt=0)
    max_active_per_user: int = setting(1, gt=0)
    uploads_per_minute: int = setting(6, gt=0)
    upload_read_timeout_seconds: float = setting(30, gt=0)
    decode_timeout_seconds: float = setting(30, gt=0)
    processing_timeout_seconds: float = setting(120, gt=0)
    cancel_poll_seconds: float = setting(0.25, gt=0)
    audio_retention_seconds: int = setting(900, gt=0)
    result_retention_seconds: int = setting(86400, gt=0)
    diagnostic_retention_seconds: int = setting(86400, gt=0)
    # Calibration recordings, kept only with consent.
    calibration_enabled: bool = setting(False)
    calibration_dir: str = setting("/data/recording-calibration")
    calibration_max_user_samples: int = setting(20, gt=0)
    calibration_max_total_samples: int = setting(200, gt=0)
    calibration_uploads_per_minute: int = setting(6, gt=0)
    calibration_retention_seconds: int = setting(604800, gt=0)
