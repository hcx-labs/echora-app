"""Process configuration (no implicit .env loading).

Environment names are case-insensitive; ECHORA_ aliases omit that prefix in Python.
Empty optional model identifiers/manifests remain disabled, not defaulted upstream.
Entry points must require database_url; pure modules may use the empty default.

Exceptions: ECHORA_JOB_ID is per-job context; TMPDIR/PYTHONPATH are subprocess
runtime state. HF_HOME and NUMBA_CACHE_DIR are vendor runtime environment, not
application configuration. Model algorithm constants remain in versioned modules;
existing fusion/aligner options remain part of their recorded model contracts.
"""

from functools import lru_cache
import secrets

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SESSION_SECRET = secrets.token_urlsafe(48)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        hide_input_in_errors=True,
        case_sensitive=False,
        populate_by_name=True,
        allow_inf_nan=False,
    )

    lid_model_path: str = Field("/models/lid.176.bin")
    indiclid_ftr_path: str = Field("/models/indiclid-ftr.bin")
    database_url: str = Field("", repr=False, exclude=True)
    redis_url: str = Field("", repr=False, exclude=True)
    essentia_models_dir: str = Field("/data/models/essentia")
    lyrics_model_id: str = Field("BAAI/bge-m3")
    lyrics_revision: str = Field("5617a9f61b028005a4858fdac845db406aefb181")
    audio_cache_bytes: int = Field(2147483648, validation_alias="ECHORA_AUDIO_CACHE_BYTES", ge=0)
    muq_model_id: str = Field("OpenMuQ/MuQ-MuLan-large")
    muq_revision: str = Field("2e01c796b71dca71b45251384c04cd7b237c9020")
    moss_model_id: str = Field("")
    moss_revision: str = Field("")
    hum_diagnostic_dir: str = Field("/models/torch/hum-diagnostics")
    fa_kara_model_id: str = Field("hcX02/echora-mms-300m-multilingual-lyrics-forced-aligner")
    # v8: fine-tuned on separated vocal stems; karaoke uses its lead output.
    fa_kara_revision: str = Field("701595f1be1125cce835d4e32fd432358ea8ed9f")
    # Local checkpoint directory; overrides the model ID snapshot when set.
    fa_kara_model_path: str | None = Field(None)
    fa_kara_audio_speed: float = Field(1.0, ge=0.5, le=1.5)
    fa_kara_aligner: str = Field("yohane")
    # Minimum durations on the audio-only path (0 = off): stop the path cramming words.
    fa_kara_min_letter_ms: int = Field(40, ge=0)
    fa_kara_min_token_ms: int = Field(100, ge=0)
    # Fixed shift of every karaoke time on the audio-only path; calibrated at +45 ms.
    fa_kara_timing_offset_ms: int = Field(45, ge=-500, le=500)
    # Minimum durations follow each line's sung pace (x this ratio, capped by the minimums above),
    # so fast rap isn't stretched; 0 applies the minimums as a fixed floor.
    fa_kara_duration_pace_ratio: float = Field(0.9, ge=0.0, le=5.0)
    # Extra cost per letter frame where the vocal stem is quiet (0: off).
    fa_kara_vocal_activity_weight: float = Field(2.0, ge=0.0, le=20.0)
    # Base image for analysis on Modal. Release images set it to the GPU image of
    # their own version, so remote dependencies match the worker's.
    modal_image: str = Field(
        "ghcr.io/hackslashx/echora-analysis-gpu:latest", validation_alias="ECHORA_MODAL_IMAGE"
    )
    preprocess_dir: str = Field("/data/preprocessed", validation_alias="ECHORA_PREPROCESS_DIR")
    preprocess_max_bytes: int = Field(
        21474836480, validation_alias="ECHORA_PREPROCESS_MAX_BYTES", ge=0
    )
    preprocess_ttl_seconds: float = Field(
        604800.0, validation_alias="ECHORA_PREPROCESS_TTL_SECONDS", gt=0
    )
    batch_size: int = Field(128, validation_alias="ECHORA_BATCH_SIZE", gt=0)
    mert_model_id: str = Field("m-a-p/MERT-v1-95M")
    mert_revision: str = Field("12af15fef9d0ac838c3f475bfbbf26d2060dd4f5")
    recording_calibration_dir: str = Field(
        "/data/recording-calibration", validation_alias="ECHORA_RECORDING_CALIBRATION_DIR"
    )
    recording_calibration_enabled: bool = Field(
        False, validation_alias="ECHORA_RECORDING_CALIBRATION_ENABLED"
    )
    recording_match_policy: str = Field("", validation_alias="ECHORA_RECORDING_MATCH_POLICY")
    source_recheck_seconds: int = Field(0, validation_alias="ECHORA_SOURCE_RECHECK_SECONDS", ge=0)
    semantic_fusion_weight_lyrics: float = Field(0.75, ge=0)
    semantic_fusion_weight_audio: float = Field(0.25, ge=0)
    db_pool_size: int = Field(10, gt=0)
    db_max_overflow: int = Field(20, ge=0)
    oidc_session_secret: str = Field(_SESSION_SECRET, repr=False, exclude=True)
    cookie_secure: bool = Field(False)
    oidc_issuer_url: str = Field("")
    oidc_client_id: str = Field("")
    oidc_client_secret: str = Field("", repr=False, exclude=True)
    oidc_scopes: str = Field("openid profile email")
    oidc_redirect_uri: str = Field("")
    oidc_post_login_redirect: str = Field("http://localhost:3000/home")
    oidc_require_verified_email: bool = Field(False)
    oidc_bootstrap_admin_email: str = Field("")
    credential_encryption_key: str = Field("", repr=False, exclude=True)
    cors_origins: str = Field("", validation_alias="ECHORA_CORS_ORIGINS")
    session_trusted_origins: str = Field("", validation_alias="ECHORA_SESSION_TRUSTED_ORIGINS")
    session_idle_timeout_seconds: int = Field(
        604800, validation_alias="ECHORA_SESSION_IDLE_TIMEOUT_SECONDS", gt=0
    )
    session_absolute_timeout_seconds: int = Field(
        2592000, validation_alias="ECHORA_SESSION_ABSOLUTE_TIMEOUT_SECONDS", gt=0
    )
    session_renew_interval_seconds: int = Field(
        3600, validation_alias="ECHORA_SESSION_RENEW_INTERVAL_SECONDS", gt=0
    )
    session_activity_check_seconds: int = Field(
        60, validation_alias="ECHORA_SESSION_ACTIVITY_CHECK_SECONDS", gt=0
    )
    session_activity_window_seconds: int = Field(
        300, validation_alias="ECHORA_SESSION_ACTIVITY_WINDOW_SECONDS", gt=0
    )
    session_request_timeout_seconds: int = Field(
        10, validation_alias="ECHORA_SESSION_REQUEST_TIMEOUT_SECONDS", gt=0
    )
    worker_poll_seconds: float = Field(2, validation_alias="ECHORA_WORKER_POLL_SECONDS", gt=0)
    worker_lease_seconds: int = Field(120, validation_alias="ECHORA_WORKER_LEASE_SECONDS", ge=15)
    worker_heartbeat_seconds: int = Field(
        10, validation_alias="ECHORA_WORKER_HEARTBEAT_SECONDS", gt=0
    )
    worker_schedule_check_seconds: int = Field(
        30, validation_alias="ECHORA_WORKER_SCHEDULE_CHECK_SECONDS", gt=0
    )
    worker_cleanup_interval_seconds: int = Field(
        60, validation_alias="ECHORA_WORKER_CLEANUP_INTERVAL_SECONDS", gt=0
    )
    worker_shutdown_grace_seconds: int = Field(
        5, validation_alias="ECHORA_WORKER_SHUTDOWN_GRACE_SECONDS", gt=0
    )
    job_retry_base_seconds: int = Field(2, validation_alias="ECHORA_JOB_RETRY_BASE_SECONDS", gt=0)
    job_retry_max_seconds: int = Field(60, validation_alias="ECHORA_JOB_RETRY_MAX_SECONDS", gt=0)
    job_max_attempts: int = Field(3, validation_alias="ECHORA_JOB_MAX_ATTEMPTS", gt=0)
    navidrome_timeout_seconds: int = Field(
        30, validation_alias="ECHORA_NAVIDROME_TIMEOUT_SECONDS", gt=0
    )
    navidrome_read_timeout_seconds: int = Field(
        300, validation_alias="ECHORA_NAVIDROME_READ_TIMEOUT_SECONDS", gt=0
    )
    navidrome_max_connections: int = Field(
        100, validation_alias="ECHORA_NAVIDROME_MAX_CONNECTIONS", gt=0
    )
    navidrome_max_keepalive_connections: int = Field(
        30, validation_alias="ECHORA_NAVIDROME_MAX_KEEPALIVE_CONNECTIONS", gt=0
    )
    navidrome_keepalive_expiry_seconds: int = Field(
        30, validation_alias="ECHORA_NAVIDROME_KEEPALIVE_EXPIRY_SECONDS", gt=0
    )
    media_cache_ttl_seconds: int = Field(
        3600, validation_alias="ECHORA_MEDIA_CACHE_TTL_SECONDS", gt=0
    )
    media_cache_connect_timeout_seconds: float = Field(
        0.5, validation_alias="ECHORA_MEDIA_CACHE_CONNECT_TIMEOUT_SECONDS", gt=0
    )
    media_cache_timeout_seconds: int = Field(
        2, validation_alias="ECHORA_MEDIA_CACHE_TIMEOUT_SECONDS", gt=0
    )
    listening_history_timeout_seconds: int = Field(
        20, validation_alias="ECHORA_LISTENING_HISTORY_TIMEOUT_SECONDS", gt=0
    )
    recording_max_upload_bytes: int = Field(
        8388608, validation_alias="ECHORA_RECORDING_MAX_UPLOAD_BYTES", gt=0
    )
    recording_calibration_max_user_samples: int = Field(
        20, validation_alias="ECHORA_RECORDING_CALIBRATION_MAX_USER_SAMPLES", gt=0
    )
    recording_calibration_max_total_samples: int = Field(
        200, validation_alias="ECHORA_RECORDING_CALIBRATION_MAX_TOTAL_SAMPLES", gt=0
    )
    recording_calibration_uploads_per_minute: int = Field(
        6, validation_alias="ECHORA_RECORDING_CALIBRATION_UPLOADS_PER_MINUTE", gt=0
    )
    recording_model_manifest: str = Field("", validation_alias="ECHORA_RECORDING_MODEL_MANIFEST")
    recording_model_id: str = Field("", validation_alias="ECHORA_RECORDING_MODEL_ID")
    recording_model_revision: str = Field("", validation_alias="ECHORA_RECORDING_MODEL_REVISION")
    recording_model_directory: str = Field(
        "/data/models/recording", validation_alias="ECHORA_RECORDING_MODEL_DIRECTORY"
    )

    recording_audio_retention_seconds: int = Field(
        900, gt=0, validation_alias="ECHORA_RECORDING_AUDIO_RETENTION_SECONDS"
    )
    recording_result_retention_seconds: int = Field(
        86400, gt=0, validation_alias="ECHORA_RECORDING_RESULT_RETENTION_SECONDS"
    )
    recording_diagnostic_retention_seconds: int = Field(
        86400, gt=0, validation_alias="ECHORA_RECORDING_DIAGNOSTIC_RETENTION_SECONDS"
    )
    recording_calibration_retention_seconds: int = Field(
        604800, gt=0, validation_alias="ECHORA_RECORDING_CALIBRATION_RETENTION_SECONDS"
    )
    recording_max_active: int = Field(32, gt=0, validation_alias="ECHORA_RECORDING_MAX_ACTIVE")
    recording_max_active_per_user: int = Field(
        1, gt=0, validation_alias="ECHORA_RECORDING_MAX_ACTIVE_PER_USER"
    )
    recording_uploads_per_minute: int = Field(
        6, gt=0, validation_alias="ECHORA_RECORDING_UPLOADS_PER_MINUTE"
    )

    job_poll_ms: int = Field(1200, gt=0, le=2147483647, validation_alias="ECHORA_JOB_POLL_MS")
    batch_poll_ms: int = Field(1500, gt=0, le=2147483647, validation_alias="ECHORA_BATCH_POLL_MS")
    hum_index_poll_ms: int = Field(
        2500, gt=0, le=2147483647, validation_alias="ECHORA_HUM_INDEX_POLL_MS"
    )
    recording_poll_ms: int = Field(
        1500, gt=0, le=2147483647, validation_alias="ECHORA_RECORDING_POLL_MS"
    )
    recording_search_timeout_ms: int = Field(
        120000, gt=0, le=2147483647, validation_alias="ECHORA_RECORDING_SEARCH_TIMEOUT_MS"
    )
    recording_request_timeout_ms: int = Field(
        10000, gt=0, le=2147483647, validation_alias="ECHORA_RECORDING_REQUEST_TIMEOUT_MS"
    )
    recording_upload_timeout_ms: int = Field(
        60000, gt=0, le=2147483647, validation_alias="ECHORA_RECORDING_UPLOAD_TIMEOUT_MS"
    )

    lastfm_validation_timeout_seconds: float = Field(
        15, gt=0, validation_alias="ECHORA_LASTFM_VALIDATION_TIMEOUT_SECONDS"
    )
    fingerprint_timeout_seconds: float = Field(
        120, gt=0, validation_alias="ECHORA_FINGERPRINT_TIMEOUT_SECONDS"
    )
    karaoke_job_timeout_seconds: float = Field(
        1800, gt=0, validation_alias="ECHORA_KARAOKE_JOB_TIMEOUT_SECONDS"
    )
    karaoke_shutdown_grace_seconds: float = Field(
        10, gt=0, validation_alias="ECHORA_KARAOKE_SHUTDOWN_GRACE_SECONDS"
    )
    karaoke_kill_grace_seconds: float = Field(
        5, gt=0, validation_alias="ECHORA_KARAOKE_KILL_GRACE_SECONDS"
    )
    karaoke_lock_poll_seconds: float = Field(
        0.25, gt=0, validation_alias="ECHORA_KARAOKE_LOCK_POLL_SECONDS"
    )
    preprocess_lock_timeout_seconds: float = Field(
        1800, gt=0, validation_alias="ECHORA_PREPROCESS_LOCK_TIMEOUT_SECONDS"
    )
    preprocess_lock_poll_seconds: float = Field(
        0.05, gt=0, validation_alias="ECHORA_PREPROCESS_LOCK_POLL_SECONDS"
    )
    recording_upload_read_timeout_seconds: float = Field(
        30, gt=0, validation_alias="ECHORA_RECORDING_UPLOAD_READ_TIMEOUT_SECONDS"
    )
    recording_decode_timeout_seconds: float = Field(
        30, gt=0, validation_alias="ECHORA_RECORDING_DECODE_TIMEOUT_SECONDS"
    )
    recording_cancel_poll_seconds: float = Field(
        0.25, gt=0, validation_alias="ECHORA_RECORDING_CANCEL_POLL_SECONDS"
    )
    # Motion artwork videos are generated output, not model files; keep them on their own volume path.
    motion_artwork_dir: str = Field(
        "/data/motion-artwork", validation_alias="ECHORA_MOTION_ARTWORK_DIR"
    )
    # The analysis worker starts this ComfyUI only while a batch renders loops (GPU image only).
    comfyui_dir: str = Field("/opt/comfyui", validation_alias="ECHORA_COMFYUI_DIR")
    comfyui_python: str = Field(
        "/opt/comfyui-venv/bin/python", validation_alias="ECHORA_COMFYUI_PYTHON"
    )
    comfyui_startup_timeout_seconds: float = Field(
        300, gt=0, validation_alias="ECHORA_COMFYUI_STARTUP_TIMEOUT_SECONDS"
    )
    # Blank starts the embedded ComfyUI. An address uses that already running ComfyUI instead;
    # administrators can also set one in Settings.
    comfyui_url: str = Field("", validation_alias="ECHORA_COMFYUI_URL")
    # Opt-in for download_models: the pinned LTX-2.5 and Gemma files are about 44 GB.
    motion_artwork_models: bool = Field(False, validation_alias="ECHORA_MOTION_ARTWORK_MODELS")
    motion_artwork_render_timeout_seconds: float = Field(
        3600, gt=0, validation_alias="ECHORA_MOTION_ARTWORK_RENDER_TIMEOUT_SECONDS"
    )
    motion_artwork_poll_seconds: float = Field(
        2, gt=0, validation_alias="ECHORA_MOTION_ARTWORK_POLL_SECONDS"
    )
    recording_processing_timeout_seconds: float = Field(
        120, gt=0, validation_alias="ECHORA_RECORDING_PROCESSING_TIMEOUT_SECONDS"
    )
    curation_due_limit: int = Field(100, gt=0, validation_alias="ECHORA_CURATION_DUE_LIMIT")
    language_backfill_batch_size: int = Field(
        200, gt=0, validation_alias="ECHORA_LANGUAGE_BACKFILL_BATCH_SIZE"
    )
    lyrics_window_batch_size: int = Field(
        2, gt=0, validation_alias="ECHORA_LYRICS_WINDOW_BATCH_SIZE"
    )
    lyrics_track_batch_size: int = Field(8, gt=0, validation_alias="ECHORA_LYRICS_TRACK_BATCH_SIZE")

    hum_max_workers: int = Field(16, gt=0, validation_alias="ECHORA_HUM_MAX_WORKERS")
    recording_intra_op_threads: int = Field(
        4, gt=0, validation_alias="ECHORA_RECORDING_INTRA_OP_THREADS"
    )
    recording_inter_op_threads: int = Field(
        1, gt=0, validation_alias="ECHORA_RECORDING_INTER_OP_THREADS"
    )

    def public_ui(self) -> dict[str, int]:
        """Explicit browser-safe allowlist; never export the settings model."""
        return {
            "job_poll_ms": self.job_poll_ms,
            "batch_poll_ms": self.batch_poll_ms,
            "hum_index_poll_ms": self.hum_index_poll_ms,
            "recording_poll_ms": self.recording_poll_ms,
            "recording_search_timeout_ms": self.recording_search_timeout_ms,
            "recording_request_timeout_ms": self.recording_request_timeout_ms,
            "recording_upload_timeout_ms": self.recording_upload_timeout_ms,
        }

    @model_validator(mode="after")
    def validate_intervals(self):
        if "*" in self.cors_origins or "*" in self.session_trusted_origins:
            raise ValueError("Credentialed requests require explicit origins, not wildcards")
        if not (
            self.session_activity_check_seconds
            < self.session_renew_interval_seconds
            < self.session_idle_timeout_seconds
            <= self.session_absolute_timeout_seconds
        ):
            raise ValueError("Session intervals require check < renew < idle <= absolute")
        if self.session_activity_window_seconds < self.session_activity_check_seconds:
            raise ValueError("Session activity window must be >= activity check")
        if self.worker_heartbeat_seconds >= self.worker_lease_seconds / 2:
            raise ValueError("Worker heartbeat must be less than half the lease")
        if self.fa_kara_aligner.lower() not in {"yohane", "mms"}:
            raise ValueError("FA_KARA_ALIGNER must be yohane or mms")
        if self.job_retry_base_seconds > self.job_retry_max_seconds:
            raise ValueError("Job retry base must be <= retry maximum")
        if self.semantic_fusion_weight_lyrics + self.semantic_fusion_weight_audio <= 0:
            raise ValueError("Semantic fusion weights must have a positive total")
        if self.navidrome_max_keepalive_connections > self.navidrome_max_connections:
            raise ValueError("Navidrome keepalive connections must not exceed maximum")
        return self


# Settings that change model computation. Analysis on Modal receives the worker's
# values with every call, so remote results match what the worker would compute.
PROCESSING_FIELDS = (
    "lyrics_model_id",
    "lyrics_revision",
    "muq_model_id",
    "muq_revision",
    "mert_model_id",
    "mert_revision",
    "moss_model_id",
    "moss_revision",
    "fa_kara_model_id",
    "fa_kara_revision",
    "fa_kara_audio_speed",
    "fa_kara_aligner",
    "fa_kara_min_letter_ms",
    "fa_kara_min_token_ms",
    "fa_kara_timing_offset_ms",
    "fa_kara_duration_pace_ratio",
    "fa_kara_vocal_activity_weight",
)


# Remote analysis also learns whether motion artwork models belong in its storage, which
# follows Modal's motion artwork switch rather than this worker's own setting.
REMOTE_FIELDS = (*PROCESSING_FIELDS, "motion_artwork_models")


def processing_settings() -> dict[str, object]:
    """The worker's model settings, as sent to remote analysis."""
    values = get_settings().model_dump(include=set(PROCESSING_FIELDS))
    return {name: values[name] for name in PROCESSING_FIELDS}


def adopt_processing_settings(values: dict[str, object]) -> None:
    """On a remote analysis container: compute with the calling worker's settings."""
    import os

    unknown = set(values) - set(REMOTE_FIELDS)
    if unknown:
        raise ValueError(f"Not processing settings: {sorted(unknown)}")
    for name, value in values.items():
        alias = Settings.model_fields[name].validation_alias
        environment_name = alias if isinstance(alias, str) else name.upper()
        os.environ[environment_name] = str(value).lower() if isinstance(value, bool) else str(value)
    get_settings.cache_clear()
    applied = get_settings().model_dump(include=set(values))
    if applied != values:
        raise ValueError("Processing settings did not apply exactly")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load once per process; tests changing environment must clear this cache."""
    return Settings()


def require_database_url() -> str:
    """Return the configured database URL or fail before a driver can use defaults."""
    database_url = get_settings().database_url
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    return database_url


def public_ui_settings() -> dict[str, int]:
    return get_settings().public_ui()
