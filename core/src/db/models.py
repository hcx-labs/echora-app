from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.db.session import Base


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    username: Mapped[str] = mapped_column(Text, unique=True)
    email: Mapped[str | None] = mapped_column(Text)
    display_name: Mapped[str] = mapped_column(Text)
    password_hash: Mapped[str | None] = mapped_column(Text)
    oidc_subject: Mapped[str | None] = mapped_column(Text)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_blocked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    preference: Mapped[UserPreference] = relationship(
        back_populates="user", cascade="all, delete-orphan", uselist=False
    )


class UserPreference(Base):
    __tablename__ = "user_preferences"
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    onboarding_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    navidrome_connection_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("navidrome_connections.id", ondelete="SET NULL")
    )
    timezone: Mapped[str] = mapped_column(Text, default="UTC")
    lastfm_username: Mapped[str | None] = mapped_column(Text)
    lastfm_api_key_encrypted: Mapped[bytes | None]
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user: Mapped[User] = relationship(back_populates="preference")


class UserSession(Base):
    __tablename__ = "user_sessions"
    token_hash: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    absolute_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_renewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user: Mapped[User] = relationship()


class OidcSetting(Base):
    __tablename__ = "oidc_settings"
    singleton: Mapped[bool] = mapped_column(Boolean, primary_key=True, default=True)
    auto_provision: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class OidcAllowedEmail(Base):
    __tablename__ = "oidc_allowed_emails"
    email: Mapped[str] = mapped_column(Text, primary_key=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class NavidromeConnection(Base):
    __tablename__ = "navidrome_connections"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    url: Mapped[str] = mapped_column(Text)
    username: Mapped[str] = mapped_column(Text)
    encrypted_password: Mapped[bytes]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    __table_args__ = (UniqueConstraint("owner_user_id", "url", "username"),)


class Track(Base):
    __tablename__ = "tracks"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    audio_hash: Mapped[str] = mapped_column(Text, unique=True)
    title: Mapped[str] = mapped_column(Text)
    artist: Mapped[str | None] = mapped_column(Text)
    album: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None]
    duration_seconds: Mapped[float] = mapped_column(Float)
    genres: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Curation(Base):
    __tablename__ = "curations"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    navidrome_connection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("navidrome_connections.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(Text)
    curation_type: Mapped[str] = mapped_column(Text, default="language")
    positive_prompt: Mapped[str] = mapped_column(Text)
    negative_prompt: Mapped[str] = mapped_column(Text, default="")
    sound_prompts: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    themes_prompts: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    sound_negative_prompts: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    themes_negative_prompts: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    sound_weight: Mapped[int] = mapped_column(Integer, default=50)
    sound_profile: Mapped[dict] = mapped_column(JSONB, default=dict)
    positive_track_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list
    )
    negative_track_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list
    )
    familiarity_percent: Mapped[int] = mapped_column(Integer, default=70)
    period_start: Mapped[str | None] = mapped_column(Text)
    period_end: Mapped[str | None] = mapped_column(Text)
    time_of_day_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    lookback_days: Mapped[int] = mapped_column(Integer, default=7)
    track_limit: Mapped[int] = mapped_column(Integer, default=30)
    refresh_mode: Mapped[str] = mapped_column(Text, default="stable")
    target_language: Mapped[str] = mapped_column(Text, default="")
    language_strictness: Mapped[str] = mapped_column(Text, default="primarily")
    refresh_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    journey_start_track_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tracks.id", ondelete="SET NULL")
    )
    journey_stop_track_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list
    )
    journey_end_track_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tracks.id", ondelete="SET NULL")
    )
    journey_lyrics_weight: Mapped[int] = mapped_column(Integer, default=25)
    refresh_interval_hours: Mapped[int] = mapped_column(Integer, default=24)
    navidrome_playlist_id: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="draft")
    last_error: Mapped[str | None] = mapped_column(Text)
    last_refreshed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_refresh_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
