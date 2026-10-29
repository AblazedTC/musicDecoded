import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(10), default="account")  # guest | account
    plan: Mapped[str] = mapped_column(String(10), default="free")  # free | pro (PLAN.md §9)
    plan_expires_at: Mapped[datetime | None]
    device_id: Mapped[str | None] = mapped_column(String(128), unique=True)  # guests only
    email: Mapped[str | None] = mapped_column(String(320), unique=True)  # accounts only
    password_hash: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(default=_now)


class Upload(Base):
    """User-owned audio (file or recording) at object_key = uploads/{id}."""
    __tablename__ = "uploads"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    sha256: Mapped[str] = mapped_column(String(64))
    content_type: Mapped[str] = mapped_column(String(50))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    object_key: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(default=_now)


class Track(Base):
    """One song, global across users. Analysis is shared by dedupe_key (PLAN.md §5 rule 3)."""
    __tablename__ = "tracks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(20))  # file | recording | youtube | soundcloud
    dedupe_key: Mapped[str] = mapped_column(String(600), unique=True)  # sha256:<hex> | yt:<videoId> | sc:<url>
    title: Mapped[str | None] = mapped_column(String(500))
    duration_sec: Mapped[float | None]
    thumbnail_url: Mapped[str | None] = mapped_column(String(1000))
    analysis_key: Mapped[str | None] = mapped_column(String(200))  # set by the worker when analysis is saved
    audio_key: Mapped[str | None] = mapped_column(String(200))  # extracted-audio/..., only with OFFLINE_STREAM_AUDIO
    analyzer_version: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(default=_now)


class Job(Base):
    """Durable job record; live progress is mirrored in Redis job:{id}."""
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    track_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tracks.id"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued | processing | complete | failed
    stage: Mapped[str | None] = mapped_column(String(40))
    progress: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(default=_now)
    started_at: Mapped[datetime | None]
    completed_at: Mapped[datetime | None]
