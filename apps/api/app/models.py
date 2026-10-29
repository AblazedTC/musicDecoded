import uuid
from datetime import datetime, timezone

from sqlalchemy import String
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
