import uuid
from datetime import datetime, timezone

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(String(100))
    kind: Mapped[str] = mapped_column(String(10), default="account")  # guest | account (guests arrive in T-101)
    plan: Mapped[str] = mapped_column(String(10), default="free")  # free | pro (PLAN.md §9)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
