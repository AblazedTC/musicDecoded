from typing import Annotated, Literal

import redis
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.auth import current_user
from app.kv import get_redis, rate_key
from app.models import User, effective_plan
from app.settings import get_settings

router = APIRouter(tags=["config"])

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # openapi /uploads sizeBytes maximum


class Limits(BaseModel):
    dailyJobs: int | None
    jobsUsedToday: int


class ClientConfig(BaseModel):
    offlineStreamAudio: bool
    soundcloudEnabled: bool
    maxUploadBytes: int
    plan: Literal["free", "pro"]
    limits: Limits


def daily_limit(user: User) -> int | None:
    """Jobs per UTC day; None = unlimited. Always from the effective plan, never from a client claim."""
    s = get_settings()
    if effective_plan(user) == "pro":
        return s.limit_pro
    return s.limit_guest if user.kind == "guest" else s.limit_free


def jobs_used_today(r: redis.Redis, user: User) -> int:
    return int(r.get(rate_key(user.id)) or 0)


@router.get("/config")
def config(user: Annotated[User, Depends(current_user)],
           r: Annotated[redis.Redis, Depends(get_redis)]) -> ClientConfig:
    s = get_settings()
    return ClientConfig(offlineStreamAudio=s.offline_stream_audio, soundcloudEnabled=s.soundcloud_enabled,
                        maxUploadBytes=MAX_UPLOAD_BYTES, plan=effective_plan(user),
                        limits=Limits(dailyJobs=daily_limit(user), jobsUsedToday=jobs_used_today(r, user)))
