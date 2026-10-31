import json
import re
from datetime import datetime, timezone
from typing import Annotated, Literal
from urllib.parse import urlsplit

import redis
from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import current_user
from app.config import daily_limit
from app.db import get_db
from app.errors import ApiError, not_found, parse_id
from app.kv import get_redis, rate_key
from app.models import Job, Track, User
from app.settings import get_settings
from app.uploads import owned_upload

router = APIRouter(prefix="/jobs", tags=["jobs"])

QUEUE = "jobs:analysis"  # FIFO: API RPUSH, worker BLPOP (docs/data-model.md §4)
JOB_TTL = 86400
RATE_TTL = 2 * 86400
ACTIVE = ("queued", "processing")
STREAMING = ("youtube", "soundcloud")
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_SOUNDCLOUD_HOSTS = {"soundcloud.com", "www.soundcloud.com", "m.soundcloud.com", "on.soundcloud.com"}


class JobRequest(BaseModel):
    source: Literal["file", "recording", "youtube", "soundcloud"]
    uploadId: str | None = None
    videoId: str | None = None
    soundcloudUrl: str | None = Field(default=None, max_length=500)
    title: str | None = Field(default=None, max_length=500)


class JobOut(BaseModel):
    jobId: str
    trackId: str
    status: Literal["queued", "processing", "complete", "failed"]
    stage: str | None
    progress: int
    error: str | None
    title: str | None = None  # track metadata, null until known (the worker fills it in)
    thumbnailUrl: str | None = None
    durationSec: float | None = None


def job_key(job_id) -> str:
    return f"job:{job_id}"


def _job_out(job: Job, track: Track, live: dict | None = None) -> JobOut:
    """Postgres row overlaid with the live Redis hash (empty string = null there), plus track metadata."""
    live = live or {}
    return JobOut(jobId=str(job.id), trackId=str(job.track_id), status=live.get("status") or job.status,
                  stage=live.get("stage") or job.stage or None,
                  progress=int(live.get("progress") or job.progress),
                  error=live.get("error") or job.error or None,
                  title=track.title, thumbnailUrl=track.thumbnail_url, durationSec=track.duration_sec)


def _dedupe_key(body: JobRequest, user: User, db: Session) -> tuple[str, str | None]:
    """→ (dedupe_key, uploadKey). Validates the source-specific fields."""
    if body.source in ("file", "recording"):
        if not body.uploadId:
            raise ApiError(400, "bad_request", f"uploadId is required for source {body.source}")
        upload = owned_upload(db, user, body.uploadId)
        return f"sha256:{upload.sha256}", upload.object_key
    if body.source == "youtube":
        if not body.videoId or not _VIDEO_ID.match(body.videoId):
            raise ApiError(400, "bad_request", "videoId must be an 11-character YouTube video id")
        return f"yt:{body.videoId}", None
    if not get_settings().soundcloud_enabled:
        raise ApiError(403, "feature_disabled", "SoundCloud is not enabled")
    url = urlsplit(body.soundcloudUrl or "")
    # Only SoundCloud hosts: the worker fetches this URL server-side.
    if url.scheme not in ("http", "https") or url.hostname not in _SOUNDCLOUD_HOSTS or len(url.path) < 2:
        raise ApiError(400, "bad_request", "soundcloudUrl must be a SoundCloud track URL")
    return f"sc:https://{url.hostname}{url.path.rstrip('/')}", None


def _take_quota(r: redis.Redis, user: User) -> None:
    key, limit = rate_key(user.id), daily_limit(user)
    used, _ = r.pipeline().incr(key).expire(key, RATE_TTL).execute()
    if limit is not None and used > limit:
        r.decr(key)
        raise ApiError(429, "rate_limited", f"Daily limit of {limit} analyses reached")


@router.post("", status_code=status.HTTP_202_ACCEPTED, responses={200: {"model": JobOut}})
def create_job(body: JobRequest, response: Response, user: Annotated[User, Depends(current_user)],
               db: Annotated[Session, Depends(get_db)], r: Annotated[redis.Redis, Depends(get_redis)]) -> JobOut:
    dedupe, upload_key = _dedupe_key(body, user, db)
    by_key = select(Track).where(Track.dedupe_key == dedupe)
    track = db.scalar(by_key)

    if track is not None and track.analysis_key:  # cache hit: no work, doesn't count toward the limit
        job = Job(track_id=track.id, user_id=user.id, status="complete", progress=100,
                  completed_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        response.status_code = status.HTTP_200_OK
        return _job_out(job, track)

    if track is not None:  # the caller already has this track in flight: return that job
        active = db.scalar(select(Job).where(Job.track_id == track.id, Job.user_id == user.id,
                                             Job.status.in_(ACTIVE)).order_by(Job.created_at.desc()))
        if active is not None:
            return _job_out(active, track, r.hgetall(job_key(active.id)))

    _take_quota(r, user)
    if track is None:
        track = Track(source=body.source, dedupe_key=dedupe,
                      title=body.title if body.source in ("file", "recording") else None)
        db.add(track)
        try:
            db.flush()
        except IntegrityError:  # a concurrent request created the same track
            db.rollback()
            track = db.scalar(by_key)
    job = Job(track_id=track.id, user_id=user.id)
    db.add(job)
    db.commit()

    s = get_settings()
    message = {"jobId": str(job.id), "trackId": str(track.id), "source": body.source, "uploadKey": upload_key,
               "videoId": body.videoId if body.source == "youtube" else None,
               "soundcloudUrl": dedupe.removeprefix("sc:") if body.source == "soundcloud" else None,
               "storeExtractedAudio": s.offline_stream_audio and body.source in STREAMING}
    try:
        (r.pipeline()
         .hset(job_key(job.id), mapping={"status": "queued", "stage": "", "progress": 0, "error": ""})
         .expire(job_key(job.id), JOB_TTL)
         .rpush(QUEUE, json.dumps(message))
         .execute())
    except redis.RedisError:
        job.status, job.error = "failed", "Could not queue the analysis"
        db.commit()
        raise
    return _job_out(job, track)


@router.get("/{jobId}")
def get_job(jobId: str, user: Annotated[User, Depends(current_user)],
            db: Annotated[Session, Depends(get_db)], r: Annotated[redis.Redis, Depends(get_redis)]) -> JobOut:
    job = db.scalar(select(Job).where(Job.id == parse_id(jobId, "Job"), Job.user_id == user.id))
    if job is None:
        raise not_found("Job")
    return _job_out(job, db.get(Track, job.track_id), r.hgetall(job_key(job.id)))
