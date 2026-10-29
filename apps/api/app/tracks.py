from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.auth import current_user
from app.db import get_db
from app.errors import ApiError, not_found, parse_id
from app.models import Job, Track, User
from app.settings import get_settings
from app.storage import Storage, get_storage

router = APIRouter(prefix="/tracks", tags=["tracks"])

URL_EXPIRES = 300
SCHEMA_VERSION = 1  # contracts/analysis.schema.json


class AnalysisLocation(BaseModel):
    url: str
    schemaVersion: int


class AudioLocation(BaseModel):
    url: str
    contentType: str


def _linked_track(db: Session, user: User, track_id: str) -> Track:
    """The track, if the caller has a job for it (library links arrive with sync, T-402); else 404."""
    tid = parse_id(track_id, "Track")
    linked = exists().where(Job.track_id == tid, Job.user_id == user.id)
    track = db.scalar(select(Track).where(Track.id == tid, linked))
    if track is None:
        raise not_found("Track")
    return track


@router.get("/{trackId}/analysis")
def get_analysis(trackId: str, user: Annotated[User, Depends(current_user)],
                 db: Annotated[Session, Depends(get_db)],
                 storage: Annotated[Storage, Depends(get_storage)]) -> AnalysisLocation:
    track = _linked_track(db, user, trackId)
    if not track.analysis_key:
        raise ApiError(404, "not_found", "Analysis not ready")
    return AnalysisLocation(url=storage.presign_get(track.analysis_key, URL_EXPIRES), schemaVersion=SCHEMA_VERSION)


@router.get("/{trackId}/audio")
def get_audio(trackId: str, user: Annotated[User, Depends(current_user)],
              db: Annotated[Session, Depends(get_db)],
              storage: Annotated[Storage, Depends(get_storage)]) -> AudioLocation:
    if not get_settings().offline_stream_audio:  # PLAN.md R-1
        raise ApiError(403, "feature_disabled", "Offline audio for streaming sources is disabled")
    track = _linked_track(db, user, trackId)
    if not track.audio_key:
        raise ApiError(404, "not_found", "No extracted audio for this track")
    return AudioLocation(url=storage.presign_get(track.audio_key, URL_EXPIRES), contentType="audio/mp4")
