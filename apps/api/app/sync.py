"""Library sync for accounts (PLAN.md §5.2): last-write-wins on the client's updatedAt, tombstones.

Pulls are ordered by (server_updated_at, track_id), the server time a row last changed, so an edit
made offline with an old updatedAt is still pulled by devices whose cursor is past that time.
"""
import base64
import binascii
import uuid
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import current_user
from app.db import get_db
from app.errors import ApiError
from app.models import LibraryItem, Track, Upload, User

router = APIRouter(prefix="/sync", tags=["sync"])

PAGE_SIZE = 500  # pull: keep calling with the returned cursor until a page has fewer items
MAX_PUSH = 500


class LibraryItemBody(BaseModel):
    trackId: str
    title: str = Field(max_length=500)
    artist: str | None = Field(default=None, max_length=500)
    source: Literal["file", "recording", "youtube", "soundcloud"]
    sourceRef: str | None = Field(default=None, max_length=600)
    durationSec: float | None = None
    thumbnailUrl: str | None = Field(default=None, max_length=1000)
    audioUploadId: str | None = None
    favorite: bool = False
    lastPlayedAt: datetime | None = None
    createdAt: datetime
    updatedAt: datetime
    deleted: bool


class PullResponse(BaseModel):
    items: list[LibraryItemBody]
    cursor: str


class PushRequest(BaseModel):
    items: list[LibraryItemBody] = Field(max_length=MAX_PUSH)


class PushResponse(BaseModel):
    items: list[LibraryItemBody]


def account_user(user: Annotated[User, Depends(current_user)]) -> User:
    if user.kind != "account":
        raise ApiError(403, "account_required", "Sync needs an account")
    return user


def _utc(dt: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes (stored as UTC); clients may send any offset."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _encode_cursor(row: LibraryItem) -> str:
    raw = f"{_utc(row.server_updated_at).isoformat()}|{row.track_id}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    try:
        at, tid = base64.urlsafe_b64decode(cursor.encode()).decode().split("|")
        return _utc(datetime.fromisoformat(at)), uuid.UUID(tid)
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise ApiError(400, "bad_request", "since: invalid cursor")


def _to_body(row: LibraryItem) -> LibraryItemBody:
    return LibraryItemBody(
        trackId=str(row.track_id), title=row.title, artist=row.artist, source=row.source, sourceRef=row.source_ref,
        durationSec=row.duration_sec, thumbnailUrl=row.thumbnail_url,
        audioUploadId=str(row.audio_upload_id) if row.audio_upload_id else None, favorite=row.favorite,
        lastPlayedAt=_utc(row.last_played_at), createdAt=_utc(row.created_at), updatedAt=_utc(row.updated_at),
        deleted=row.deleted)


@router.get("/library")
def pull(user: Annotated[User, Depends(account_user)], db: Annotated[Session, Depends(get_db)],
         since: str | None = None) -> PullResponse:
    query = select(LibraryItem).where(LibraryItem.user_id == user.id)
    if since:
        at, tid = _decode_cursor(since)
        query = query.where(or_(LibraryItem.server_updated_at > at,
                                and_(LibraryItem.server_updated_at == at, LibraryItem.track_id > tid)))
    rows = db.scalars(query.order_by(LibraryItem.server_updated_at, LibraryItem.track_id).limit(PAGE_SIZE)).all()
    cursor = _encode_cursor(rows[-1]) if rows else (since or "")
    return PullResponse(items=[_to_body(r) for r in rows], cursor=cursor)


def _uuid(value: str, field: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise ApiError(400, "bad_request", f"{field}: not a valid id")


def _validate(db: Session, user: User, items: list[LibraryItemBody]) -> None:
    """Rejects the whole push (400) on an unknown trackId or an audioUploadId the caller doesn't own."""
    track_ids = {_uuid(i.trackId, "trackId") for i in items}
    known = set(db.scalars(select(Track.id).where(Track.id.in_(track_ids)))) if track_ids else set()
    if missing := track_ids - known:
        raise ApiError(400, "bad_request", f"trackId: unknown track {next(iter(missing))}")
    upload_ids = {_uuid(i.audioUploadId, "audioUploadId") for i in items if i.audioUploadId}
    if upload_ids:
        owned = set(db.scalars(select(Upload.id).where(Upload.id.in_(upload_ids), Upload.user_id == user.id)))
        if foreign := upload_ids - owned:
            raise ApiError(400, "bad_request", f"audioUploadId: unknown upload {next(iter(foreign))}")


def _merge(db: Session, user: User, items: list[LibraryItemBody]) -> list[LibraryItem]:
    now = datetime.now(timezone.utc)
    rows: dict[uuid.UUID, LibraryItem] = {}
    for item in items:
        tid = uuid.UUID(item.trackId)
        row = rows.get(tid) or db.get(LibraryItem, (user.id, tid))
        if row is None:
            row = LibraryItem(user_id=user.id, track_id=tid)
            db.add(row)
        elif _utc(item.updatedAt) <= _utc(row.updated_at):
            rows[tid] = row  # the server copy is as new or newer: it wins
            continue
        row.title, row.artist, row.source, row.source_ref = item.title, item.artist, item.source, item.sourceRef
        row.duration_sec, row.thumbnail_url, row.favorite = item.durationSec, item.thumbnailUrl, item.favorite
        row.audio_upload_id = uuid.UUID(item.audioUploadId) if item.audioUploadId else None
        row.last_played_at, row.created_at = _utc(item.lastPlayedAt), _utc(item.createdAt)
        row.updated_at, row.deleted, row.server_updated_at = _utc(item.updatedAt), item.deleted, now
        rows[tid] = row
    db.commit()
    return list(rows.values())


@router.put("/library")
def push(body: PushRequest, user: Annotated[User, Depends(account_user)],
         db: Annotated[Session, Depends(get_db)]) -> PushResponse:
    _validate(db, user, body.items)
    try:
        rows = _merge(db, user, body.items)
    except IntegrityError:  # a concurrent push inserted one of these rows first; merge against it
        db.rollback()
        rows = _merge(db, user, body.items)
    return PushResponse(items=[_to_body(r) for r in rows])
