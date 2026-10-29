import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import current_user
from app.config import MAX_UPLOAD_BYTES
from app.db import get_db
from app.errors import ApiError, not_found, parse_id
from app.models import Upload, User
from app.storage import Storage, get_storage

router = APIRouter(prefix="/uploads", tags=["uploads"])

PUT_EXPIRES = 3600  # a 100 MB upload on a slow mobile link needs time to start
GET_EXPIRES = 300


class UploadRequest(BaseModel):
    contentType: Literal["audio/mpeg", "audio/mp4", "audio/aac", "audio/wav", "audio/x-wav", "audio/flac", "audio/ogg"]
    sizeBytes: int = Field(ge=1)  # the upper bound is a 413, not a 400 (below)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class UploadTarget(BaseModel):
    uploadId: str
    putUrl: str
    expiresAt: datetime


class UploadLocation(BaseModel):
    url: str
    contentType: str
    expiresAt: datetime


def _expires(seconds: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_upload(body: UploadRequest, user: Annotated[User, Depends(current_user)],
                  db: Annotated[Session, Depends(get_db)],
                  storage: Annotated[Storage, Depends(get_storage)]) -> UploadTarget:
    if body.sizeBytes > MAX_UPLOAD_BYTES:
        raise ApiError(413, "too_large", f"sizeBytes must be at most {MAX_UPLOAD_BYTES}")
    upload_id = uuid.uuid4()
    key = f"uploads/{upload_id}"
    db.add(Upload(id=upload_id, user_id=user.id, sha256=body.sha256, content_type=body.contentType,
                  size_bytes=body.sizeBytes, object_key=key))
    db.commit()
    return UploadTarget(uploadId=str(upload_id), putUrl=storage.presign_put(key, body.contentType, PUT_EXPIRES),
                        expiresAt=_expires(PUT_EXPIRES))


def owned_upload(db: Session, user: User, upload_id: str) -> Upload:
    """The caller's upload, or 404 (also for another user's upload)."""
    upload = db.scalar(select(Upload).where(Upload.id == parse_id(upload_id, "Upload"), Upload.user_id == user.id))
    if upload is None:
        raise not_found("Upload")
    return upload


@router.get("/{uploadId}")
def get_upload(uploadId: str, user: Annotated[User, Depends(current_user)],
               db: Annotated[Session, Depends(get_db)],
               storage: Annotated[Storage, Depends(get_storage)]) -> UploadLocation:
    upload = owned_upload(db, user, uploadId)
    return UploadLocation(url=storage.presign_get(upload.object_key, GET_EXPIRES), contentType=upload.content_type,
                          expiresAt=_expires(GET_EXPIRES))
