import time
import uuid
from typing import Annotated, Literal

import bcrypt
import jwt
from fastapi import APIRouter, Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.errors import ApiError
from app.models import Job, Upload, User
from app.settings import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])

# Checked against when the email is unknown, so a failed login takes the same time either way.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt())


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)

    @field_validator("password")
    @classmethod
    def _bcrypt_limit(cls, v: str) -> str:
        if len(v.encode()) > 72:
            raise ValueError("must be at most 72 bytes")
        return v


class AuthResponse(BaseModel):
    token: str
    userId: str
    kind: Literal["guest", "account"]
    plan: Literal["free", "pro"]


class AnonymousRequest(BaseModel):
    deviceId: str = Field(min_length=1, max_length=128)
    platform: Literal["ios", "android"]


def create_token(user: User) -> str:
    s = get_settings()
    now = int(time.time())
    claims = {"sub": str(user.id), "kind": user.kind, "plan": user.plan,
              "iat": now, "exp": now + s.jwt_expiration_seconds}
    return jwt.encode(claims, s.jwt_secret, algorithm="HS256")


def _auth_response(user: User) -> AuthResponse:
    return AuthResponse(token=create_token(user), userId=str(user.id), kind=user.kind, plan=user.plan)


@router.post("/anonymous")
def anonymous(body: AnonymousRequest, db: Annotated[Session, Depends(get_db)]) -> AuthResponse:
    """Idempotent per deviceId: the same device always gets the same guest user."""
    find = select(User).where(User.device_id == body.deviceId)
    user = db.scalar(find)
    if user is None:
        user = User(kind="guest", device_id=body.deviceId)
        db.add(user)
        try:
            db.commit()
        except IntegrityError:  # a concurrent call for the same device won the insert
            db.rollback()
            user = db.scalar(find)
    return _auth_response(user)


_bearer = HTTPBearer(auto_error=False)
Bearer = Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]

# Rows a guest owns that move to the account on signup/login (add library rows here in T-402).
_GUEST_OWNED = (Job, Upload)


def _user_from_token(creds: HTTPAuthorizationCredentials | None, db: Session) -> User | None:
    if creds is None:
        return None
    try:
        claims = jwt.decode(creds.credentials, get_settings().jwt_secret, algorithms=["HS256"])
        return db.get(User, uuid.UUID(claims["sub"]))
    except (jwt.PyJWTError, KeyError, ValueError):
        return None


def _link_guest(creds: HTTPAuthorizationCredentials | None, db: Session, account: User) -> None:
    """Move a guest's data to the account. The guest row stays, so the device can still re-auth
    as a (now empty) guest after logout. Any other or invalid bearer is ignored."""
    guest = _user_from_token(creds, db)
    if guest is None or guest.kind != "guest":
        return
    for model in _GUEST_OWNED:
        db.execute(update(model).where(model.user_id == guest.id).values(user_id=account.id))


@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(body: Credentials, creds: Bearer, db: Annotated[Session, Depends(get_db)]) -> AuthResponse:
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise ApiError(409, "conflict", "Email already in use")
    user = User(email=email, password_hash=bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode())
    db.add(user)
    try:
        db.flush()  # get user.id, and hit the unique email constraint before moving any rows
        _link_guest(creds, db, user)
        db.commit()
    except IntegrityError:  # lost a race with a concurrent signup for the same email
        db.rollback()
        raise ApiError(409, "conflict", "Email already in use")
    return _auth_response(user)


@router.post("/login")
def login(body: Credentials, creds: Bearer, db: Annotated[Session, Depends(get_db)]) -> AuthResponse:
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    stored = user.password_hash.encode() if user and user.password_hash else _DUMMY_HASH
    # Same error for unknown email and wrong password, so callers can't probe which emails have accounts.
    if not bcrypt.checkpw(body.password.encode(), stored) or user is None or user.password_hash is None:
        raise ApiError(401, "unauthorized", "Invalid email or password")
    _link_guest(creds, db, user)
    db.commit()
    return _auth_response(user)


def current_user(creds: Bearer, db: Annotated[Session, Depends(get_db)]) -> User:
    """Dependency for every authenticated route (all routes except /auth/* and /health)."""
    if creds is None:
        raise ApiError(401, "unauthorized", "Missing bearer token")
    user = _user_from_token(creds, db)
    if user is None:
        raise ApiError(401, "unauthorized", "Invalid or expired token")
    return user
