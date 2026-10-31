"""RevenueCat webhook → users.plan (PLAN.md §9, D-10), and the `require_pro` guard for Pro-only routes."""
import hmac
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import current_user
from app.db import get_db
from app.errors import ApiError, not_found
from app.models import BillingEvent, User, as_utc, effective_plan
from app.settings import get_settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/billing", tags=["billing"])

# Pro until the event's expiration_at_ms (null = non-expiring).
_GRANTS = {"INITIAL_PURCHASE", "RENEWAL", "PRODUCT_CHANGE", "UNCANCELLATION"}


class RevenueCatEvent(BaseModel):
    model_config = ConfigDict(extra="allow")  # RevenueCat sends many more fields; we read only these
    id: str = Field(min_length=1, max_length=100)
    type: str = Field(max_length=40)
    app_user_id: str = Field(max_length=200)
    expiration_at_ms: int | None = None
    grace_period_expiration_at_ms: int | None = None


class RevenueCatWebhook(BaseModel):
    model_config = ConfigDict(extra="allow")
    api_version: str | None = None
    event: RevenueCatEvent


def _webhook_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    """Runs before the body is parsed, so a caller without the secret learns nothing about the payload."""
    secret = get_settings().revenuecat_webhook_secret
    if not secret:
        raise not_found("Billing")  # not configured on this server: the endpoint doesn't exist
    if not hmac.compare_digest((authorization or "").encode(), f"Bearer {secret}".encode()):
        raise ApiError(401, "unauthorized", "Invalid webhook secret")


def _from_ms(ms: int | None) -> datetime | None:
    return None if ms is None else datetime.fromtimestamp(ms / 1000, timezone.utc)


def _find_user(db: Session, app_user_id: str) -> User | None:
    try:
        return db.get(User, uuid.UUID(app_user_id))
    except ValueError:  # e.g. a RevenueCat anonymous id ($RCAnonymousID:...)
        return None


def _apply(user: User, ev: RevenueCatEvent) -> None:
    expires = _from_ms(ev.expiration_at_ms)
    current = as_utc(user.plan_expires_at) if user.plan_expires_at else None
    if ev.type in _GRANTS:
        # Deliveries can arrive out of order: never shorten a later expiry already granted.
        if user.plan == "pro" and current and expires and current > expires:
            expires = current
        user.plan, user.plan_expires_at = "pro", expires
    elif ev.type == "EXPIRATION":
        if user.plan == "pro" and current and expires and current > expires:
            log.info("billing: stale EXPIRATION for user %s ignored (renewed since)", user.id)
            return
        user.plan, user.plan_expires_at = "free", None
    elif ev.type == "BILLING_ISSUE":
        grace = _from_ms(ev.grace_period_expiration_at_ms)
        if user.plan == "pro" and current and grace and grace > current:
            user.plan_expires_at = grace  # keep Pro through the store's grace period
        log.warning("billing: BILLING_ISSUE for user %s, Pro until %s", user.id, user.plan_expires_at)
    elif ev.type == "CANCELLATION":
        log.info("billing: CANCELLATION for user %s, Pro until %s", user.id, user.plan_expires_at)
    else:
        log.info("billing: event type %s ignored", ev.type)


@router.post("/revenuecat", dependencies=[Depends(_webhook_auth)])
def revenuecat(body: RevenueCatWebhook, db: Annotated[Session, Depends(get_db)]) -> Response:
    """Always 200 once authenticated: a non-2xx makes RevenueCat retry, which can't fix an unknown
    user or an unhandled type. Every handled id is recorded, so redeliveries are no-ops."""
    ev = body.event
    if db.get(BillingEvent, ev.id) is not None:
        return Response(status_code=200)
    user = _find_user(db, ev.app_user_id)
    if user is None:
        log.warning("billing: %s event %s for unknown app_user_id ignored", ev.type, ev.id)
    else:
        _apply(user, ev)
    db.add(BillingEvent(id=ev.id, type=ev.type, app_user_id=ev.app_user_id))
    try:
        db.commit()
    except IntegrityError:  # a concurrent delivery of the same event won
        db.rollback()
    return Response(status_code=200)


def require_pro(user: Annotated[User, Depends(current_user)]) -> User:
    """Dependency for Pro-only routes: free (or expired Pro) users get 403 `pro_required`."""
    if effective_plan(user) != "pro":
        raise ApiError(403, "pro_required", "This feature requires MusicDecoded Pro")
    return user
