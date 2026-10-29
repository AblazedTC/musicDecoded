"""Redis: the analysis queue, live job status, and daily rate counters (docs/data-model.md §4)."""
from datetime import datetime, timezone
from functools import lru_cache

import redis

from app.settings import get_settings


@lru_cache
def _client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def get_redis() -> redis.Redis:
    """FastAPI dependency (tests override it with fakeredis)."""
    if not get_settings().redis_url:
        raise RuntimeError("REDIS_URL is not set, or run with APP_ENV=dev for local development.")
    return _client()


def rate_key(user_id) -> str:
    return f"rate:{user_id}:{datetime.now(timezone.utc):%Y%m%d}"
