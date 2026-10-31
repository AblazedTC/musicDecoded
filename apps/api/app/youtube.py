"""YouTube search proxy (PLAN.md D-4): the Data API key stays on the server.

Quota (PLAN.md R-6): search.list costs 100 units of a 10,000/day budget, so results are
cached in Redis for an hour per (query, pageToken).
"""
import hashlib
import html
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Annotated

import redis
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.auth import current_user
from app.errors import ApiError
from app.kv import get_redis
from app.models import User
from app.settings import get_settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/youtube", tags=["youtube"])

API = "https://www.googleapis.com/youtube/v3/"
CACHE_TTL = 3600
MAX_DURATION_SEC = 600  # longer videos can't be decoded (worker's 10-minute limit)
_QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded", "userRateLimitExceeded"}
_ISO_DURATION = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")


class VideoSummary(BaseModel):
    videoId: str
    title: str
    channelTitle: str
    thumbnailUrl: str
    durationSec: int | None


class SearchResponse(BaseModel):
    items: list[VideoSummary]
    nextPageToken: str | None


def _get(endpoint: str, params: dict) -> dict:
    """One Data API call. Tests replace this function, so nothing else touches the network."""
    url = API + endpoint + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.load(resp)


def _call(endpoint: str, params: dict) -> dict:
    try:
        return _get(endpoint, params)
    except urllib.error.HTTPError as e:
        try:
            reasons = {err.get("reason") for err in json.load(e).get("error", {}).get("errors", [])}
        except (ValueError, AttributeError):
            reasons = set()
        if e.code == 429 or (e.code == 403 and reasons & _QUOTA_REASONS):
            raise ApiError(429, "rate_limited", "Search is busy right now, try again later")
        log.error("YouTube %s failed: HTTP %s %s", endpoint, e.code, sorted(r for r in reasons if r))
        raise ApiError(500, "internal", "Search failed")
    except (urllib.error.URLError, TimeoutError) as e:
        log.error("YouTube %s unreachable: %s", endpoint, e)
        raise ApiError(500, "internal", "Search failed")


def parse_duration(value: str | None) -> int | None:
    """ISO-8601 duration (PT4M13S, PT1H2S, P1DT2H) → seconds."""
    m = _ISO_DURATION.match(value or "")
    if not m or not value.strip("PT"):
        return None
    d, h, mi, s = (int(g or 0) for g in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s


def _thumbnail(snippet: dict) -> str:
    thumbs = snippet.get("thumbnails") or {}
    for size in ("high", "medium", "default"):
        if url := (thumbs.get(size) or {}).get("url"):
            return url
    return f"https://i.ytimg.com/vi/{snippet.get('videoId', '')}/hqdefault.jpg"


def _search(key: str, q: str, page_token: str | None) -> SearchResponse:
    params = {"part": "snippet", "type": "video", "videoCategoryId": "10", "maxResults": 20,
              "safeSearch": "moderate", "q": q, "key": key}
    if page_token:
        params["pageToken"] = page_token
    found = _call("search", params)
    hits = [h for h in found.get("items", [])
            if h.get("id", {}).get("videoId") and h.get("snippet", {}).get("liveBroadcastContent", "none") == "none"]
    durations = {}
    if hits:
        details = _call("videos", {"part": "contentDetails", "key": key,
                                   "id": ",".join(h["id"]["videoId"] for h in hits)})
        durations = {v["id"]: parse_duration(v.get("contentDetails", {}).get("duration"))
                     for v in details.get("items", [])}
    items = []
    for h in hits:
        vid, sn = h["id"]["videoId"], h["snippet"]
        dur = durations.get(vid)
        if dur is not None and dur > MAX_DURATION_SEC:
            continue
        items.append(VideoSummary(videoId=vid, title=html.unescape(sn.get("title", "")),
                                  channelTitle=html.unescape(sn.get("channelTitle", "")),
                                  thumbnailUrl=_thumbnail({**sn, "videoId": vid}), durationSec=dur))
    return SearchResponse(items=items, nextPageToken=found.get("nextPageToken"))


def cache_key(q: str, page_token: str | None) -> str:
    return "yt:search:" + hashlib.sha1(f"{q}|{page_token or ''}".encode()).hexdigest()


@router.get("/search")
def search(q: Annotated[str, Query(min_length=1, max_length=200)],
           _: Annotated[User, Depends(current_user)],
           r: Annotated[redis.Redis, Depends(get_redis)],
           pageToken: Annotated[str | None, Query(max_length=200)] = None) -> SearchResponse:
    key = get_settings().youtube_api_key
    if not key:
        raise ApiError(403, "feature_disabled", "Search unavailable")
    q = " ".join(q.split()).lower()  # same results, more cache hits
    if not q:
        raise ApiError(400, "bad_request", "q: must not be blank")
    ck = cache_key(q, pageToken)
    if cached := r.get(ck):
        return SearchResponse.model_validate_json(cached)
    result = _search(key, q, pageToken)
    r.set(ck, result.model_dump_json(), ex=CACHE_TTL)
    return result
