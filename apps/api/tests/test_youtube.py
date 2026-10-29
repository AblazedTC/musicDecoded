import io
import json
import urllib.error

import pytest

from app import youtube
from app.settings import get_settings


def _hit(vid, title="Song", live="none"):
    return {"id": {"kind": "youtube#video", "videoId": vid},
            "snippet": {"title": title, "channelTitle": "Chan &amp; Co", "liveBroadcastContent": live,
                        "thumbnails": {"default": {"url": f"https://i.ytimg.com/vi/{vid}/default.jpg"},
                                       "high": {"url": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"}}}}


class FakeYouTube:
    """Replaces youtube._get: canned search/videos responses, records every call."""

    def __init__(self, pages=None, durations=None, error=None):
        self.calls = []
        self.pages = pages or {None: ([_hit("aaaaaaaaaaa", "Don&#39;t Stop"), _hit("bbbbbbbbbbb")], "NEXT")}
        self.durations = durations or {"aaaaaaaaaaa": "PT3M30S", "bbbbbbbbbbb": "PT1H2M"}
        self.error = error

    def __call__(self, endpoint, params):
        self.calls.append((endpoint, dict(params)))
        if self.error:
            raise self.error
        if endpoint == "search":
            items, token = self.pages[params.get("pageToken")]
            return {"items": items, "nextPageToken": token}
        ids = params["id"].split(",")
        return {"items": [{"id": i, "contentDetails": {"duration": self.durations[i]}}
                          for i in ids if i in self.durations]}


@pytest.fixture
def yt(monkeypatch):
    monkeypatch.setattr(get_settings(), "youtube_api_key", "test-key")
    fake = FakeYouTube()
    monkeypatch.setattr(youtube, "_get", fake)
    return fake


def search(client, headers, **params):
    return client.get("/youtube/search", params=params, headers=headers)


def test_maps_results_and_drops_long_videos(client, guest, yt):
    r = search(client, guest[0], q="beatles")
    assert r.status_code == 200
    assert r.json() == {"nextPageToken": "NEXT", "items": [{
        "videoId": "aaaaaaaaaaa", "title": "Don't Stop", "channelTitle": "Chan & Co",
        "thumbnailUrl": "https://i.ytimg.com/vi/aaaaaaaaaaa/hqdefault.jpg", "durationSec": 210}]}
    (ep1, search_params), (ep2, video_params) = yt.calls
    assert ep1 == "search" and search_params == {
        "part": "snippet", "type": "video", "videoCategoryId": "10", "maxResults": 20,
        "safeSearch": "moderate", "q": "beatles", "key": "test-key"}
    assert ep2 == "videos" and video_params["part"] == "contentDetails"
    assert video_params["id"] == "aaaaaaaaaaa,bbbbbbbbbbb"


def test_skips_live_broadcasts_and_keeps_unknown_durations(client, guest, yt):
    yt.pages = {None: ([_hit("ccccccccccc", live="live"), _hit("ddddddddddd")], None)}
    yt.durations = {}
    body = search(client, guest[0], q="x").json()
    assert body == {"nextPageToken": None, "items": [{
        "videoId": "ddddddddddd", "title": "Song", "channelTitle": "Chan & Co",
        "thumbnailUrl": "https://i.ytimg.com/vi/ddddddddddd/hqdefault.jpg", "durationSec": None}]}


def test_caches_per_query_and_page(client, guest, yt, redis_fake):
    first = search(client, guest[0], q="Beatles ").json()
    assert search(client, guest[0], q="beatles").json() == first  # normalized → same cache entry
    assert len(yt.calls) == 2  # one search.list + one videos.list, then cache hits
    key = youtube.cache_key("beatles", None)
    assert 0 < redis_fake.ttl(key) <= 3600


def test_pagination_passes_page_token(client, guest, yt):
    yt.pages["NEXT"] = ([_hit("eeeeeeeeeee")], None)
    yt.durations["eeeeeeeeeee"] = "PT45S"
    body = search(client, guest[0], q="beatles", pageToken="NEXT").json()
    assert body["nextPageToken"] is None and [i["videoId"] for i in body["items"]] == ["eeeeeeeeeee"]
    assert yt.calls[0][1]["pageToken"] == "NEXT"
    search(client, guest[0], q="beatles")  # page 1 is a different cache entry
    assert len(yt.calls) == 4


def _http_error(code, reason):
    body = json.dumps({"error": {"code": code, "errors": [{"reason": reason}]}}).encode()
    return urllib.error.HTTPError("https://www.googleapis.com", code, "err", {}, io.BytesIO(body))


def test_quota_exceeded_is_rate_limited(client, guest, yt, redis_fake):
    yt.error = _http_error(403, "quotaExceeded")
    r = search(client, guest[0], q="beatles")
    assert r.status_code == 429 and r.json()["code"] == "rate_limited"
    assert redis_fake.get(youtube.cache_key("beatles", None)) is None  # errors aren't cached


def test_other_upstream_errors_are_internal(client, guest, yt):
    yt.error = _http_error(400, "keyInvalid")
    r = search(client, guest[0], q="beatles")
    assert r.status_code == 500 and r.json() == {"code": "internal", "message": "Search failed"}


def test_network_errors_are_internal(client, guest, yt):
    yt.error = urllib.error.URLError("timed out")
    r = search(client, guest[0], q="beatles")
    assert r.status_code == 500 and r.json()["code"] == "internal"


def test_missing_api_key_disables_search(client, guest, monkeypatch):
    monkeypatch.setattr(get_settings(), "youtube_api_key", "")
    monkeypatch.setattr(youtube, "_get", lambda *a: pytest.fail("must not call YouTube"))
    r = search(client, guest[0], q="beatles")
    assert r.status_code == 403 and r.json() == {"code": "feature_disabled", "message": "Search unavailable"}


def test_requires_auth_and_valid_query(client, guest, yt):
    assert search(client, {}, q="beatles").status_code == 401
    for q in ("", "x" * 201, "   "):
        r = search(client, guest[0], q=q)
        assert r.status_code == 400 and r.json()["code"] == "bad_request"
    assert yt.calls == []


@pytest.mark.parametrize("value,expected", [
    ("PT4M13S", 253), ("PT1H2S", 3602), ("PT45S", 45), ("P1DT2H", 93600), ("P0D", 0),
    ("PT", None), ("", None), (None, None), ("garbage", None)])
def test_parse_duration(value, expected):
    assert youtube.parse_duration(value) == expected
