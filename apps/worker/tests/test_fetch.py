"""services/fetch.py with a fake yt-dlp (no network)."""
from pathlib import Path

import pytest

from services import fetch


class FakeYDL:
    """Mimics yt_dlp.YoutubeDL: extract_info(download=False), then process_ie_result(download=True)."""

    def __init__(self, info=None, failures=None, ext="webm"):
        self.info = info or {"title": "Song", "duration": 180, "thumbnail": "https://x/t.jpg", "ext": ext}
        self.failures = list(failures or [])  # exceptions raised by successive extract_info calls
        self.ext, self.opts, self.extract_calls, self.downloads = ext, None, [], 0

    def __call__(self, opts):  # used as the ydl_factory
        self.opts = opts
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        assert download is False
        self.extract_calls.append(url)
        if self.failures:
            raise self.failures.pop(0)
        return dict(self.info)

    def process_ie_result(self, info, download=True):
        assert download is True
        self.downloads += 1
        Path(self.opts["outtmpl"].replace("%(ext)s", self.ext)).write_bytes(b"audio")
        return info


def run(tmp_path, ydl, url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"):
    sleeps = []
    result = fetch.fetch_stream(url, tmp_path, ydl_factory=ydl, sleep=sleeps.append, backoff=7)
    return result, sleeps


def test_downloads_best_audio_into_dest(tmp_path):
    ydl = FakeYDL()
    (path, info), sleeps = run(tmp_path, ydl)
    assert path == tmp_path / "stream.webm" and path.read_bytes() == b"audio"
    assert info["title"] == "Song" and sleeps == []
    assert ydl.opts["format"] == "bestaudio/best" and ydl.opts["noplaylist"] is True
    assert ydl.opts["outtmpl"].startswith(str(tmp_path))


@pytest.mark.parametrize("info,error", [
    ({"duration": 601}, "This track is longer than 10 minutes"),
    ({"duration": None}, "Could not determine the length of this track"),
    ({"duration": 100, "is_live": True}, "Live streams can't be analyzed"),
    ({"_type": "playlist", "entries": []}, "Playlists are not supported; pick a single track"),
])
def test_rejected_before_download(tmp_path, info, error):
    ydl = FakeYDL(info=info)
    with pytest.raises(fetch.FetchError, match=error):
        run(tmp_path, ydl)
    assert ydl.downloads == 0 and len(ydl.extract_calls) == 1  # no retry for a rejection


def test_one_retry_with_backoff_then_success(tmp_path):
    ydl = FakeYDL(failures=[RuntimeError("HTTP Error 503")])
    (path, _), sleeps = run(tmp_path, ydl)
    assert sleeps == [7] and len(ydl.extract_calls) == 2 and path.exists()


@pytest.mark.parametrize("message,readable", [
    ("ERROR: [youtube] abc: Sign in to confirm you’re not a bot. Use --cookies",
     "The streaming site blocked the download (bot check). Please try again later."),
    ("ERROR: [youtube] abc: Video unavailable", "This track is unavailable (private, removed or region-locked)."),
    ("boom", "Could not fetch the audio from the streaming site."),
])
def test_fails_readably_after_one_retry(tmp_path, message, readable):
    ydl = FakeYDL(failures=[RuntimeError(message), RuntimeError(message)])
    with pytest.raises(fetch.FetchError) as exc:
        run(tmp_path, ydl)
    assert str(exc.value) == readable
    assert len(ydl.extract_calls) == 2 and ydl.downloads == 0


def test_source_url():
    assert fetch.source_url({"source": "youtube", "videoId": "dQw4w9WgXcQ"}) == (
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ")
    url = "https://on.soundcloud.com/abc"
    assert fetch.source_url({"source": "soundcloud", "soundcloudUrl": url}) == (url, url)
    for bad in ("http://soundcloud.com/a", "https://notsoundcloud.com/a", None):
        with pytest.raises(fetch.FetchError):
            fetch.source_url({"source": "soundcloud", "soundcloudUrl": bad})


def test_to_m4a_copies_m4a_without_ffmpeg(tmp_path):
    src = tmp_path / "stream.m4a"
    src.write_bytes(b"aac")
    fetch.to_m4a(src, tmp_path / "out.m4a")
    assert (tmp_path / "out.m4a").read_bytes() == b"aac"
