"""Streaming sources (T-302): yt-dlp as a library (PLAN.md D-8), audio only.

fetch_stream(url, dest_dir) checks the duration with extract_info(download=False) first, then downloads
`bestaudio/best` into dest_dir. Extraction errors get one retry with backoff (PLAN.md R-8), then a
FetchError whose message is safe to show to users.
"""
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

MAX_DURATION_SEC = 600
RETRY_BACKOFF_SEC = 5.0
YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")

log = logging.getLogger("fetch")


class FetchError(Exception):
    """A fetch failure with a readable message for clients."""


def source_url(msg: Dict[str, Any]) -> Tuple[str, str]:
    """Queue message -> (url to fetch, analysis meta.sourceRef)."""
    if msg["source"] == "youtube":
        video_id = msg.get("videoId") or ""
        if not YOUTUBE_ID.match(video_id):
            raise FetchError("Invalid YouTube video id")
        return f"https://www.youtube.com/watch?v={video_id}", video_id
    url = msg.get("soundcloudUrl") or ""
    host = (urlparse(url).hostname or "").lower()
    if urlparse(url).scheme != "https" or not (host == "soundcloud.com" or host.endswith(".soundcloud.com")):
        raise FetchError("Invalid SoundCloud URL")
    return url, url


def _readable(err: Exception) -> str:
    text = str(err)
    if "confirm you" in text and "bot" in text:
        return "The streaming site blocked the download (bot check). Please try again later."
    if any(s in text for s in ("Private video", "Video unavailable", "This video is unavailable", "404")):
        return "This track is unavailable (private, removed or region-locked)."
    if "age" in text.lower() and "sign in" in text.lower():
        return "This track is age-restricted and can't be fetched."
    return "Could not fetch the audio from the streaming site."


def _ydl_factory(opts: Dict[str, Any]):
    import yt_dlp

    return yt_dlp.YoutubeDL(opts)


def fetch_stream(url: str, dest_dir: Path, *, ydl_factory: Callable[[Dict[str, Any]], Any] = _ydl_factory,
                 sleep: Callable[[float], None] = time.sleep,
                 backoff: float = RETRY_BACKOFF_SEC) -> Tuple[Path, Dict[str, Any]]:
    """Download the audio of url into dest_dir. Returns (file path, yt-dlp info dict)."""
    opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": str(Path(dest_dir) / "stream.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
    }
    last: Optional[Exception] = None
    for attempt in (1, 2):
        try:
            with ydl_factory(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info.get("_type") == "playlist" or info.get("entries") is not None:
                    raise FetchError("Playlists are not supported; pick a single track")
                if info.get("is_live"):
                    raise FetchError("Live streams can't be analyzed")
                duration = info.get("duration")
                if not duration:
                    raise FetchError("Could not determine the length of this track")
                if duration > MAX_DURATION_SEC:
                    raise FetchError(f"This track is longer than {MAX_DURATION_SEC // 60} minutes")
                ydl.process_ie_result(info, download=True)
            files = [p for p in Path(dest_dir).glob("stream.*") if not p.name.endswith(".part")]
            if not files:
                raise FetchError("Could not fetch the audio from the streaming site.")
            return files[0], info
        except FetchError:
            raise
        except Exception as e:  # yt_dlp.utils.DownloadError / ExtractorError, network errors
            last = e
            log.warning("yt-dlp attempt %d for %s failed: %s", attempt, url, e)
            if attempt == 1:
                sleep(backoff)
    raise FetchError(_readable(last)) from last


def to_m4a(src: Path, dst: Path) -> None:
    """Remux/transcode to AAC in an .m4a container (for extracted-audio/{trackId}.m4a)."""
    if src.suffix.lower() == ".m4a":
        shutil.copyfile(src, dst)
        return
    if not shutil.which("ffmpeg"):
        raise FetchError("ffmpeg is required to store extracted audio")
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                           "-vn", "-c:a", "aac", "-b:a", "160k", str(dst)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
    if proc.returncode != 0:
        log.warning("ffmpeg m4a transcode failed: %s", proc.stderr.decode(errors="replace")[-500:])
        raise FetchError("Could not store the extracted audio")
