"""services/consumer.py with fake Redis, S3, DB and analyzer (no torch, librosa or services needed)."""
import io
import json
from pathlib import Path

import pytest

from services import consumer as c

FIXTURE = c.CONTRACTS_DIR / "fixtures" / "sample.analysis.json"
OPENAPI_STAGES = ["fetching_media", "normalizing_audio", "detecting_beats", "detecting_chords", "detecting_key",
                  "rendering_spectrogram", "saving_results"]


class FakeRedis:
    def __init__(self):
        self.hashes, self.history = {}, []

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update({k: str(v) for k, v in mapping.items()})
        self.history.append(dict(self.hashes[key]))

    def expire(self, key, ttl):
        assert ttl == c.JOB_TTL


class FakeS3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.content_types = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[Key], self.content_types[Key] = Body, ContentType


class FakeJobs:
    def __init__(self):
        self.job = {"status": "queued", "stage": None, "progress": 0, "error": None}
        self.track = {}

    def save_source(self, track_id, title, duration_sec, thumbnail_url):
        self.track.update(title=title, source_duration=duration_sec, thumbnail_url=thumbnail_url)

    def save_audio_key(self, track_id, audio_key):
        self.track.update(audio_key=audio_key)

    def start(self, job_id):
        self.job.update(status="processing", started=True)

    def stage(self, job_id, stage, progress):
        self.job.update(stage=stage, progress=progress)

    def save_track(self, track_id, analysis_key, analyzer_version, duration_sec):
        self.track.update(analysis_key=analysis_key, analyzer_version=analyzer_version, duration_sec=duration_sec)

    def complete(self, job_id):
        self.job.update(status="complete", stage=None, progress=100)

    def fail(self, job_id, error):
        self.job.update(status="failed", error=error)


class FakeAnalyzer:
    """Emits the real analyzer's stage names; returns the contracts fixture (or a broken doc)."""

    def __init__(self, doc=None, raise_at=None):
        self.doc, self.raise_at, self.seen_paths, self.timings = doc, raise_at, [], {}

    def analyze(self, audio_path, *, track_id, source, source_ref, on_stage):
        self.seen_paths.append(Path(audio_path))
        assert Path(audio_path).read_bytes() == b"wav"
        for name in ("load", "beats", "chords", "key", "spectrogram"):
            on_stage(name)
            self.timings[name] = 0.01
            if name == self.raise_at:
                raise RuntimeError("model exploded at C:\\secret\\path")
        doc = json.loads(json.dumps(self.doc or json.loads(FIXTURE.read_text())))
        doc["meta"].update(trackId=track_id, source=source, sourceRef=source_ref)
        return doc


def fake_normalize(src, dst):
    assert src.read_bytes() in (b"mp3-bytes", b"stream-bytes")
    dst.write_bytes(b"wav")
    return "fake"


@pytest.fixture
def validate():
    pytest.importorskip("jsonschema")
    return c.load_validator()


class FakeFetch:
    """Stands in for fetch.fetch_stream: writes stream.<ext> into the job's temp dir."""

    def __init__(self, ext="m4a", error=None):
        self.ext, self.error, self.calls = ext, error, []

    def __call__(self, url, dest_dir):
        self.calls.append((url, Path(dest_dir)))
        if self.error:
            raise c.fetch.FetchError(self.error)
        path = Path(dest_dir) / f"stream.{self.ext}"
        path.write_bytes(b"stream-bytes")
        return path, {"title": "Song", "duration": 200.0, "thumbnail": "https://i.ytimg.com/x.jpg"}


def fake_to_m4a(src, dst):
    dst.write_bytes(b"m4a:" + src.read_bytes())


def make(validate, analyzer=None, objects=None, fetch=None, store_stream_audio=False):
    redis, s3, jobs = FakeRedis(), FakeS3({"uploads/u1": b"mp3-bytes"} if objects is None else objects), FakeJobs()
    analyzer = analyzer or FakeAnalyzer()
    worker = c.Consumer(analyzer=analyzer, redis=redis, s3=s3, bucket="b", jobs=jobs, validate=validate,
                        normalize=fake_normalize, fetch_stream=fetch or FakeFetch(), to_m4a=fake_to_m4a,
                        store_stream_audio=store_stream_audio)
    return worker, redis, s3, jobs, analyzer


def message(source="file", upload_key="uploads/u1", video_id=None, soundcloud_url=None):
    return json.dumps({"jobId": "j1", "trackId": "t1", "source": source, "uploadKey": upload_key,
                       "videoId": video_id, "soundcloudUrl": soundcloud_url, "storeExtractedAudio": False})


def test_success(validate):
    worker, redis, s3, jobs, analyzer = make(validate)
    worker.handle(message())

    assert jobs.job == {"status": "complete", "stage": None, "progress": 100, "error": None, "started": True}
    assert jobs.track["analysis_key"] == "analysis/t1.json" and jobs.track["duration_sec"] > 0
    assert s3.content_types["analysis/t1.json"] == "application/json"
    saved = json.loads(s3.objects["analysis/t1.json"])
    assert saved["meta"]["trackId"] == "t1" and validate(saved) == []
    assert redis.hashes["job:j1"] == {"status": "complete", "stage": "", "progress": "100", "error": ""}
    stages = [h["stage"] for h in redis.history if h["stage"]]
    assert list(dict.fromkeys(stages)) == OPENAPI_STAGES  # every stage, in order, from the openapi enum
    progress = [int(h["progress"]) for h in redis.history]
    assert progress == sorted(progress)  # never goes backwards
    assert not analyzer.seen_paths[0].parent.exists()  # temp media deleted on success


def test_invalid_output_fails_the_job(validate):
    worker, redis, s3, jobs, _ = make(validate, FakeAnalyzer(doc={"schemaVersion": 1, "meta": {}}))
    worker.handle(message())

    assert jobs.job["status"] == "failed" and jobs.job["error"] == "The analysis did not pass validation"
    assert redis.hashes["job:j1"]["status"] == "failed"
    assert "analysis/t1.json" not in s3.objects and jobs.track == {}


def test_exception_cleans_up_and_hides_details(validate):
    worker, redis, s3, jobs, analyzer = make(validate, FakeAnalyzer(raise_at="chords"))
    worker.handle(message())

    assert jobs.job["status"] == "failed"
    assert jobs.job["error"] == "Analysis failed while detecting chords"  # no exception text / paths
    assert redis.hashes["job:j1"] == {"status": "failed", "stage": "detecting_chords",
                                      "progress": str(c.PROGRESS["detecting_chords"]),
                                      "error": "Analysis failed while detecting chords"}
    assert not analyzer.seen_paths[0].parent.exists()  # temp dir deleted on failure


def test_missing_upload(validate):
    worker, redis, _, jobs, _ = make(validate, objects={})
    worker.handle(message())
    assert jobs.job["error"] == "Could not download the uploaded audio"
    assert redis.hashes["job:j1"]["stage"] == "fetching_media"


STREAMS = [("youtube", dict(video_id="dQw4w9WgXcQ"), "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
           ("soundcloud", dict(soundcloud_url="https://soundcloud.com/a/b"), "https://soundcloud.com/a/b",
            "https://soundcloud.com/a/b")]


@pytest.mark.parametrize("source,ids,url,ref", STREAMS)
def test_streaming_source(validate, source, ids, url, ref):
    fetch = FakeFetch()
    worker, redis, s3, jobs, analyzer = make(validate, fetch=fetch)
    worker.handle(message(source=source, upload_key=None, **ids))

    assert jobs.job["status"] == "complete"
    assert fetch.calls[0][0] == url
    assert jobs.track["title"] == "Song" and jobs.track["thumbnail_url"].startswith("https://")
    saved = json.loads(s3.objects["analysis/t1.json"])
    assert saved["meta"]["source"] == source and saved["meta"]["sourceRef"] == ref and validate(saved) == []
    assert "audio_key" not in jobs.track and not any(k.startswith("extracted-audio/") for k in s3.objects)
    assert not fetch.calls[0][1].exists()  # fetched audio deleted with the temp dir


@pytest.mark.parametrize("ext", ["m4a", "webm"])
def test_offline_stream_audio_is_stored(validate, ext):
    worker, _, s3, jobs, _ = make(validate, fetch=FakeFetch(ext=ext), store_stream_audio=True)
    worker.handle(message(source="youtube", upload_key=None, video_id="dQw4w9WgXcQ"))

    assert jobs.job["status"] == "complete"
    assert jobs.track["audio_key"] == "extracted-audio/t1.m4a"
    assert s3.objects["extracted-audio/t1.m4a"] == b"m4a:stream-bytes"
    assert s3.content_types["extracted-audio/t1.m4a"] == "audio/mp4"


def test_fetch_error_is_readable(validate):
    worker, redis, _, jobs, analyzer = make(validate, fetch=FakeFetch(error="This track is longer than 10 minutes"))
    worker.handle(message(source="youtube", upload_key=None, video_id="dQw4w9WgXcQ"))

    assert jobs.job["status"] == "failed" and jobs.job["error"] == "This track is longer than 10 minutes"
    assert redis.hashes["job:j1"]["stage"] == "fetching_media" and analyzer.seen_paths == []


@pytest.mark.parametrize("ids", [dict(video_id="../etc"), dict(video_id=None)])
def test_bad_video_id(validate, ids):
    worker, _, _, jobs, _ = make(validate)
    worker.handle(message(source="youtube", upload_key=None, **ids))
    assert jobs.job["error"] == "Invalid YouTube video id"


def test_bad_soundcloud_url(validate):
    worker, _, _, jobs, _ = make(validate)
    worker.handle(message(source="soundcloud", upload_key=None, soundcloud_url="https://evil.example/soundcloud.com"))
    assert jobs.job["error"] == "Invalid SoundCloud URL"


def test_malformed_message_is_dropped(validate):
    worker, redis, _, jobs, _ = make(validate)
    worker.handle("not json")
    worker.handle(json.dumps({"jobId": "j1"}))
    assert redis.hashes == {} and jobs.job["status"] == "queued"


def test_stage_maps_cover_openapi_enum():
    assert list(c.PROGRESS) == OPENAPI_STAGES
    assert set(c.ANALYZER_STAGES.values()) <= set(OPENAPI_STAGES)
