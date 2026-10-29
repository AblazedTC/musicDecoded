"""Queue consumer (T-113): BLPOP jobs:analysis -> fetch -> normalize -> analyze -> validate -> save.

Run: python services/consumer.py   (env: REDIS_URL, DATABASE_URL, S3_*, CHORD_MODEL_DIR, OFFLINE_STREAM_AUDIO)

Uploads (file/recording) come from object storage; youtube/soundcloud are fetched with yt-dlp
(services/fetch.py, T-302).

Status is written to Postgres (`jobs`, `tracks`) and mirrored into the Redis hash `job:{id}`
(docs/data-model.md §4; empty string = null, the API reads it that way). Stages are the openapi
`Job.stage` enum. Heavy imports (numpy, librosa, torch, psycopg, botocore) happen inside functions
so the tests run with only pytest + jsonschema.
"""
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

WORKER_DIR = Path(__file__).resolve().parent.parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from services import fetch  # noqa: E402

# Repo checkout: <repo>/contracts. Docker image: /contracts (see Dockerfile).
CONTRACTS_DIR = Path(os.environ.get("CONTRACTS_DIR", WORKER_DIR.parent.parent / "contracts"))

QUEUE = "jobs:analysis"
JOB_TTL = 86400
SAMPLE_RATE = 22050
STREAMING = ("youtube", "soundcloud")

# openapi Job.stage -> progress reported when the stage starts.
PROGRESS = {
    "fetching_media": 5,
    "normalizing_audio": 15,
    "detecting_beats": 25,
    "detecting_chords": 40,
    "detecting_key": 80,
    "rendering_spectrogram": 88,
    "saving_results": 95,
}
# ChordBeatAnalyzer stage name -> openapi stage ("load" just reads the normalized WAV).
ANALYZER_STAGES = {
    "beats": "detecting_beats",
    "chords": "detecting_chords",
    "key": "detecting_key",
    "spectrogram": "rendering_spectrogram",
}

log = logging.getLogger("consumer")


class JobError(Exception):
    """A failure whose message is safe and readable for clients."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------- adapters (real implementations; tests pass fakes with the same methods) ----------

class PgJobs:
    """jobs/tracks writes over psycopg 3. One short connection per call keeps a long-idle worker
    immune to dropped connections; a job makes fewer than 15 writes."""

    def __init__(self, url: str):
        self.url = url.replace("postgresql+psycopg://", "postgresql://", 1)  # SQLAlchemy-style URL, as the API takes

    def _exec(self, sql: str, params: tuple) -> None:
        import psycopg

        with psycopg.connect(self.url, autocommit=True, connect_timeout=10) as conn:
            conn.execute(sql, params)

    def start(self, job_id: str) -> None:
        self._exec("UPDATE jobs SET status='processing', stage=NULL, progress=0, error=NULL, started_at=%s "
                   "WHERE id=%s", (_now(), job_id))

    def stage(self, job_id: str, stage: str, progress: int) -> None:
        self._exec("UPDATE jobs SET stage=%s, progress=%s WHERE id=%s", (stage, progress, job_id))

    def save_source(self, track_id: str, title: Optional[str], duration_sec: Optional[float],
                    thumbnail_url: Optional[str]) -> None:
        self._exec("UPDATE tracks SET title=COALESCE(%s, title), duration_sec=COALESCE(%s, duration_sec), "
                   "thumbnail_url=COALESCE(%s, thumbnail_url) WHERE id=%s",
                   (title, duration_sec, thumbnail_url, track_id))

    def save_audio_key(self, track_id: str, audio_key: str) -> None:
        self._exec("UPDATE tracks SET audio_key=%s WHERE id=%s", (audio_key, track_id))

    def save_track(self, track_id: str, analysis_key: str, analyzer_version: str, duration_sec: float) -> None:
        self._exec("UPDATE tracks SET analysis_key=%s, analyzer_version=%s, duration_sec=%s WHERE id=%s",
                   (analysis_key, analyzer_version, duration_sec, track_id))

    def complete(self, job_id: str) -> None:
        self._exec("UPDATE jobs SET status='complete', stage=NULL, progress=100, error=NULL, completed_at=%s "
                   "WHERE id=%s", (_now(), job_id))

    def fail(self, job_id: str, error: str) -> None:
        self._exec("UPDATE jobs SET status='failed', error=%s, completed_at=%s WHERE id=%s",
                   (error[:1000], _now(), job_id))


def s3_client():
    import botocore.session
    from botocore.config import Config

    return botocore.session.get_session().create_client(
        "s3", region_name=os.environ.get("S3_REGION", "us-east-1"),
        endpoint_url=os.environ.get("S3_ENDPOINT") or None,
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY") or None,
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY") or None,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))


def load_validator() -> Callable[[dict], List[str]]:
    """contracts/validate.py check(doc) -> list of errors."""
    spec = importlib.util.spec_from_file_location("contracts_validate", CONTRACTS_DIR / "validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check


def normalize_audio(src: Path, dst: Path) -> str:
    """Any input -> mono 22.05 kHz 16-bit WAV. ffmpeg when on PATH, else librosa. Returns the method used."""
    if shutil.which("ffmpeg"):
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
             "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", str(dst)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
        if proc.returncode != 0:
            log.warning("ffmpeg failed: %s", proc.stderr.decode(errors="replace")[-500:])
            raise JobError("Could not decode the audio file")
        return "ffmpeg"
    import librosa
    import soundfile

    try:
        y, _ = librosa.load(str(src), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        log.warning("librosa could not decode %s: %s", src, e)
        raise JobError("Could not decode the audio file") from e
    soundfile.write(str(dst), y, SAMPLE_RATE, subtype="PCM_16")
    return "librosa"


# ---------- the job ----------

class Consumer:
    def __init__(self, *, analyzer, redis, s3, bucket: str, jobs,
                 validate: Callable[[dict], List[str]],
                 normalize: Callable[[Path, Path], str] = normalize_audio,
                 fetch_stream: Callable[[str, Path], Any] = fetch.fetch_stream,
                 to_m4a: Callable[[Path, Path], None] = fetch.to_m4a,
                 store_stream_audio: bool = False):
        self.analyzer = analyzer
        self.redis = redis
        self.s3 = s3
        self.bucket = bucket
        self.jobs = jobs
        self.validate = validate
        self.normalize = normalize
        self.fetch_stream = fetch_stream
        self.to_m4a = to_m4a
        self.store_stream_audio = store_stream_audio  # OFFLINE_STREAM_AUDIO (PLAN.md R-1)

    def _live(self, job_id: str, **fields: Any) -> None:
        key = f"job:{job_id}"
        self.redis.hset(key, mapping={k: "" if v is None else v for k, v in fields.items()})
        self.redis.expire(key, JOB_TTL)

    def _stage(self, job_id: str, stage: str) -> None:
        progress = PROGRESS[stage]
        self.jobs.stage(job_id, stage, progress)
        self._live(job_id, stage=stage, progress=progress)

    def handle(self, raw: str) -> None:
        """Process one queue message. Never raises: failures are written to the job."""
        try:
            msg = json.loads(raw)
            job_id, track_id, source = msg["jobId"], msg["trackId"], msg["source"]
        except (ValueError, KeyError, TypeError):
            log.error("dropping malformed queue message: %.200r", raw)
            return

        started = time.perf_counter()
        timings: Dict[str, float] = {}
        stage: Optional[str] = None
        log.info("job %s: start (track %s, source %s)", job_id, track_id, source)
        try:
            self.jobs.start(job_id)
            self._live(job_id, status="processing", stage=None, progress=0, error=None)
            if source in STREAMING:
                url, source_ref = fetch.source_url(msg)
            elif source in ("file", "recording") and msg.get("uploadKey"):
                url, source_ref = None, None
            else:
                raise JobError("Unsupported job source")

            with tempfile.TemporaryDirectory(prefix="md-job-") as tmp:
                tmp_dir = Path(tmp)
                src, wav = tmp_dir / "input", tmp_dir / "audio.wav"

                stage = "fetching_media"
                self._stage(job_id, stage)
                t = time.perf_counter()
                if url:
                    src, info = self.fetch_stream(url, tmp_dir)
                    self.jobs.save_source(track_id, info.get("title"), info.get("duration"), info.get("thumbnail"))
                else:
                    self._download(msg["uploadKey"], src)
                timings["fetch"] = time.perf_counter() - t

                stage = "normalizing_audio"
                self._stage(job_id, stage)
                t = time.perf_counter()
                method = self.normalize(src, wav)
                timings["normalize"] = time.perf_counter() - t
                log.info("job %s: normalized with %s", job_id, method)

                def on_stage(name: str) -> None:
                    nonlocal stage
                    if name in ANALYZER_STAGES:
                        stage = ANALYZER_STAGES[name]
                        self._stage(job_id, stage)

                doc = self.analyzer.analyze(str(wav), track_id=track_id, source=source,
                                            source_ref=source_ref, on_stage=on_stage)
                timings.update(getattr(self.analyzer, "timings", {}) or {})

                stage = "saving_results"
                self._stage(job_id, stage)
                t = time.perf_counter()
                errors = self.validate(doc)
                if errors:
                    log.error("job %s: analysis failed validation: %s", job_id, errors[:5])
                    raise JobError("The analysis did not pass validation")
                key = f"analysis/{track_id}.json"
                self.s3.put_object(Bucket=self.bucket, Key=key, ContentType="application/json",
                                   Body=json.dumps(doc, separators=(",", ":")).encode())
                self.jobs.save_track(track_id, key, doc["meta"]["analyzerVersion"], doc["meta"]["durationSec"])
                if url and self.store_stream_audio:
                    self._store_audio(track_id, src, tmp_dir / "extracted.m4a")
            # Leaving the with-block deletes the fetched audio unless it was stored above.
            self.jobs.complete(job_id)
            self._live(job_id, status="complete", stage=None, progress=100, error=None)
            timings["save"] = time.perf_counter() - t
        except Exception as e:
            if isinstance(e, (JobError, fetch.FetchError)):
                error = str(e)
                log.warning("job %s: failed at %s: %s", job_id, stage, error)
            else:
                error = f"Analysis failed while {stage.replace('_', ' ')}" if stage else "Analysis failed"
                log.exception("job %s: unexpected error at %s", job_id, stage)
            self._fail(job_id, error)
        finally:
            timings["total"] = time.perf_counter() - started
            log.info("job %s: timings %s", job_id, " ".join(f"{k}={v:.2f}s" for k, v in timings.items()))

    def _download(self, key: str, dst: Path) -> None:
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"]
            with open(dst, "wb") as f:
                shutil.copyfileobj(body, f, 1 << 20)
        except Exception as e:
            log.warning("download of %s failed: %s", key, e)
            raise JobError("Could not download the uploaded audio") from e

    def _store_audio(self, track_id: str, src: Path, m4a: Path) -> None:
        """extracted-audio/{trackId}.m4a, only when OFFLINE_STREAM_AUDIO=true (docs/data-model.md §3)."""
        self.to_m4a(src, m4a)
        key = f"extracted-audio/{track_id}.m4a"
        self.s3.put_object(Bucket=self.bucket, Key=key, ContentType="audio/mp4", Body=m4a.read_bytes())
        self.jobs.save_audio_key(track_id, key)

    def _fail(self, job_id: str, error: str) -> None:
        # Each write on its own: a dead Postgres must not stop the Redis mirror, and vice versa.
        for write in (lambda: self.jobs.fail(job_id, error),
                      lambda: self._live(job_id, status="failed", error=error)):
            try:
                write()
            except Exception:
                log.exception("job %s: could not record the failure", job_id)

    def run_forever(self, timeout: int = 5) -> None:
        log.info("waiting for jobs on %s", QUEUE)
        while True:
            item = self.redis.blpop([QUEUE], timeout=timeout)
            if item:
                self.handle(item[1])


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import redis

    from services.chord_beat_analysis import DEFAULT_CHORD_MODEL_DIR, ChordBeatAnalyzer

    for var in ("REDIS_URL", "DATABASE_URL"):
        if not os.environ.get(var):
            sys.exit(f"{var} is not set")
    t = time.perf_counter()
    analyzer = ChordBeatAnalyzer(os.environ.get("CHORD_MODEL_DIR", DEFAULT_CHORD_MODEL_DIR))
    log.info("analyzer ready in %.1fs (ffmpeg %s)", time.perf_counter() - t,
             "found" if shutil.which("ffmpeg") else "not found: normalizing with librosa")
    Consumer(
        analyzer=analyzer,
        redis=redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True),
        s3=s3_client(),
        bucket=os.environ.get("S3_BUCKET", "musicdecoded"),
        jobs=PgJobs(os.environ["DATABASE_URL"]),
        validate=load_validator(),
        store_stream_audio=os.environ.get("OFFLINE_STREAM_AUDIO", "").lower() == "true",
    ).run_forever()


if __name__ == "__main__":
    main()
