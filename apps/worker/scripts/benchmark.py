"""Analysis cost benchmark (T-115).

Runs the same steps as a queue job (fetch -> normalize -> analyze -> validate) over N inputs and reports
median/p90 seconds per song and per stage, the real-time factor, and peak RSS. No Redis/Postgres/S3.

Inputs are local audio paths or 11-char YouTube video ids (fetched with services/fetch.py, deleted after).

  python scripts/benchmark.py data/sunnydays.mp3 dQw4w9WgXcQ ... [--json out.json]

To approximate a small cloud instance, run it in the worker image with a CPU/RAM cap:

  docker run --rm --cpus 2 --memory 4g -v "$PWD/scripts:/app/scripts" -v "$PWD/data:/app/data" \
      <worker-image> python scripts/benchmark.py data/sunnydays.mp3 <videoId> ...
"""
import argparse
import json
import logging
import os
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKER_DIR))

from services import consumer, fetch  # noqa: E402

STAGES = ["fetch", "normalize", "load", "beats", "chords", "key", "spectrogram", "save", "total"]


def pct(values, q):
    """Linear-interpolated percentile (q in 0..100)."""
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    k = (len(v) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def peak_rss_mb():
    """Peak RSS of this process in MB (Linux reports KB). ffmpeg children are small and not counted
    (RUSAGE_CHILDREN would report the forked parent's RSS, not ffmpeg's)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def rss_mb():
    """Current RSS in MB (what a usage-billed host charges for while idle)."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def cpu_seconds():
    """CPU time used so far by this process and its finished children (ffmpeg)."""
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def run_one(analyzer, validate, item):
    timings = {}
    started = time.perf_counter()
    cpu_start = cpu_seconds()
    with tempfile.TemporaryDirectory(prefix="md-bench-") as tmp:
        tmp_dir = Path(tmp)
        t = time.perf_counter()
        if Path(item).is_file():
            src, name = Path(item), Path(item).name
        elif fetch.YOUTUBE_ID.match(item):
            src, info = fetch.fetch_stream(f"https://www.youtube.com/watch?v={item}", tmp_dir)
            name = f"{item} {info.get('title', '')}"[:60]
        else:
            raise SystemExit(f"not a file or YouTube id: {item}")
        timings["fetch"] = time.perf_counter() - t

        wav = tmp_dir / "audio.wav"
        t = time.perf_counter()
        consumer.normalize_audio(src, wav)
        timings["normalize"] = time.perf_counter() - t

        doc = analyzer.analyze(str(wav), track_id="bench", source="file")
        timings.update(analyzer.timings)

        t = time.perf_counter()
        errors = validate(doc)
        if errors:
            raise SystemExit(f"{item}: analysis failed validation: {errors[:3]}")
        json.dumps(doc, separators=(",", ":"))
        timings["save"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - started
    return {"input": item, "name": name, "durationSec": doc["meta"]["durationSec"], "timings": timings,
            "cpuSec": cpu_seconds() - cpu_start}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", help="audio file paths or YouTube video ids")
    parser.add_argument("--json", help="also write the raw results here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    from services.chord_beat_analysis import DEFAULT_CHORD_MODEL_DIR, ChordBeatAnalyzer

    t = time.perf_counter()
    analyzer = ChordBeatAnalyzer(os.environ.get("CHORD_MODEL_DIR", DEFAULT_CHORD_MODEL_DIR))
    startup = time.perf_counter() - t
    validate = consumer.load_validator()
    import torch

    print(f"cpus visible: {os.cpu_count()}  torch threads: {torch.get_num_threads()}  analyzer startup: "
          f"{startup:.1f}s  RSS after startup: {rss_mb():.0f} MB", flush=True)

    results = []
    for item in args.inputs:
        r = run_one(analyzer, validate, item)
        r["peakRssMb"] = round(peak_rss_mb())
        r["rssAfterMb"] = round(rss_mb())
        results.append(r)
        tm = r["timings"]
        print(f"{r['name']:<60} {r['durationSec']:6.0f}s audio  {tm['total']:6.1f}s total  "
              f"(fetch {tm['fetch']:.1f}, beats {tm['beats']:.1f}, chords {tm['chords']:.1f})  "
              f"cpu {r['cpuSec']:.1f}s  peak RSS {r['peakRssMb']} MB, after {r['rssAfterMb']} MB", flush=True)

    print(f"\n{len(results)} songs, {sum(r['durationSec'] for r in results) / 60:.1f} min of audio")
    print(f"{'stage':<12}{'median s':>10}{'p90 s':>10}{'share':>8}")
    total_sum = sum(r["timings"]["total"] for r in results)
    for stage in STAGES:
        vals = [r["timings"].get(stage, 0.0) for r in results]
        share = sum(vals) / total_sum * 100
        print(f"{stage:<12}{statistics.median(vals):>10.2f}{pct(vals, 90):>10.2f}{share:>7.0f}%")
    rtf = [r["timings"]["total"] / r["durationSec"] for r in results]
    compute = [r["timings"]["total"] - r["timings"]["fetch"] for r in results]
    print(f"real-time factor (total / audio length): median {statistics.median(rtf):.3f}, p90 {pct(rtf, 90):.3f}")
    print(f"compute without fetch: median {statistics.median(compute):.1f}s, p90 {pct(compute, 90):.1f}s")
    cpu = [r["cpuSec"] for r in results]
    print(f"CPU time per song (all cores): median {statistics.median(cpu):.1f}s, p90 {pct(cpu, 90):.1f}s")
    peak = peak_rss_mb()
    print(f"peak RSS: {peak:.0f} MB, RSS when idle after the last job: {rss_mb():.0f} MB")

    if args.json:
        Path(args.json).write_text(json.dumps({"startupSec": startup, "peakRssMb": peak, "results": results}, indent=2))


if __name__ == "__main__":
    main()
