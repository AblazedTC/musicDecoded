# Analysis cost model

Measured numbers for PLAN.md §9's cost model (T-115). Prices were checked on **2026-10-30**; re-check them
before making a pricing decision.

## How it was measured

`apps/worker/scripts/benchmark.py` runs the same steps as a queue job (fetch → normalize → analyze →
validate) on each input and prints per-stage median/p90, CPU time and RSS. It takes local files or YouTube
ids, and deletes the fetched audio afterwards.

```bash
# from apps/worker, against the built worker image
docker run --rm --cpus 2 --cpuset-cpus 0-1 --memory 4g \
  -v "$PWD/scripts:/app/scripts:ro" -v "$PWD/data:/app/data:ro" <worker-image> \
  python scripts/benchmark.py data/sunnydays.mp3 2eZVbrO6Z1M MIb5eknCKSA N0KuBFK9r24 fOgsFH0s5uU \
    yEWFxNNjalI YlTQSg4so8k 2pXa5jzaOBg yJEPTk7FLmk Pk9vPMurHVo
```

- **Machine:** worker Docker image (Python 3.14.10, torch 2.15.0+cpu, librosa 1.0.0), capped to 2 CPUs
  (pinned, so torch uses 2 threads) and 4 GB, on an Intel i7-9700K (Docker Desktop, Windows). It stands in
  for a 2-vCPU cloud instance; cloud vCPUs are usually hyperthreads and somewhat slower, so treat the
  numbers as a lower bound.
- **Songs:** 10 tracks, 30 min of audio, 127–249 s each: `data/sunnydays.mp3` plus 9 Creative Commons
  (CC BY) Kevin MacLeod tracks fetched from YouTube: Monkeys Spinning Monkeys (ukulele), Local Forecast
  (lounge jazz), Cipher and Wallpaper (electronic), Hall of the Mountain King (orchestral), Gymnopedie No 1
  (solo piano), Volatile Reaction (rock), Pamgaea (world), Sneaky Snitch (comedic).
- **Path:** the standard (free) pipeline: librosa beats + CNN-LSTM chords, with the T-118 weight cache.
- The worker already logs per-stage wall times for every job (`job …: timings fetch=… normalize=… …`).

## Results

| Stage | Median s | p90 s | Share of time |
|---|---:|---:|---:|
| fetch (YouTube, yt-dlp) | 7.57 | 8.09 | 32% |
| normalize (FFmpeg) | 0.51 | 0.76 | 3% |
| load | 0.02 | 0.09 | 0% |
| beats (librosa) | 0.27 | 1.17 | 6% |
| chords (CNN-LSTM, 5 nets) | 9.05 | 13.81 | 53% |
| key | 0.89 | 1.29 | 5% |
| spectrogram + peaks | 0.18 | 0.28 | 1% |
| validate + serialize | 0.03 | 0.05 | 0% |
| **total per song** | **18.0** | **24.3** | |

| Metric | Value |
|---|---|
| Wall time without fetch (file uploads) | median 11.1 s, p90 17.4 s |
| CPU time per song (both cores) | median 20.4 s, p90 30.3 s |
| Real-time factor (total ÷ song length) | median 0.108, p90 0.120 (≈ 6.5 s per minute of audio) |
| Worker startup (load 5 chord nets) | 2.4 s, 369 MB RSS |
| Peak RSS | 1,347 MB |
| RSS while idle after jobs | ~860 MB |

Notes:
- **Chords are the cost.** Over half the time, and it scales with song length; beats/key/spectrogram are
  negligible. The p90 beats figure is the first job only (librosa/numba JIT warm-up, ~8 s once per process).
- **Fetch is network time**, not CPU (7.6 s on a home connection). Overlapping the next job's fetch with
  the current analysis would cut per-song host time to ~11 s; uploads already skip it.
- **Memory**: 2 GB is the minimum instance size (peak 1.35 GB). Songs up to the 10-minute cap
  (`MAX_DURATION_SEC=600`) will push the peak higher; 4 GB is the safe size.
- **Not measured:** songs over 4 min (assume linear: ~0.11 × length, so a 10-minute song ≈ 65 s), and the
  20-song / 2–8 min set T-115 asks for. Re-run with longer tracks before launch.

## Hosts and $ per 1,000 uncached songs

Formula (PLAN.md §9): `monthly compute ≈ (uncached songs/day × seconds/song ÷ 3600) × $/worker-hour × 30`.

| Host | Price (published) | $/hour | Seconds billed per song | **$ / 1,000 uncached songs** | One worker's capacity |
|---|---|---:|---|---:|---|
| Fly.io `performance-2x`, 4 GB | $66.00/month (720 h, `iad`/`ewr`) | $0.0917 | 18.0 wall (p90 24.3) | **$0.46** (p90 $0.62; $0.28 if fetch overlaps) | ~4,800 songs/day |
| Fly.io `shared-cpu-2x`, 2 GB, throttled | $13.39/month | $0.0186 | ~163 wall (20.4 CPU-s at the 2 × 6.25% baseline) | **$0.84** | ~530 songs/day sustained |
| Fly.io `shared-cpu-2x`, 2 GB, bursting | same | $0.0186 | 18.0 wall | $0.09 | only until the burst balance runs out |
| Railway (usage-based) | vCPU $0.00000772/s, RAM $0.00000386/GB-s | — | 20.4 vCPU-s + 18 s × 1.35 GB | **$0.25** (p90 $0.36) ~4,800 songs/day per 2-vCPU replica |

How the rows are computed:
- Fly bills a running Machine per second regardless of load, so cost = wall seconds × hourly price.
  `shared` vCPUs get a **6.25% baseline** with a burst balance (5 s initial, 500 s max); a sustained queue
  drains it, after which a 20 s CPU job takes ~160 s. It is cheap per hour but slower per song than
  `performance-2x` once busy.
- Railway bills **actual usage**: CPU-seconds used and RAM held. Per song: 20.4 × $0.00000772 = $0.00016
  CPU + 18 × 1.35 × $0.00000386 = $0.00009 RAM.

### Fixed cost (the worker idles most of the day at low volume)

| Host | Idle worker / month |
|---|---|
| Fly `performance-2x` always on | $66.00 |
| Fly `shared-cpu-2x` always on | $13.39 |
| Railway, ~0.86 GB idle RSS, ~0 CPU | ~$8.60 RAM (Hobby plan $5/month includes $5 of usage) |

Fly's autostop wakes Machines on HTTP traffic, and the worker pulls from Redis instead, so scale-to-zero on
Fly would need the API to start the Machine through the Machines API. Not worth it before real traffic.

### Example: 1,000 uncached songs/day

- Fly `performance-2x`: 1,000 × 18 s ÷ 3600 = 5 h/day × $0.0917 × 30 = **$13.75/month** of busy time, but the
  machine is billed all month: **$66**.
- Fly `shared-cpu-2x`: above its ~530/day sustained capacity; needs 2 machines (**$26.78**) and jobs wait
  minutes.
- Railway: 30,000 songs × $0.00025 + ~$8.60 idle ≈ **$16/month**.

Storage on R2: ≤1.5 MB per analysis at $0.015/GB-month, so 1,000 songs add ~$0.02/month, cumulative; R2 has
no egress fees. Streaming fetches are inbound traffic, which neither host bills.

## Cache hits

Analysis is global and deduped by content hash / videoId / URL (PLAN.md §5 rule 3), and a cache hit costs
only an API request and an R2 read (~$0). So:

`cost per 1,000 transcription requests = (1 − cache hit rate) × $ per 1,000 uncached songs`

At a 50% hit rate, Railway is ~$0.13 and Fly `performance-2x` ~$0.23 per 1,000 requests. Popular YouTube
songs should hit far more often than personal uploads and recordings, which almost never hit.

## Recommendation

1. **Host: run the worker on Railway** (usage-based, ~$9/month idle, ~$0.25 per 1,000 uncached songs). It
   is cheaper than Fly `performance-2x` even when that machine is fully busy ($0.46 per 1,000), because
   Railway bills only the CPU-seconds used (~$20 per vCPU-month vs ~$33 on Fly). Revisit only for reserved
   or GPU pricing. Avoid Fly `shared-cpu` for the worker: the CPU baseline makes it slow under any
   sustained queue.
2. **Free limits:** compute is cheap per song (~$0.0005 at worst), so limits are about abuse and queue
   latency, not margin. A maxed-out free account at 50/day all uncached costs ~$0.75/month on Fly
   `performance-2x` (~$0.38 on Railway). Keep **guest 10/day**, lower **account to 30/day** for launch
   (one 2-vCPU worker, ~4,800 songs/day, then serves ~160 maxed-out accounts), and raise it once the cache hit rate is known.
3. **Pro:** "unlimited" with a fair-use cap of **300 uncached songs/day** per user. Pro's GPU models
   (below) will set the real Pro price; the standard pipeline is not a meaningful cost.
4. Add a **global daily uncached-job budget** in the API (e.g. 5× one worker's capacity) as a circuit
   breaker, and alert on queue depth.

## GPU / Pro path: not measured

T-115's GPU criterion is **not done**: no GPU host was available, and Beat Transformer is not wired into
the job path yet (the image ships CPU-only torch). Re-run `scripts/benchmark.py` on a GPU host with a CUDA
torch build once the Pro path exists.

## Sources

- Fly.io pricing, <https://docs.fly.io/about/pricing> (checked 2026-10-30): `shared-cpu-2x` 2 GB
  $13.39/month, `performance-2x` 4 GB $66.00/month, cheapest regions `iad`/`ewr`, 720 h months, billed per
  second.
- Fly.io CPU performance, <https://fly.io/docs/machines/cpu-performance/> (checked 2026-10-30): shared vCPU
  6.25% baseline, 5 s initial / 500 s max burst balance.
- Railway pricing, <https://railway.com/pricing> (checked 2026-10-30): $0.00000772 per vCPU-second,
  $0.00000386 per GB-second of RAM, Hobby $5/month with $5 usage included, Pro $20/month with $20 included.
