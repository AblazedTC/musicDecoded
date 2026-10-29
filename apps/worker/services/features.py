"""Pure feature functions for analysis v1 (PLAN.md §5.3).

Everything here takes NumPy arrays / lists and returns plain Python values, so it
can be unit-tested on synthetic inputs without librosa, torch or audio files.
"""
import base64
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles, index 0 = tonic.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Same pattern as contracts/analysis.schema.json chords[].label
CHORD_LABEL = re.compile(r"^(N|X|[A-G](#|b)?:[A-Za-z0-9()#b,/]+)$")

SPEC_FPS = 20
SPEC_BINS = 64
PEAKS_FPS = 20
DB_FLOOR = -80.0

# T-119 gates. Measured: real songs correlate 0.70-0.78 with their key profile; white/pink noise and hum
# 0.35-0.42 (and still "win" some key), so below KEY_MIN_CONFIDENCE the key is noise.
KEY_MIN_CONFIDENCE = 0.5
SILENCE_DBFS = -50.0  # whole-track RMS below this = near silence: no key, no beats
MIN_BEATS = 8  # fewer beats than ~2 bars can't support a tempo


def rms_dbfs(y: Sequence[float]) -> float:
    """Whole-signal RMS in dBFS (full scale = 1.0); -inf-safe (floor -200)."""
    y = np.asarray(y, dtype=float)
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    return 20.0 * float(np.log10(max(rms, 1e-10)))


def is_near_silent(y: Sequence[float]) -> bool:
    return bool(rms_dbfs(y) < SILENCE_DBFS)


def estimate_key(mean_chroma: Sequence[float]) -> Optional[Dict[str, Any]]:
    """Krumhansl-Schmuckler: correlate the mean chroma with all 24 rotated profiles.

    Returns {"tonic", "mode", "confidence"}, or None when the chroma has no variance or the
    best correlation is below KEY_MIN_CONFIDENCE (noise, keyless audio). Confidence is the
    winning Pearson correlation clipped to [0, 1].
    """
    chroma = np.asarray(mean_chroma, dtype=float)
    if chroma.shape != (12,) or not np.all(np.isfinite(chroma)) or np.std(chroma) < 1e-9:
        return None
    best = None
    for mode, profile in (("major", KS_MAJOR), ("minor", KS_MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(chroma, np.roll(profile, tonic))[0, 1])
            if best is None or r > best[0]:
                best = (r, tonic, mode)
    r, tonic, mode = best
    if r < KEY_MIN_CONFIDENCE:
        return None
    return {"tonic": PITCH_CLASSES[tonic], "mode": mode, "confidence": round(min(max(r, 0.0), 1.0), 3)}


def estimate_bpm(beats: Sequence[float]) -> Optional[float]:
    """60 / median inter-beat interval, or None with fewer than MIN_BEATS beats."""
    if len(beats) < MIN_BEATS:
        return None
    ibi = np.diff(np.sort(np.asarray(beats, dtype=float)))
    ibi = ibi[ibi > 0]
    if ibi.size == 0:
        return None
    return round(60.0 / float(np.median(ibi)), 2)


def estimate_time_signature(beats: Sequence[float], downbeats: Sequence[float]) -> Optional[Dict[str, int]]:
    """Median number of beats per bar (downbeat to next downbeat), over 4."""
    b = np.sort(np.asarray(beats, dtype=float))
    d = np.sort(np.asarray(downbeats, dtype=float))
    if d.size < 2 or b.size == 0:
        return None
    # Half a beat of tolerance so a beat landing exactly on a downbeat counts in the bar it starts.
    tol = 0.5 * float(np.median(np.diff(b))) if b.size > 1 else 0.0
    counts = [int(np.sum((b >= s - tol) & (b < e - tol))) for s, e in zip(d[:-1], d[1:])]
    counts = [c for c in counts if c > 0]
    if not counts:
        return None
    return {"numerator": max(1, int(round(float(np.median(counts))))), "denominator": 4}


def build_beats(beats: Sequence[float], downbeats: Sequence[float], atol: float = 0.05) -> List[Dict[str, Any]]:
    """Sorted [{t, downbeat}] with a beat marked as a downbeat when within atol of one."""
    d = np.sort(np.asarray(downbeats, dtype=float))
    out = []
    for t in sorted(float(x) for x in beats if x >= 0):
        is_down = bool(d.size and np.min(np.abs(d - t)) <= atol)
        out.append({"t": round(t, 3), "downbeat": is_down})
    return out


def normalize_label(label: str) -> str:
    """Map a chord-model label onto the schema pattern ('C' -> 'C:maj', unknown -> 'X')."""
    label = label.strip()
    if re.fullmatch(r"[A-G][#b]?", label):
        label += ":maj"
    return label if CHORD_LABEL.match(label) else "X"


def contiguous_chords(chords: Sequence[Dict[str, Any]], duration: float) -> List[Dict[str, Any]]:
    """Sorted, contiguous, non-overlapping segments covering [0, duration].

    Input items have start/end and 'label' (or 'chord'). Gaps become 'N', overlaps are
    trimmed, adjacent equal labels are merged.
    """
    segs = sorted(
        ((float(c["start"]), float(c["end"]), normalize_label(c.get("label", c.get("chord", "N")))) for c in chords),
        key=lambda s: s[0],
    )
    out: List[Dict[str, Any]] = []
    cursor = 0.0

    def push(start, end, label):
        start, end = round(start, 3), round(end, 3)
        if end <= start:
            return
        if out and out[-1]["label"] == label:
            out[-1]["end"] = end
        else:
            out.append({"start": start, "end": end, "label": label})

    for start, end, label in segs:
        start, end = max(start, cursor), min(end, duration)
        if end <= start:
            continue
        if start > cursor:
            push(cursor, start, "N")
        push(start, end, label)
        cursor = end
    if duration > cursor:
        push(cursor, duration, "N")
    # Rounding can leave a 1 ms seam; snap each start to the previous end.
    for a, b in zip(out, out[1:]):
        b["start"] = a["end"]
    return out


def frame_index(n_items: int, items_per_sec: float, fps: float) -> np.ndarray:
    """Output-frame index (at fps) of each input item sampled at items_per_sec."""
    return np.floor(np.arange(n_items) * (fps / items_per_sec)).astype(int)


def pool(values: np.ndarray, idx: np.ndarray, n_out: int, how: str) -> np.ndarray:
    """Reduce rows of values into n_out frames by idx (mean or max). Empty frames are 0."""
    values = np.asarray(values, dtype=float)
    out = np.zeros((n_out,) + values.shape[1:])
    keep = idx < n_out
    values, idx = values[keep], idx[keep]
    if how == "max":
        np.maximum.at(out, idx, values)
    else:
        counts = np.bincount(idx, minlength=n_out).astype(float)
        np.add.at(out, idx, values)
        out /= np.maximum(counts, 1).reshape((-1,) + (1,) * (values.ndim - 1))
    return out


def encode_spectrogram(db: np.ndarray, fps: float = SPEC_FPS) -> Dict[str, Any]:
    """(frames, bins) dB matrix -> schema spectrogram object.

    dB is clipped to [-80, 0] and scaled to 0..255, row-major frames x bins.
    """
    db = np.clip(np.asarray(db, dtype=float), DB_FLOOR, 0.0)
    frames, bins = db.shape
    u8 = np.round((db - DB_FLOOR) / -DB_FLOOR * 255).astype(np.uint8)
    return {
        "fps": fps,
        "bins": int(bins),
        "frames": int(frames),
        "encoding": "uint8-base64",
        "data": base64.b64encode(np.ascontiguousarray(u8).tobytes()).decode("ascii"),
    }


def mel_power_to_db(mel_power: np.ndarray) -> np.ndarray:
    """Power -> dB relative to the track's max (so the loudest bin is 0 dB)."""
    mel_power = np.maximum(np.asarray(mel_power, dtype=float), 1e-10)
    return 10.0 * np.log10(mel_power / mel_power.max())


def waveform_peaks(y: np.ndarray, sr: int, duration: float, fps: float = PEAKS_FPS) -> Dict[str, Any]:
    """Max |sample| per 1/fps window, normalized so the loudest window is 1."""
    n_out = max(1, int(np.ceil(duration * fps)))
    peaks = pool(np.abs(np.asarray(y, dtype=float)), frame_index(len(y), sr, fps), n_out, "max")
    top = peaks.max()
    if top > 0:
        peaks = peaks / top
    return {"fps": fps, "peaks": [round(float(p), 3) for p in peaks]}


def spectrogram(y: np.ndarray, sr: int, duration: float, fps: float = SPEC_FPS, bins: int = SPEC_BINS,
                hop_length: int = 512) -> Dict[str, Any]:
    """64-bin mel at 20 fps. Computes mel at a fine hop, then averages into 1/fps frames
    (22.05 kHz / 20 fps is not an integer hop)."""
    import librosa

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=bins, hop_length=hop_length)  # (bins, frames)
    n_out = max(1, int(np.ceil(duration * fps)))
    pooled = pool(mel.T, frame_index(mel.shape[1], sr / hop_length, fps), n_out, "mean")
    return encode_spectrogram(mel_power_to_db(pooled), fps)


def mean_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
