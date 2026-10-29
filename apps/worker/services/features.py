"""Pure feature functions for analysis v1 (PLAN.md §5.3).

Everything here takes NumPy arrays / lists and returns plain Python values, so it
can be unit-tested on synthetic inputs without librosa, torch or audio files.
"""
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles, index 0 = tonic.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Same pattern as contracts/analysis.schema.json chords[].label
CHORD_LABEL = re.compile(r"^(N|X|[A-G](#|b)?:[A-Za-z0-9()#b,/]+)$")


def estimate_key(mean_chroma: Sequence[float]) -> Optional[Dict[str, Any]]:
    """Krumhansl-Schmuckler: correlate the mean chroma with all 24 rotated profiles.

    Returns {"tonic", "mode", "confidence"} or None when the chroma has no variance
    (silence / noise). Confidence is the winning Pearson correlation clipped to [0, 1].
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
    return {"tonic": PITCH_CLASSES[tonic], "mode": mode, "confidence": round(min(max(r, 0.0), 1.0), 3)}


def estimate_bpm(beats: Sequence[float]) -> Optional[float]:
    """60 / median inter-beat interval, or None with fewer than 2 beats."""
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


def mean_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
