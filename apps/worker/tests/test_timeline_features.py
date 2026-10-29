"""Spectrogram + waveform peaks (T-111): sizes and encoding on synthetic signals."""
import base64
import json

import pytest

np = pytest.importorskip("numpy")

from services import features  # noqa: E402
from services.chord_beat_analysis import build_analysis  # noqa: E402

SR = 22050


def test_encode_spectrogram_scaling_and_layout():
    db = np.array([[0.0, -80.0, -40.0], [-100.0, 5.0, -20.0]])  # 2 frames x 3 bins, out-of-range clipped
    spec = features.encode_spectrogram(db)
    assert spec["fps"] == 20 and spec["frames"] == 2 and spec["bins"] == 3
    assert spec["encoding"] == "uint8-base64"
    raw = list(base64.b64decode(spec["data"]))
    assert raw == [255, 0, 128, 0, 255, 191]  # row-major: data[frame * bins + bin]


def test_waveform_peaks_20fps_normalized():
    duration = 3.0
    t = np.arange(int(SR * duration)) / SR
    y = np.sin(2 * np.pi * 440 * t) * np.where(t < 1.0, 0.25, 0.5)
    out = features.waveform_peaks(y, SR, duration)
    peaks = np.array(out["peaks"])
    assert out["fps"] == 20 and len(peaks) == 60
    assert peaks.max() == pytest.approx(1.0) and peaks.min() >= 0
    assert peaks[:20] == pytest.approx(0.5, abs=0.01)  # first second is half as loud


def test_waveform_peaks_silence():
    out = features.waveform_peaks(np.zeros(SR), SR, 1.0)
    assert out["peaks"] == [0.0] * 20


def test_pool_mean_and_max():
    idx = np.array([0, 0, 1, 1, 1])
    vals = np.array([1.0, 3.0, 2.0, 4.0, 6.0])
    assert features.pool(vals, idx, 3, "mean").tolist() == [2.0, 4.0, 0.0]
    assert features.pool(vals, idx, 3, "max").tolist() == [3.0, 6.0, 0.0]


def test_spectrogram_from_audio():
    pytest.importorskip("librosa")
    duration = 5.0
    t = np.arange(int(SR * duration)) / SR
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    spec = features.spectrogram(y, SR, duration)
    assert spec["bins"] == 64 and spec["frames"] == 100 and spec["fps"] == 20
    raw = np.frombuffer(base64.b64decode(spec["data"]), dtype=np.uint8)
    assert raw.size == 100 * 64 and raw.max() == 255


def test_four_minute_analysis_under_1_5_mb(validate):
    pytest.importorskip("librosa")
    duration = 240.0
    rng = np.random.default_rng(1)
    y = (0.1 * rng.standard_normal(int(SR * duration))).astype(np.float32)
    beats = 0.3 + np.arange(int(duration * 3)) / 3  # 180 bpm: lots of beats
    chords = [{"start": s, "end": s + 1.0, "chord": ["C:maj", "A:min7"][int(s) % 2]} for s in range(240)]
    doc = build_analysis(
        track_id="synthetic", source="file", source_ref=None, analyzer_version="test",
        duration=duration, beats=beats, downbeats=beats[::4], chords=chords, chroma=None,
        spectrogram=features.spectrogram(y, SR, duration),
        waveform_peaks=features.waveform_peaks(y, SR, duration),
    )
    assert doc["spectrogram"]["frames"] == 4800 and len(doc["waveformPeaks"]["peaks"]) == 4800
    assert validate(doc) == []
    size = len(json.dumps(doc, separators=(",", ":")).encode("utf-8"))
    assert size < 1.5 * 1024 * 1024, size
