"""Key / BPM / time signature / beats / chords on synthetic inputs (numpy only)."""
import pytest

np = pytest.importorskip("numpy")

from services import features  # noqa: E402


def scale_chroma(pitch_classes, weights):
    c = np.full(12, 0.05)
    for pc, w in zip(pitch_classes, weights):
        c[pc] += w
    return c


C_MAJOR = scale_chroma([0, 2, 4, 5, 7, 9, 11], [1.0, 0.5, 0.8, 0.6, 0.9, 0.5, 0.4])


def test_key_c_major():
    key = features.estimate_key(C_MAJOR)
    assert key["tonic"] == "C" and key["mode"] == "major"
    assert 0 < key["confidence"] <= 1


def test_key_a_minor():
    chroma = scale_chroma([9, 11, 0, 2, 4, 5, 7], [1.0, 0.4, 0.8, 0.5, 0.9, 0.5, 0.5])
    key = features.estimate_key(chroma)
    assert key["tonic"] == "A" and key["mode"] == "minor"


def test_key_follows_transposition():
    assert features.estimate_key(np.roll(C_MAJOR, 7))["tonic"] == "G"


def test_key_none_for_flat_chroma():
    assert features.estimate_key(np.ones(12)) is None
    assert features.estimate_key(np.zeros(12)) is None


def grid(bpm, n, start=0.5):
    return start + np.arange(n) * 60.0 / bpm


def test_bpm_120():
    assert features.estimate_bpm(grid(120, 64)) == pytest.approx(120.0)


def test_bpm_robust_to_one_missed_beat():
    assert features.estimate_bpm(np.delete(grid(120, 64), 10)) == pytest.approx(120.0)


def test_bpm_none_without_enough_beats():
    assert features.estimate_bpm([1.0]) is None
    assert features.estimate_bpm([]) is None


@pytest.mark.parametrize("beats_per_bar", [3, 4])
def test_time_signature(beats_per_bar):
    beats = grid(120, beats_per_bar * 16)
    expected = {"numerator": beats_per_bar, "denominator": 4}
    assert features.estimate_time_signature(beats, beats[::beats_per_bar]) == expected


def test_time_signature_with_jitter():
    rng = np.random.default_rng(0)
    beats = grid(120, 48) + rng.normal(0, 0.01, 48)
    assert features.estimate_time_signature(beats, beats[::3])["numerator"] == 3


def test_time_signature_none_without_bars():
    assert features.estimate_time_signature(grid(120, 8), [0.5]) is None


def test_build_beats_sorted_and_flagged():
    out = features.build_beats([2.0, 0.5, 1.0, 1.5], [0.51, 2.0])
    assert [b["t"] for b in out] == [0.5, 1.0, 1.5, 2.0]
    assert [b["downbeat"] for b in out] == [True, False, False, True]


def test_contiguous_chords_fills_gaps_and_trims_overlaps():
    chords = [
        {"start": 1.0, "end": 3.0, "chord": "C:maj"},
        {"start": 2.5, "end": 5.0, "chord": "A:min"},  # overlaps previous
        {"start": 6.0, "end": 7.0, "chord": "G"},  # gap before, bare root
        {"start": 7.0, "end": 8.0, "chord": "weird"},  # not Harte
    ]
    out = features.contiguous_chords(chords, duration=10.0)
    assert [c["label"] for c in out] == ["N", "C:maj", "A:min", "N", "G:maj", "X", "N"]
    assert out[0]["start"] == 0 and out[-1]["end"] == 10.0
    for a, b in zip(out, out[1:]):
        assert a["end"] == b["start"]


def test_contiguous_chords_merges_and_clips_to_duration():
    chords = [
        {"start": 0, "end": 2, "chord": "N"},
        {"start": 2, "end": 4, "chord": "N"},
        {"start": 4, "end": 12, "chord": "D:maj7"},
    ]
    assert features.contiguous_chords(chords, 10.0) == [
        {"start": 0.0, "end": 4.0, "label": "N"},
        {"start": 4.0, "end": 10.0, "label": "D:maj7"},
    ]
