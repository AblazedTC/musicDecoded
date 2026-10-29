"""T-119: no key for near-silent or keyless audio, no bpm without enough beats."""
import pytest

np = pytest.importorskip("numpy")

from services import features  # noqa: E402
from services.chord_beat_analysis import build_analysis  # noqa: E402

SR = 22050
rng = np.random.default_rng(0)


def tone_chord(seconds=10.0, freqs=(261.63, 329.63, 392.0), amp=0.2):
    """C major triad with a few harmonics: has a key and is clearly not silent."""
    t = np.arange(int(SR * seconds)) / SR
    return sum(amp / k * np.sin(2 * np.pi * f * k * t) for f in freqs for k in (1, 2, 3)).astype(np.float32)


@pytest.mark.parametrize("y,silent", [
    (np.zeros(SR * 5), True),
    (rng.normal(0, 0.001, SR * 5), True),  # -60 dBFS hiss: a phone recording of a quiet room
    (rng.normal(0, 0.1, SR * 5), False),  # -20 dBFS noise is loud, just keyless
    (tone_chord(), False),
    (np.array([]), True),
])
def test_near_silence(y, silent):
    assert features.is_near_silent(y) is silent


def test_rms_dbfs():
    assert features.rms_dbfs(np.full(100, 0.5)) == pytest.approx(-6.02, abs=0.01)
    assert features.rms_dbfs(np.zeros(10)) == pytest.approx(-200.0)


def test_low_correlation_chroma_has_no_key():
    # Close to flat with a little random variance: some key "wins", but with a weak correlation.
    chroma = 1.0 + rng.normal(0, 0.05, 12)
    best = max(np.corrcoef(chroma, np.roll(p, k))[0, 1] for p in (features.KS_MAJOR, features.KS_MINOR)
               for k in range(12))
    assert best < features.KEY_MIN_CONFIDENCE
    assert features.estimate_key(chroma) is None


@pytest.mark.parametrize("n,expected", [(0, None), (1, None), (features.MIN_BEATS - 1, None),
                                        (features.MIN_BEATS, 120.0)])
def test_bpm_needs_enough_beats(n, expected):
    beats = 0.5 + np.arange(n) * 0.5
    assert features.estimate_bpm(beats) == (pytest.approx(expected) if expected else None)


def test_silent_document_validates(validate):
    beats = 0.5 + np.arange(40) * 0.5
    chroma = np.array([1, 0.1, 0.5, 0.1, 0.8, 0.6, 0.1, 0.9, 0.1, 0.5, 0.1, 0.4])
    doc = build_analysis(track_id="t1", source="recording", source_ref=None, analyzer_version="test", duration=21.0,
                         beats=beats, downbeats=beats[::4], chords=[], chroma=chroma, silent=True)
    assert doc["key"] is None and doc["bpm"] is None and doc["timeSignature"] is None and doc["beats"] == []
    assert validate(doc) == []


# --- with librosa: real chroma of synthetic audio ---

def chroma_key(y):
    pytest.importorskip("librosa")
    return features.estimate_key(features.mean_chroma(np.asarray(y, dtype=np.float32), SR))


@pytest.mark.parametrize("name,y", [
    ("white noise", rng.normal(0, 0.1, SR * 10)),
    ("brown noise", np.cumsum(rng.normal(0, 0.01, SR * 10)) * 0.01),
])
def test_noise_has_no_key(name, y):
    assert chroma_key(y) is None, name


def test_triad_still_has_a_key():
    key = chroma_key(tone_chord())
    assert key is not None and key["confidence"] >= features.KEY_MIN_CONFIDENCE
