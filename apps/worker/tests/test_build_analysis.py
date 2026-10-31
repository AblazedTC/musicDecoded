"""build_analysis() output validates against contracts/analysis.schema.json."""
import pytest

np = pytest.importorskip("numpy")

from services.chord_beat_analysis import build_analysis  # noqa: E402


def make_doc(**overrides):
    beats = 0.5 + np.arange(40) * 0.5
    kwargs = dict(
        track_id="t1",
        source="file",
        source_ref=None,
        analyzer_version="test",
        duration=21.0,
        beats=beats,
        downbeats=beats[::4],
        chords=[{"start": 0.2, "end": 8.0, "chord": "C:maj"}, {"start": 9.0, "end": 20.9, "chord": "A:min7"}],
        chroma=np.array([1, 0.1, 0.5, 0.1, 0.8, 0.6, 0.1, 0.9, 0.1, 0.5, 0.1, 0.4]),
    )
    kwargs.update(overrides)
    return build_analysis(**kwargs)


def test_document_shape():
    doc = make_doc()
    assert doc["schemaVersion"] == 1
    assert doc["meta"]["trackId"] == "t1" and doc["meta"]["sourceRef"] is None
    assert doc["key"]["tonic"] == "C" and doc["key"]["mode"] == "major"
    assert doc["bpm"] == pytest.approx(120.0)
    assert doc["timeSignature"] == {"numerator": 4, "denominator": 4}
    assert doc["chords"][0] == {"start": 0.0, "end": 0.2, "label": "N"}
    assert doc["chords"][-1]["end"] == 21.0


def test_document_validates(validate):
    assert validate(make_doc()) == []


def test_degenerate_input_validates(validate):
    doc = make_doc(beats=[], downbeats=[], chords=[], chroma=None, source="youtube", source_ref="abc123")
    assert doc["key"] is None and doc["bpm"] is None and doc["timeSignature"] is None
    assert doc["chords"] == [{"start": 0.0, "end": 21.0, "label": "N"}]
    assert validate(doc) == []
