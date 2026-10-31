"""
Chord-Beat Analysis: audio file -> analysis v1 (contracts/analysis.schema.json).
Uses LibrosaDetectorService (beats) and ChordCNNLSTMDetectorService (chords);
key / BPM / time signature come from services/features.py.
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Add worker to path
worker_dir = Path(__file__).parent.parent
if str(worker_dir) not in sys.path:
    sys.path.insert(0, str(worker_dir))

from services import features
from services.librosa_detector_service import LibrosaDetectorService
from services.chord_cnn_lstm_service import ChordCNNLSTMDetectorService

SCHEMA_VERSION = 1
SAMPLE_RATE = 22050
DEFAULT_ANALYZER_VERSION = os.environ.get("ANALYZER_VERSION", "0.1.0")
DEFAULT_CHORD_MODEL_DIR = os.environ.get(
    "CHORD_MODEL_DIR", str(worker_dir / "models" / "chord-recognition-model")
)


def build_analysis(
    *,
    track_id: str,
    source: str,
    source_ref: Optional[str],
    analyzer_version: str,
    duration: float,
    beats: Sequence[float],
    downbeats: Sequence[float],
    chords: Sequence[Dict[str, Any]],
    chroma: Optional[Sequence[float]] = None,
    analyzed_at: Optional[str] = None,
    spectrogram: Optional[Dict[str, Any]] = None,
    waveform_peaks: Optional[Dict[str, Any]] = None,
    silent: bool = False,
) -> Dict[str, Any]:
    """Assemble an analysis v1 document from raw model outputs (no I/O, no models).

    silent (near-silent input, features.is_near_silent): the beat tracker and chroma only see noise
    there, so key/bpm/timeSignature are null and there are no beats (T-119).
    """
    if silent:
        chroma, beats, downbeats = None, [], []
    return {
        "schemaVersion": SCHEMA_VERSION,
        "meta": {
            "trackId": track_id,
            "source": source,
            "sourceRef": source_ref,
            "durationSec": round(float(duration), 3),
            "analyzedAt": analyzed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "analyzerVersion": analyzer_version,
        },
        "key": features.estimate_key(chroma) if chroma is not None else None,
        "bpm": features.estimate_bpm(beats),
        "timeSignature": features.estimate_time_signature(beats, downbeats),
        "beats": features.build_beats(beats, downbeats),
        "chords": features.contiguous_chords(chords, float(duration)),
        "spectrogram": spectrogram,
        "waveformPeaks": waveform_peaks,
    }


class ChordBeatAnalyzer:
    """Analyzer that combines chord recognition with beat detection."""

    def __init__(self, chord_model_dir: str = DEFAULT_CHORD_MODEL_DIR):
        self.chord_service = ChordCNNLSTMDetectorService(chord_model_dir)
        # librosa is the beat path. madmom is not allowed: its pretrained
        # models are CC BY-NC-SA (PLAN.md R-7, see MODELS.md).
        self.librosa_service = LibrosaDetectorService()
        self.timings: Dict[str, float] = {}
        self._on_stage: Optional[Callable[[str], None]] = None

        if not self.librosa_service.is_available():
            raise RuntimeError("No beat detection service available (librosa is not installed)")

        if not self.chord_service.is_available():
            raise RuntimeError("Chord-CNN-LSTM is not available")
        # Weights load once per process, here, not on every job (T-118).
        print(f"Chord-CNN-LSTM weights loaded in {self.chord_service.preload():.2f}s")

    @contextmanager
    def _stage(self, name: str):
        if self._on_stage:
            self._on_stage(name)
        start = time.perf_counter()
        yield
        self.timings[name] = round(time.perf_counter() - start, 3)
        print(f"  [stage] {name}: {self.timings[name]:.2f}s")

    def extract_beats(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        result = self.librosa_service.detect_beats(audio_path)
        if not result["success"]:
            raise RuntimeError(f"Beat detection failed: {result.get('error', 'Unknown error')}")
        return np.array(result["beats"]), np.array(result["downbeats"])

    def extract_chords(self, audio_path: str) -> List[Dict[str, Any]]:
        result = self.chord_service.recognize_chords(audio_path)
        if not result["success"]:
            raise RuntimeError(f"Chord recognition failed: {result.get('error', 'Unknown error')}")
        return result["chords"]

    def analyze(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        *,
        track_id: Optional[str] = None,
        source: str = "file",
        source_ref: Optional[str] = None,
        analyzer_version: str = DEFAULT_ANALYZER_VERSION,
        on_stage: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline and return (and optionally write) an analysis v1 document.

        Per-stage wall times are left in self.timings (T-115). on_stage(name) is called as each
        stage starts (load, beats, chords, key, spectrogram, save); the queue consumer uses it
        to report progress.
        """
        import librosa

        self.timings = {}
        self._on_stage = on_stage
        audio_path = str(audio_path)
        with self._stage("load"):
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            duration = len(y) / sr
        with self._stage("beats"):
            beats, downbeats = self.extract_beats(audio_path)
        with self._stage("chords"):
            chords = self.extract_chords(audio_path)
        with self._stage("key"):
            chroma = features.mean_chroma(y, sr)
        with self._stage("spectrogram"):
            spec = features.spectrogram(y, sr, duration)
            peaks = features.waveform_peaks(y, sr, duration)

        doc = build_analysis(
            spectrogram=spec,
            waveform_peaks=peaks,
            track_id=track_id or Path(audio_path).stem,
            source=source,
            source_ref=source_ref,
            analyzer_version=analyzer_version,
            duration=duration,
            beats=beats,
            downbeats=downbeats,
            chords=chords,
            chroma=chroma,
            silent=features.is_near_silent(y),
        )

        if output_path:
            with self._stage("save"):
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        return doc


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Analyze an audio file into analysis v1 JSON.")
    parser.add_argument("audio_file")
    parser.add_argument("output_file")
    parser.add_argument("--track-id", help="meta.trackId (default: audio file name without extension)")
    parser.add_argument("--source", default="file", choices=["file", "recording", "youtube", "soundcloud"])
    parser.add_argument("--source-ref", default=None, help="YouTube videoId or SoundCloud URL")
    parser.add_argument("--analyzer-version", default=DEFAULT_ANALYZER_VERSION)
    parser.add_argument("--chord-model-dir", default=DEFAULT_CHORD_MODEL_DIR, help="or set CHORD_MODEL_DIR")
    args = parser.parse_args(argv)

    if not os.path.exists(args.audio_file):
        sys.exit(f"Error: Audio file not found: {args.audio_file}")
    if not Path(args.chord_model_dir).exists():
        sys.exit(f"Error: Chord recognition model directory not found: {args.chord_model_dir}")

    analyzer = ChordBeatAnalyzer(args.chord_model_dir)
    start = time.perf_counter()
    doc = analyzer.analyze(
        args.audio_file,
        args.output_file,
        track_id=args.track_id,
        source=args.source,
        source_ref=args.source_ref,
        analyzer_version=args.analyzer_version,
    )
    print(
        f"Wrote {args.output_file}: {doc['meta']['durationSec']:.1f}s, key={doc['key']}, bpm={doc['bpm']}, "
        f"timeSignature={doc['timeSignature']}, {len(doc['beats'])} beats, {len(doc['chords'])} chords, "
        f"total {time.perf_counter() - start:.2f}s"
    )


if __name__ == "__main__":
    main()
