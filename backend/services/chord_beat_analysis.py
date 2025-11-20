"""
Chord-Beat Analysis Script
Combines chord recognition with beat detection to link each chord to specific beats.
Uses the BeatTransformerDetectorService and ChordCNNLSTMDetectorService.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add backend to path
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from services.beat_transformer_service import BeatTransformerDetectorService
from services.librosa_detector_service import LibrosaDetectorService
from services.chord_cnn_lstm_service import ChordCNNLSTMDetectorService


class ChordBeatAnalyzer:
    """Analyzer that combines chord recognition with beat detection."""
    
    def __init__(self, chord_model_dir: str, use_librosa_fallback: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            chord_model_dir: Path to chord recognition model directory
            use_librosa_fallback: Use librosa as fallback if Beat Transformer fails
        """
        self.chord_service = ChordCNNLSTMDetectorService(chord_model_dir)
        self.beat_service = BeatTransformerDetectorService()
        self.librosa_service = LibrosaDetectorService() if use_librosa_fallback else None
        
        # Check if at least one beat detection method is available
        beat_available = self.beat_service.is_available()
        librosa_available = self.librosa_service.is_available() if self.librosa_service else False
        
        if not beat_available and not librosa_available:
            raise RuntimeError("No beat detection service available (tried Beat Transformer and Librosa)")
        
        if not self.chord_service.is_available():
            raise RuntimeError("Chord-CNN-LSTM is not available")
        
        # Determine which beat detector to use
        if beat_available:
            print("Chord-Beat Analyzer initialized (using Beat Transformer)")
        else:
            print("Chord-Beat Analyzer initialized (using Librosa fallback)")
        
        self.use_transformer = beat_available
    
    def extract_beats(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract beats and downbeats from audio using Beat Transformer or Librosa fallback.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (beats array, downbeats array) in seconds
        """
        print(f"\nExtracting beats from {Path(audio_path).name}...")
        
        # Try Beat Transformer first if available
        if self.use_transformer:
            result = self.beat_service.detect_beats(audio_path)
            
            # If Beat Transformer fails and librosa fallback is available, use it
            if not result["success"] and self.librosa_service:
                print(f"  Beat Transformer failed, falling back to Librosa...")
                result = self.librosa_service.detect_beats(audio_path)
        else:
            # Use librosa directly
            result = self.librosa_service.detect_beats(audio_path)
        
        if not result["success"]:
            raise RuntimeError(f"Beat detection failed: {result.get('error', 'Unknown error')}")
        
        beats = np.array(result["beats"])
        downbeats = np.array(result["downbeats"])
        
        print(f"  Model used: {result['model_name']}")
        print(f"  Detected tempo: {result['bpm']:.1f} BPM")
        print(f"  Time signature: {result['time_signature']}")
        print(f"  Found {len(beats)} beats and {len(downbeats)} downbeats")
        print(f"  Processing time: {result['processing_time']:.2f}s")
        
        return beats, downbeats
    
    def extract_chords(self, audio_path: str) -> List[Tuple[float, float, str]]:
        """
        Extract chord progressions from audio using the chord recognition service.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of tuples: (start_time, end_time, chord_label)
        """
        print(f"\nExtracting chords from {Path(audio_path).name}...")
        
        result = self.chord_service.recognize_chords(audio_path)
        
        if not result["success"]:
            raise RuntimeError(f"Chord recognition failed: {result.get('error', 'Unknown error')}")
        
        print(f"  Found {result['total_chords']} chord segments")
        print(f"  Processing time: {result['processing_time']:.2f}s")
        
        return result["chords"]
    
    def link_chords_to_beats(
        self, 
        chords: List[Dict[str, Any]], 
        beats: np.ndarray, 
        downbeats: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Link each chord segment to the beats that occur during it.
        
        Args:
            chords: List of chord dictionaries with 'start', 'end', 'chord' keys
            beats: Array of beat timestamps
            downbeats: Array of downbeat timestamps
        
        Returns:
            List of dictionaries with chord and beat information
        """
        print("\nLinking chords to beats...")
        
        linked_data = []
        
        for chord_info in chords:
            chord_start = chord_info['start']
            chord_end = chord_info['end']
            chord_label = chord_info['chord']
            
            # Find beats within this chord segment
            chord_beats = beats[(beats >= chord_start) & (beats < chord_end)]
            chord_downbeats = downbeats[(downbeats >= chord_start) & (downbeats < chord_end)]
            
            # Mark which beats are downbeats
            beat_info = []
            for beat_time in chord_beats:
                is_downbeat = any(np.isclose(beat_time, chord_downbeats, atol=0.05))
                beat_info.append({
                    'time': float(beat_time),
                    'is_downbeat': bool(is_downbeat)
                })
            
            linked_data.append({
                'chord': chord_label,
                'start_time': float(chord_start),
                'end_time': float(chord_end),
                'duration': float(chord_end - chord_start),
                'beats': beat_info,
                'num_beats': len(beat_info),
                'num_downbeats': int(np.sum([b['is_downbeat'] for b in beat_info]))
            })
        
        print(f"  Linked {len(linked_data)} chord segments to beats")
        return linked_data
    
    def analyze(self, audio_path: str, output_path: str):
        """
        Perform complete chord-beat analysis.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to output JSON file
        """
        print(f"\n{'='*60}")
        print(f"Chord-Beat Analysis")
        print(f"{'='*60}")
        
        # Extract beats
        beats, downbeats = self.extract_beats(audio_path)
        
        # Extract chords
        chords = self.extract_chords(audio_path)
        
        # Link chords to beats
        linked_data = self.link_chords_to_beats(chords, beats, downbeats)
        
        # Prepare output
        output = {
            'audio_file': str(Path(audio_path).name),
            'total_duration': float(chords[-1]['end']) if chords else 0.0,
            'total_beats': int(len(beats)),
            'total_downbeats': int(len(downbeats)),
            'total_chords': len(chords),
            'analysis': linked_data
        }
        
        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Analysis complete!")
        print(f"Output saved to: {output_path}")
        print(f"{'='*60}\n")
        
        # Print summary
        print("Summary:")
        print(f"  Duration: {output['total_duration']:.2f} seconds")
        print(f"  Beats: {output['total_beats']}")
        print(f"  Downbeats: {output['total_downbeats']}")
        print(f"  Chord segments: {output['total_chords']}")
        print(f"\nFirst 5 chord segments:")
        for i, segment in enumerate(linked_data[:5]):
            print(f"  {i+1}. {segment['chord']:10s} [{segment['start_time']:6.2f}s - {segment['end_time']:6.2f}s] "
                  f"({segment['num_beats']} beats, {segment['num_downbeats']} downbeats)")


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python chord_beat_analysis.py <audio_file> <output_file>")
        print("Example: python chord_beat_analysis.py data/song.mp3 outputs/analysis.json")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Set up paths
    script_dir = Path(__file__).parent.parent
    chord_model_dir = script_dir / "models" / "chord-recognition-model"
    
    if not chord_model_dir.exists():
        print(f"Error: Chord recognition model directory not found: {chord_model_dir}")
        sys.exit(1)
    
    # Create analyzer
    try:
        analyzer = ChordBeatAnalyzer(str(chord_model_dir))
        
        # Run analysis
        analyzer.analyze(audio_path, output_path)
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        sys.exit(1)


if __name__ == "__main__":
    main()
