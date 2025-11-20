"""
Beat Transformer model wrapper.

This module provides a wrapper around the Beat-Transformer model
for beat and downbeat detection.
"""

import os
import sys
import time
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple


def is_beat_transformer_available() -> bool:
    """Check if Beat Transformer dependencies are available."""
    try:
        import torch
        import librosa
        import numpy as np
        return True
    except ImportError:
        return False


class BeatTransformerDetector:
    """
    Wrapper for the Beat-Transformer model for beat detection.
    """
    
    def __init__(self, checkpoint_path: str = None):
        """
        Initialize the Beat Transformer detector.
        
        Args:
            checkpoint_path: Path to checkpoint directory. If None, uses default path.
        """
        if checkpoint_path is None:
            # Default to the Beat-Transformer checkpoint directory
            backend_dir = Path(__file__).parent.parent
            checkpoint_path = backend_dir / "models" / "Beat-Transformer" / "checkpoint"
        
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fps = 44100 / 1024  # Frames per second for the model
        self.models = []
        
        # Add Beat-Transformer code to path
        beat_transformer_code = self.checkpoint_path.parent / "code"
        if str(beat_transformer_code) not in sys.path:
            sys.path.insert(0, str(beat_transformer_code))
        
        self._load_models()
    
    def _load_models(self):
        """Load the Beat Transformer models."""
        try:
            from DilatedTransformer import Demixed_DilatedTransformerModel
            
            # Load multiple model folds for ensemble
            for fold in range(5):  # Use 5 folds for good accuracy/speed balance
                checkpoint_file = self.checkpoint_path / f"fold_{fold}_trf_param.pt"
                if checkpoint_file.exists():
                    model = Demixed_DilatedTransformerModel(
                        attn_len=5,
                        instr=5,
                        ntoken=2,
                        dmodel=256,
                        nhead=8,
                        d_hid=1024,
                        nlayers=9,
                        norm_first=True
                    )
                    checkpoint = torch.load(checkpoint_file, map_location=self.device)
                    model.load_state_dict(
                        checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                    )
                    model.to(self.device)
                    model.eval()
                    self.models.append(model)
            
            if not self.models:
                raise RuntimeError("No Beat Transformer checkpoints found")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load Beat Transformer models: {e}")
    
    def _prepare_audio(self, audio_path: str) -> Tuple[torch.Tensor, float]:
        """
        Load and prepare audio for the model.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (spectrogram tensor, duration in seconds)
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=44100)
        duration = len(y) / sr
        
        # Create mel spectrogram (126 mel bins as expected by model)
        spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=1024, n_mels=126
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        # Transpose and normalize
        spec_db = spec_db.T  # (time, 126)
        spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-8)
        
        # Stack for 5 instrument channels (simplified - using same spec for all)
        # Shape: (5, time, 126)
        demixed_spec = np.stack([spec_db] * 5, axis=0)
        
        # Convert to tensor with batch dimension: (1, 5, time, 126)
        spec_tensor = torch.from_numpy(demixed_spec).float().unsqueeze(0).to(self.device)
        
        return spec_tensor, duration
    
    def _run_inference(self, spec_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on the spectrogram.
        
        Args:
            spec_tensor: Input spectrogram tensor
            
        Returns:
            Tuple of (beat_activations, downbeat_activations)
        """
        beat_activations = []
        downbeat_activations = []
        
        with torch.no_grad():
            for model in self.models:
                pred, _ = model(spec_tensor)
                beat_act = torch.sigmoid(pred[0, :, 0]).cpu().numpy()
                downbeat_act = torch.sigmoid(pred[0, :, 1]).cpu().numpy()
                beat_activations.append(beat_act)
                downbeat_activations.append(downbeat_act)
        
        # Average predictions from all models
        avg_beat_act = np.mean(beat_activations, axis=0)
        avg_downbeat_act = np.mean(downbeat_activations, axis=0)
        
        return avg_beat_act, avg_downbeat_act
    
    def _extract_beats_from_activations(
        self, 
        beat_act: np.ndarray, 
        downbeat_act: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract beat positions from activation functions.
        
        Args:
            beat_act: Beat activation array
            downbeat_act: Downbeat activation array
            
        Returns:
            Tuple of (beat timestamps, downbeat timestamps)
        """
        from scipy.signal import find_peaks
        
        # Find peaks in beat activation
        # Use much lower thresholds - use mean as baseline, not percentile
        # This ensures we catch most beats even with lower activations
        beat_mean = np.mean(beat_act)
        beat_std = np.std(beat_act)
        beat_threshold = max(0.05, beat_mean - 0.5 * beat_std)  # Very sensitive threshold
        
        beat_peaks, _ = find_peaks(
            beat_act, 
            height=beat_threshold, 
            distance=int(0.15 * self.fps),  # Min 0.15s between beats (allows up to 400 BPM)
            prominence=0.02  # Require slight prominence to avoid noise
        )
        beats = beat_peaks / self.fps
        
        # Find peaks in downbeat activation
        downbeat_mean = np.mean(downbeat_act)
        downbeat_std = np.std(downbeat_act)
        downbeat_threshold = max(0.08, downbeat_mean)
        
        downbeat_peaks, _ = find_peaks(
            downbeat_act,
            height=downbeat_threshold,
            distance=int(0.6 * self.fps),  # Min 0.6s between downbeats
            prominence=0.03
        )
        downbeats = downbeat_peaks / self.fps
        
        return beats, downbeats
    
    def _estimate_tempo(self, beats: np.ndarray) -> float:
        """
        Estimate tempo from beat positions.
        
        Args:
            beats: Array of beat timestamps
            
        Returns:
            Estimated BPM
        """
        if len(beats) < 2:
            return 120.0  # Default tempo
        
        # Calculate inter-beat intervals
        intervals = np.diff(beats)
        # Remove outliers
        median_interval = np.median(intervals)
        valid_intervals = intervals[np.abs(intervals - median_interval) < median_interval * 0.5]
        
        if len(valid_intervals) == 0:
            return 120.0
        
        # Convert to BPM
        avg_interval = np.mean(valid_intervals)
        bpm = 60.0 / avg_interval
        
        return float(bpm)
    
    def detect_beats(self, file_path: str) -> Dict[str, Any]:
        """
        Detect beats and downbeats in an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dict containing beat detection results
        """
        start_time = time.time()
        
        try:
            # Prepare audio
            spec_tensor, duration = self._prepare_audio(file_path)
            
            # Run inference
            beat_act, downbeat_act = self._run_inference(spec_tensor)
            
            # Extract beat positions
            beats, downbeats = self._extract_beats_from_activations(beat_act, downbeat_act)
            
            # Estimate tempo
            bpm = self._estimate_tempo(beats)
            
            # Estimate time signature based on downbeat spacing
            if len(downbeats) > 1:
                beats_per_bar = len(beats) / max(len(downbeats), 1)
                if 3.5 <= beats_per_bar < 4.5:
                    time_signature = "4/4"
                elif 2.5 <= beats_per_bar < 3.5:
                    time_signature = "3/4"
                else:
                    time_signature = "4/4"  # Default
            else:
                time_signature = "4/4"
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "beats": beats.tolist(),
                "downbeats": downbeats.tolist(),
                "total_beats": len(beats),
                "total_downbeats": len(downbeats),
                "bpm": bpm,
                "time_signature": time_signature,
                "duration": duration,
                "processing_time": processing_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "num_models": len(self.models)
        }
