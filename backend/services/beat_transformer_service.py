"""
Beat Transformer detector service.

This module provides a wrapper around the BeatTransformerDetector model
with a normalized interface for the beat detection service.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class BeatTransformerDetectorService:
    """
    Service wrapper for BeatTransformerDetector with normalized interface.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize the Beat Transformer detector service.

        Args:
            checkpoint_path: Path to the model checkpoint file
        """
        self.checkpoint_path = checkpoint_path
        self._detector = None
        self._available = None

    def is_available(self) -> bool:
        """
        Check if Beat Transformer is available.

        Returns:
            bool: True if the detector can be used
        """
        if self._available is not None:
            return self._available

        try:
            # Try to import the Beat Transformer detector
            from models.beat_transformer import BeatTransformerDetector, is_beat_transformer_available
            self._available = is_beat_transformer_available()
            return self._available
        except ImportError as e:
            self._available = False
            return False

    def _get_detector(self):
        """
        Get or create the detector instance.

        Returns:
            BeatTransformerDetector instance
        """
        if self._detector is None:
            if not self.is_available():
                raise RuntimeError("Beat Transformer is not available")

            try:
                from models.beat_transformer import BeatTransformerDetector
                self._detector = BeatTransformerDetector(self.checkpoint_path)
            except Exception as e:
                raise

        return self._detector

    def detect_beats(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Detect beats in an audio file using Beat Transformer.

        Args:
            file_path: Path to the audio file
            **kwargs: Additional parameters (unused for Beat Transformer)

        Returns:
            Dict containing normalized beat detection results:
            {
                "success": bool,
                "beats": List[float],           # Beat positions in seconds
                "downbeats": List[float],       # Downbeat positions in seconds
                "total_beats": int,
                "total_downbeats": int,
                "bpm": float,
                "time_signature": str,
                "duration": float,
                "model_used": str,
                "model_name": str,
                "processing_time": float,
                "error": str (if success=False)
            }
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Beat Transformer is not available",
                "model_used": "beat-transformer",
                "model_name": "Beat-Transformer"
            }

        try:
            detector = self._get_detector()

            # Run beat detection
            result = detector.detect_beats(file_path)

            # Normalize the result format
            if result.get("success"):
                return {
                    "success": True,
                    "beats": result.get("beats", []),
                    "downbeats": result.get("downbeats", []),
                    "total_beats": result.get("total_beats", 0),
                    "total_downbeats": result.get("total_downbeats", 0),
                    "bpm": result.get("bpm", 120.0),
                    "time_signature": result.get("time_signature", "4/4"),
                    "duration": result.get("duration", 0.0),
                    "model_used": "beat-transformer",
                    "model_name": "Beat-Transformer",
                    "processing_time": result.get("processing_time", 0.0)
                }
            else:
                error_msg = result.get("error", "Unknown error in Beat Transformer detection")
                return {
                    "success": False,
                    "error": error_msg,
                    "model_used": "beat-transformer",
                    "model_name": "Beat-Transformer"
                }

        except Exception as e:
            error_msg = f"Beat Transformer detection error: {str(e)}"
            return {
                "success": False,
                "error": error_msg,
                "model_used": "beat-transformer",
                "model_name": "Beat-Transformer"
            }

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get device information for the Beat Transformer detector.

        Returns:
            Dict containing device information
        """
        if not self.is_available():
            return {"error": "Beat Transformer not available"}

        try:
            detector = self._get_detector()
            return detector.get_device_info()
        except Exception as e:
            return {"error": str(e)}