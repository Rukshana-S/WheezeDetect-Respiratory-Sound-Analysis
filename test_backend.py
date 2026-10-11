"""Comprehensive test script for WheezeDetect backend endpoints.
Works when executed from either the project root or the backend/ directory.
"""

import io
import os
import sys
import unittest
import numpy as np
import soundfile as sf

# Allow importing whether run from project root or inside backend/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from backend.app import app
except ImportError:
    from app import app


def create_wav_bytes(duration_sec: float, sr: int = 16000, freq: float = 440.0) -> io.BytesIO:
    """Generate in-memory WAV audio for testing."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Sine wave with harmonics
    data = (0.25 * np.sin(2 * np.pi * freq * t) + 0.1 * np.sin(2 * np.pi * freq * 2 * t)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, data, sr, format="WAV")
    bio.seek(0)
    return bio


class TestBackend(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_check(self):
        """Test /api/health returns 200 and model_loaded: True."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["model_name"], "Extra Trees Classifier")
        self.assertEqual(data["sample_rate"], 16000)
        self.assertAlmostEqual(data["threshold"], 0.225, places=3)

    def test_analyze_no_file(self):
        """Test /api/analyze without file part returns 400."""
        res = self.client.post("/api/analyze")
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("No file uploaded", data["error"])

    def test_analyze_empty_file(self):
        """Test /api/analyze with empty file returns 400."""
        data = {
            "file": (io.BytesIO(b""), "empty.wav")
        }
        res = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertFalse(json_data["success"])
        self.assertIn("empty", json_data["error"])

    def test_analyze_unsupported_format(self):
        """Test /api/analyze with non-wav file returns 400."""
        data = {
            "file": (io.BytesIO(b"dummy text content"), "audio.mp3")
        }
        res = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertFalse(json_data["success"])
        self.assertIn("Unsupported file format", json_data["error"])

    def test_analyze_audio_too_short(self):
        """Test /api/analyze with audio < 2.0s returns 400 with descriptive error."""
        wav_bytes = create_wav_bytes(duration_sec=1.2)
        data = {
            "file": (wav_bytes, "short_audio.wav")
        }
        res = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertFalse(json_data["success"])
        self.assertIn("shorter than the required window duration", json_data["error"])

    def test_analyze_valid_audio(self):
        """Test /api/analyze with valid 4.0s WAV runs real inference."""
        wav_bytes = create_wav_bytes(duration_sec=4.0, freq=350.0)
        data = {
            "file": (wav_bytes, "test_respiratory.wav")
        }
        res = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()

        self.assertTrue(json_data["success"])
        self.assertIn(json_data["prediction"], ["Wheeze-like Pattern Detected", "No Wheeze-like Pattern Detected"])
        self.assertIsInstance(json_data["score"], float)
        self.assertAlmostEqual(json_data["threshold"], 0.225, places=3)
        self.assertEqual(json_data["audio_duration"], 4.0)
        self.assertEqual(json_data["segments_analyzed"], 3)
        self.assertEqual(len(json_data["segment_scores"]), 3)
        self.assertTrue(all(isinstance(s, float) for s in json_data["segment_scores"]))
        self.assertEqual(json_data["model_name"], "Extra Trees Classifier")
        self.assertEqual(json_data["source"], "backend")


if __name__ == "__main__":
    unittest.main()
