"""WheezeDetect Machine Learning Inference Pipeline.

Loads the trained Extra Trees model artifact and performs feature extraction
and inference on respiratory audio files.
"""

import io
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

# Suppress sklearn unpickling version warnings
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

import joblib

logger = logging.getLogger(__name__)


class AudioProcessingError(Exception):
    """Raised when audio decoding, resampling, or validation fails."""
    pass


class AudioTooShortError(AudioProcessingError):
    """Raised when audio duration is shorter than the minimum segment window."""
    pass


class WheezeModelPipeline:
    """Inference pipeline for segment-based wheeze detection."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "improved_wheeze_model.joblib")
        self.model_path = model_path
        self.model = None
        self.feature_columns: List[str] = []
        self.sample_rate: int = 16000
        self.window_seconds: float = 2.0
        self.stride_seconds: float = 1.0
        self.aggregation: str = "top_1"
        self.threshold: float = 0.225
        self.model_name: str = "Extra Trees Classifier"
        self._is_loaded: bool = False

    def load(self) -> None:
        """Load and validate the serialized model artifact."""
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")

        logger.info(f"Loading model artifact from {self.model_path}...")
        artifact = joblib.load(self.model_path)

        if not isinstance(artifact, dict):
            raise ValueError(f"Expected model artifact to be a dict, got {type(artifact)}")

        required_keys = ["model", "feature_columns"]
        for key in required_keys:
            if key not in artifact:
                raise KeyError(f"Missing required key '{key}' in model artifact.")

        self.model = artifact["model"]
        self.feature_columns = list(artifact["feature_columns"])
        self.sample_rate = int(artifact.get("sample_rate", 16000))
        self.window_seconds = float(artifact.get("window_seconds", 2.0))
        self.stride_seconds = float(artifact.get("stride_seconds", 1.0))
        self.aggregation = str(artifact.get("aggregation", "top_1"))
        self.threshold = float(artifact.get("threshold", 0.225))

        raw_name = artifact.get("model_name", "Extra Trees Classifier")
        if raw_name == "ExtraTrees":
            self.model_name = "Extra Trees Classifier"
        else:
            self.model_name = str(raw_name)

        if len(self.feature_columns) != 264:
            raise ValueError(
                f"Expected 264 feature columns in model artifact, found {len(self.feature_columns)}"
            )

        self._is_loaded = True
        logger.info(
            f"Successfully loaded '{self.model_name}' with {len(self.feature_columns)} features. "
            f"SR={self.sample_rate}Hz, Window={self.window_seconds}s, Stride={self.stride_seconds}s, "
            f"Threshold={self.threshold:.4f}, Aggregation={self.aggregation}"
        )

    def is_loaded(self) -> bool:
        """Check if the model pipeline is loaded and ready."""
        return self._is_loaded and self.model is not None

    def read_audio(self, file_source: Any) -> Tuple[np.ndarray, float, int]:
        """Safely read and decode audio file or bytes into mono 16kHz float array.

        Returns:
            Tuple of (audio_signal, duration_in_seconds, original_sample_rate)
        """
        try:
            orig_sr = self.sample_rate
            if isinstance(file_source, bytes):
                file_source = io.BytesIO(file_source)

            # Probe original sample rate if stream supports seek
            if hasattr(file_source, "seek") and hasattr(file_source, "tell"):
                try:
                    pos = file_source.tell()
                    info = sf.info(file_source)
                    orig_sr = info.samplerate
                    file_source.seek(pos)
                except Exception:
                    pass

            # Resample to 16 kHz mono using high quality soxr resampler
            y, sr = librosa.load(
                file_source,
                sr=self.sample_rate,
                res_type="soxr_hq",
                mono=True
            )
            duration = float(len(y) / sr) if sr > 0 else 0.0
            return y, duration, orig_sr
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            raise AudioProcessingError(f"Could not decode audio file: {str(e)}") from e

    def extract_segment_features(self, segment: np.ndarray) -> Dict[str, float]:
        """Extract exact 264 features for a 2-second audio segment.

        Librosa extraction parameters match the training pipeline:
        - n_fft = 512
        - hop_length = 160
        - n_mfcc = 40
        - frame_length = 512 (matches STFT window length for zcr and rms)
        """
        n_fft = 512
        hop_length = 160
        n_mfcc = 40
        sr = self.sample_rate

        features: Dict[str, float] = {}

        # 1. 40 MFCCs (mean and std) -> 80 features
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        for i in range(n_mfcc):
            features[f"mfcc_mean_{i}"] = float(np.mean(mfcc[i]))
        for i in range(n_mfcc):
            features[f"mfcc_std_{i}"] = float(np.std(mfcc[i]))

        # 2. First-order Delta MFCCs (mean and std) -> 80 features
        delta = librosa.feature.delta(mfcc)
        for i in range(n_mfcc):
            features[f"delta_mean_{i}"] = float(np.mean(delta[i]))
        for i in range(n_mfcc):
            features[f"delta_std_{i}"] = float(np.std(delta[i]))

        # 3. Second-order Delta MFCCs (mean and std) -> 80 features
        delta2 = librosa.feature.delta(mfcc, order=2)
        for i in range(n_mfcc):
            features[f"delta2_mean_{i}"] = float(np.mean(delta2[i]))
        for i in range(n_mfcc):
            features[f"delta2_std_{i}"] = float(np.std(delta2[i]))

        # 4. Spectral Centroid (mean and std) -> 2 features
        centroid = librosa.feature.spectral_centroid(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features["spectral_centroid_mean"] = float(np.mean(centroid))
        features["spectral_centroid_std"] = float(np.std(centroid))

        # 5. Spectral Bandwidth (mean and std) -> 2 features
        bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
        features["spectral_bandwidth_std"] = float(np.std(bandwidth))

        # 6. Spectral Rolloff (mean and std) -> 2 features
        rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features["spectral_rolloff_mean"] = float(np.mean(rolloff))
        features["spectral_rolloff_std"] = float(np.std(rolloff))

        # 7. Zero-Crossing Rate (mean and std) -> 2 features (frame_length=n_fft)
        zcr = librosa.feature.zero_crossing_rate(y=segment, frame_length=n_fft, hop_length=hop_length)
        features["zero_crossing_rate_mean"] = float(np.mean(zcr))
        features["zero_crossing_rate_std"] = float(np.std(zcr))

        # 8. RMS Energy (mean and std) -> 2 features (frame_length=n_fft)
        rms = librosa.feature.rms(y=segment, frame_length=n_fft, hop_length=hop_length)
        features["rms_energy_mean"] = float(np.mean(rms))
        features["rms_energy_std"] = float(np.std(rms))

        # 9. Spectral Contrast across 7 frequency bands (mean and std) -> 14 features
        contrast = librosa.feature.spectral_contrast(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
        for i in range(contrast.shape[0]):
            features[f"spectral_contrast_mean_{i}"] = float(np.mean(contrast[i]))
            features[f"spectral_contrast_std_{i}"] = float(np.std(contrast[i]))

        return features

    def segment_audio(self, y: np.ndarray) -> List[np.ndarray]:
        """Split audio signal into 2-second windows with 1-second stride."""
        window_samples = int(self.window_seconds * self.sample_rate)
        stride_samples = int(self.stride_seconds * self.sample_rate)

        if len(y) < window_samples:
            duration = len(y) / self.sample_rate
            raise AudioTooShortError(
                f"Audio duration ({duration:.2f}s) is shorter than the required window duration "
                f"({self.window_seconds:.1f}s). At least {self.window_seconds:.1f} seconds of audio is required."
            )

        segments = []
        for start in range(0, len(y) - window_samples + 1, stride_samples):
            segments.append(y[start : start + window_samples])

        return segments

    def analyze_audio(self, file_source: Any) -> Dict[str, Any]:
        """Run the end-to-end wheeze detection pipeline on an audio input."""
        if not self.is_loaded():
            raise RuntimeError("Model pipeline is not loaded.")

        # 1. Read & resample
        y, audio_duration, orig_sr = self.read_audio(file_source)

        # 2. Segment
        segments = self.segment_audio(y)

        # 3. Extract features for all segments
        segment_feature_dicts = [self.extract_segment_features(seg) for seg in segments]

        # 4. Construct DataFrame aligned strictly to model's feature_columns
        df = pd.DataFrame(segment_feature_dicts)

        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing extracted features: {missing_cols}")

        df = df[self.feature_columns]

        # 5. Model Inference (select positive class label dynamically)
        classes = list(getattr(self.model, "classes_", [0, 1]))
        pos_class_idx = classes.index(1) if 1 in classes else 1

        probs = self.model.predict_proba(df.values)
        wheeze_probs = probs[:, pos_class_idx]
        segment_scores = [round(float(p), 4) for p in wheeze_probs]

        # 6. Aggregation (top-1 segment score)
        if self.aggregation == "top_1":
            overall_score = float(np.max(wheeze_probs))
        else:
            overall_score = float(np.max(wheeze_probs))

        # 7. Apply threshold
        is_wheeze = overall_score >= self.threshold
        prediction_text = (
            "Wheeze-like Pattern Detected" if is_wheeze else "No Wheeze-like Pattern Detected"
        )

        logger.info(
            f"\n--- Diagnostic Inference Output ---"
            f"\n  Original sample rate: {orig_sr} Hz | Resampled rate: {self.sample_rate} Hz"
            f"\n  Audio samples: {len(y)} | Duration: {audio_duration:.2f} s"
            f"\n  Extracted segments: {len(segments)}"
            f"\n  Feature matrix shape: {df.shape} (Matches 264 saved columns: True)"
            f"\n  Model classes: {classes} | Positive class index: {pos_class_idx}"
            f"\n  Segment positive-class scores: {segment_scores}"
            f"\n  Top-1 aggregated score: {overall_score:.4f} (rounded: {round(overall_score, 3)})"
            f"\n  Threshold: {self.threshold:.4f} | Final prediction: {prediction_text}"
            f"\n-----------------------------------"
        )

        return {
            "success": True,
            "prediction": prediction_text,
            "score": round(overall_score, 4),
            "threshold": round(self.threshold, 4),
            "segments_analyzed": len(segments),
            "audio_duration": round(audio_duration, 2),
            "segment_scores": segment_scores,
            "model_name": self.model_name,
            "source": "backend"
        }
