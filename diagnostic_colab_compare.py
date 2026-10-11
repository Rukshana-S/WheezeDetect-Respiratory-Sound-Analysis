"""Stage-by-Stage Diagnostic Script for WheezeDetect Pipeline Comparison.

This script runs the 8 verification stages required by the specification.
It can be executed locally and also pasted directly into Google Colab
to produce an identical side-by-side verification report.
"""

import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import joblib

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "models", "improved_wheeze_model.joblib")


def get_sha256(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_stage_by_stage_diagnostic(filepath: str, model_path: str = MODEL_PATH):
    print("=" * 80)
    print(f"DIAGNOSTIC REPORT FOR: {os.path.basename(filepath)}")
    print("=" * 80)

    # -------------------------------------------------------------
    # Stage 1: File Integrity & SHA-256
    # -------------------------------------------------------------
    file_sha256 = get_sha256(filepath)
    file_size_bytes = os.path.getsize(filepath)
    print(f"[Stage 1] File Integrity:")
    print(f"  File: {os.path.basename(filepath)}")
    print(f"  Size: {file_size_bytes:,} bytes")
    print(f"  SHA-256: {file_sha256}")

    # -------------------------------------------------------------
    # Stage 2: Audio Ingestion & Resampling
    # -------------------------------------------------------------
    info = sf.info(filepath)
    orig_sr = info.samplerate
    channels = info.channels
    orig_frames = info.frames
    duration_sec = info.duration

    print(f"\n[Stage 2] Audio Properties:")
    print(f"  Original Sample Rate: {orig_sr} Hz")
    print(f"  Channels: {channels} ({'mono' if channels == 1 else 'stereo'})")
    print(f"  Total Frames: {orig_frames:,}")
    print(f"  Duration: {duration_sec:.2f} seconds")

    # Load and resample to 16,000 Hz mono
    target_sr = 16000
    y, sr = librosa.load(filepath, sr=target_sr, mono=True)
    print(f"  Resampled Audio Length: {len(y):,} samples at {sr} Hz")
    print(f"  Sample Value Range: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.6f}, std={y.std():.4f}")

    # -------------------------------------------------------------
    # Stage 3: Windowing and Segmentation
    # -------------------------------------------------------------
    win_samples = int(2.0 * sr)     # 32,000 samples (2.0s)
    stride_samples = int(1.0 * sr)  # 16,000 samples (1.0s)
    segments = []
    seg_ranges = []

    for start in range(0, len(y) - win_samples + 1, stride_samples):
        end = start + win_samples
        segments.append(y[start:end])
        seg_ranges.append((start, end))

    print(f"\n[Stage 3] Segmentation:")
    print(f"  Window: {win_samples} samples (2.0s) | Stride: {stride_samples} samples (1.0s)")
    print(f"  Segments Extracted: {len(segments)}")
    print(f"  Segment 1 Sample Range: [{seg_ranges[0][0]}:{seg_ranges[0][1]}]")
    print(f"  Segment {len(segments)} Sample Range: [{seg_ranges[-1][0]}:{seg_ranges[-1][1]}]")

    # -------------------------------------------------------------
    # Stage 4: Feature Extraction (264 Features)
    # -------------------------------------------------------------
    n_fft = 512
    hop_length = 160
    n_mfcc = 40

    feature_rows = []
    for i, seg in enumerate(segments):
        f = {}
        # 1. 40 MFCC mean & std (80)
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        for idx in range(n_mfcc): f[f"mfcc_mean_{idx}"] = float(np.mean(mfcc[idx]))
        for idx in range(n_mfcc): f[f"mfcc_std_{idx}"] = float(np.std(mfcc[idx]))

        # 2. Delta MFCC mean & std (80)
        delta = librosa.feature.delta(mfcc)
        for idx in range(n_mfcc): f[f"delta_mean_{idx}"] = float(np.mean(delta[idx]))
        for idx in range(n_mfcc): f[f"delta_std_{idx}"] = float(np.std(delta[idx]))

        # 3. Delta2 MFCC mean & std (80)
        delta2 = librosa.feature.delta(mfcc, order=2)
        for idx in range(n_mfcc): f[f"delta2_mean_{idx}"] = float(np.mean(delta2[idx]))
        for idx in range(n_mfcc): f[f"delta2_std_{idx}"] = float(np.std(delta2[idx]))

        # 4. Spectral Centroid (2)
        cent = librosa.feature.spectral_centroid(y=seg, sr=sr, n_fft=n_fft, hop_length=hop_length)
        f["spectral_centroid_mean"] = float(np.mean(cent))
        f["spectral_centroid_std"] = float(np.std(cent))

        # 5. Spectral Bandwidth (2)
        bw = librosa.feature.spectral_bandwidth(y=seg, sr=sr, n_fft=n_fft, hop_length=hop_length)
        f["spectral_bandwidth_mean"] = float(np.mean(bw))
        f["spectral_bandwidth_std"] = float(np.std(bw))

        # 6. Spectral Rolloff (2)
        ro = librosa.feature.spectral_rolloff(y=seg, sr=sr, n_fft=n_fft, hop_length=hop_length)
        f["spectral_rolloff_mean"] = float(np.mean(ro))
        f["spectral_rolloff_std"] = float(np.std(ro))

        # 7. Zero Crossing Rate (2)
        zcr = librosa.feature.zero_crossing_rate(y=seg, hop_length=hop_length)
        f["zero_crossing_rate_mean"] = float(np.mean(zcr))
        f["zero_crossing_rate_std"] = float(np.std(zcr))

        # 8. RMS Energy (2)
        rms = librosa.feature.rms(y=seg, hop_length=hop_length)
        f["rms_energy_mean"] = float(np.mean(rms))
        f["rms_energy_std"] = float(np.std(rms))

        # 9. Spectral Contrast (14)
        sc = librosa.feature.spectral_contrast(y=seg, sr=sr, n_fft=n_fft, hop_length=hop_length)
        for idx in range(sc.shape[0]):
            f[f"spectral_contrast_mean_{idx}"] = float(np.mean(sc[idx]))
            f[f"spectral_contrast_std_{idx}"] = float(np.std(sc[idx]))

        feature_rows.append(f)

    df = pd.DataFrame(feature_rows)

    # Load artifact
    artifact = joblib.load(model_path)
    expected_cols = artifact["feature_columns"]
    df = df[expected_cols]

    print(f"\n[Stage 4] Feature Extraction:")
    print(f"  Feature Matrix Shape: {df.shape}")
    print(f"  Matches Expected 264 Columns: {list(df.columns) == expected_cols}")
    print(f"  Sample Feature Values for Segment 1:")
    print(f"    mfcc_mean_0:              {df.loc[0, 'mfcc_mean_0']:.4f}")
    print(f"    spectral_centroid_mean:   {df.loc[0, 'spectral_centroid_mean']:.4f}")
    print(f"    rms_energy_mean:          {df.loc[0, 'rms_energy_mean']:.4f}")
    print(f"    zero_crossing_rate_mean:  {df.loc[0, 'zero_crossing_rate_mean']:.4f}")

    # -------------------------------------------------------------
    # Stage 5: Estimator Positive-Class Selection
    # -------------------------------------------------------------
    model = artifact["model"]
    classes = list(getattr(model, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else 1

    print(f"\n[Stage 5] Estimator Class Labels:")
    print(f"  Model Classes: {classes}")
    print(f"  Positive Class Index: {pos_idx} (class {classes[pos_idx]})")

    # -------------------------------------------------------------
    # Stage 6: 19 Segment-Level Probability Scores
    # -------------------------------------------------------------
    probs = model.predict_proba(df.values)[:, pos_idx]
    segment_scores = [round(float(p), 4) for p in probs]

    print(f"\n[Stage 6] Segment-Level Scores (19 segments):")
    for i, s in enumerate(segment_scores):
        print(f"  Segment {i+1:2d} (t={i*1.0:4.1f}s - {i*1.0+2.0:4.1f}s): {s:.4f}")

    # -------------------------------------------------------------
    # Stage 7: Aggregation & Threshold Application
    # -------------------------------------------------------------
    top_score = float(np.max(probs))
    threshold = float(artifact.get("threshold", 0.225))
    is_wheeze = top_score >= threshold
    prediction_label = "Wheeze-like pattern detected" if is_wheeze else "No wheeze-like pattern detected"

    print(f"\n[Stage 7] Aggregation & Prediction:")
    print(f"  Aggregation Strategy: top-1 (max segment score)")
    print(f"  Top-1 Score: {top_score:.4f} (rounded to 3 decimals: {round(top_score, 3)})")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Above Threshold: {is_wheeze}")
    print(f"  Final Prediction: {prediction_label}")

    # -------------------------------------------------------------
    # Stage 8: API Response Consistency
    # -------------------------------------------------------------
    api_response = {
        "success": True,
        "prediction": "Wheeze-like Pattern Detected" if is_wheeze else "No Wheeze-like Pattern Detected",
        "score": round(top_score, 4),
        "threshold": round(threshold, 4),
        "segments_analyzed": len(segments),
        "audio_duration": round(duration_sec, 2),
        "segment_scores": segment_scores,
        "model_name": artifact.get("model_name", "Extra Trees Classifier"),
        "source": "backend"
    }
    print(f"\n[Stage 8] API Response Contract:")
    print(f"  Score matches max(segment_scores): {max(segment_scores) == round(top_score, 4)}")
    print(f"  Chart segments count: {len(api_response['segment_scores'])}")
    print("=" * 80 + "\n")
    return api_response


if __name__ == "__main__":
    for filename in ["224_1b1_Tc_sc_Meditron.wav", "224_1b2_Al_sc_Meditron.wav"]:
        if os.path.exists(filename):
            run_stage_by_stage_diagnostic(filename)
        else:
            print(f"File not found: {filename}")
