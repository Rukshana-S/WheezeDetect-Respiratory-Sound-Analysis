"""Repeatable diagnostic script comparing local WheezeDetect pipeline against saved Colab ground-truth.
Colab ground truth from Wheeze_Detection_Project_final/results/improved_final_test_predictions.csv:
- 224_1b1_Tc_sc_Meditron.wav: score = 0.231737, prediction = 1 (Wheeze Detected), true label = 0
- 224_1b2_Al_sc_Meditron.wav: score = 0.206209, prediction = 0 (No Wheeze),       true label = 0
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

COLAB_BENCHMARK = {
    "224_1b1_Tc_sc_Meditron.wav": {
        "expected_score": 0.231737,
        "expected_pred": 1,
        "expected_label": "Wheeze-like pattern detected",
        "true_label": 0
    },
    "224_1b2_Al_sc_Meditron.wav": {
        "expected_score": 0.206209,
        "expected_pred": 0,
        "expected_label": "No wheeze-like pattern detected",
        "true_label": 0
    }
}


def get_sha256(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_pipeline(filepath: str, res_type: str = "kaiser_fast", zcr_rms_fl: int = 512):
    # 1. SHA256
    file_hash = get_sha256(filepath)
    info = sf.info(filepath)
    
    # 2. Resampling & Mono
    target_sr = 16000
    y, sr = librosa.load(filepath, sr=target_sr, res_type=res_type, mono=True)
    
    # 3. Slicing
    win = int(2.0 * sr)
    stride = int(1.0 * sr)
    segments = [y[i : i + win] for i in range(0, len(y) - win + 1, stride)]
    
    # 4. Features
    n_fft, hop_length, n_mfcc = 512, 160, 40
    feat_rows = []
    for seg in segments:
        f = {}
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        for i in range(n_mfcc): f[f"mfcc_mean_{i}"] = float(np.mean(mfcc[i]))
        for i in range(n_mfcc): f[f"mfcc_std_{i}"] = float(np.std(mfcc[i]))
        delta = librosa.feature.delta(mfcc)
        for i in range(n_mfcc): f[f"delta_mean_{i}"] = float(np.mean(delta[i]))
        for i in range(n_mfcc): f[f"delta_std_{i}"] = float(np.std(delta[i]))
        delta2 = librosa.feature.delta(mfcc, order=2)
        for i in range(n_mfcc): f[f"delta2_mean_{i}"] = float(np.mean(delta2[i]))
        for i in range(n_mfcc): f[f"delta2_std_{i}"] = float(np.std(delta2[i]))
        cent = librosa.feature.spectral_centroid(y=seg, sr=sr, n_fft=n_fft, hop_length=hop_length)
        f["spectral_centroid_mean"] = float(np.mean(cent))
        f["spectral_centroid_std"] = float(np.std(cent))
        bw = librosa.feature.spectral_bandwidth(y=seg, sr=sr, n_fft=n_fft, hop_length=hop_length)
        f["spectral_bandwidth_mean"] = float(np.mean(bw))
        f["spectral_bandwidth_std"] = float(np.std(bw))
        ro = librosa.feature.spectral_rolloff(y=seg, sr=sr, n_fft=n_fft, hop_length=hop_length)
        f["spectral_rolloff_mean"] = float(np.mean(ro))
        f["spectral_rolloff_std"] = float(np.std(ro))
        zcr = librosa.feature.zero_crossing_rate(y=seg, frame_length=zcr_rms_fl, hop_length=hop_length)
        f["zero_crossing_rate_mean"] = float(np.mean(zcr))
        f["zero_crossing_rate_std"] = float(np.std(zcr))
        rms = librosa.feature.rms(y=seg, frame_length=zcr_rms_fl, hop_length=hop_length)
        f["rms_energy_mean"] = float(np.mean(rms))
        f["rms_energy_std"] = float(np.std(rms))
        sc = librosa.feature.spectral_contrast(y=seg, sr=sr, n_fft=n_fft, hop_length=hop_length)
        for i in range(sc.shape[0]):
            f[f"spectral_contrast_mean_{i}"] = float(np.mean(sc[i]))
            f[f"spectral_contrast_std_{i}"] = float(np.std(sc[i]))
        feat_rows.append(f)
        
    artifact = joblib.load(MODEL_PATH)
    cols = artifact["feature_columns"]
    df = pd.DataFrame(feat_rows)[cols]
    
    # 5. Positive class
    model = artifact["model"]
    classes = list(getattr(model, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else 1
    
    # 6. Segment scores
    probs = model.predict_proba(df.values)[:, pos_idx]
    segment_scores = [round(float(p), 6) for p in probs]
    
    # 7. Aggregation & threshold
    top_score = float(np.max(probs))
    threshold = float(artifact["threshold"])
    pred = 1 if top_score >= threshold else 0
    
    return {
        "sha256": file_hash,
        "orig_sr": info.samplerate,
        "channels": info.channels,
        "duration": info.duration,
        "segments": len(segments),
        "feature_count": df.shape[1],
        "classes": classes,
        "pos_idx": pos_idx,
        "segment_scores": segment_scores,
        "top_score": top_score,
        "threshold": threshold,
        "pred": pred
    }


def main():
    print("=" * 80)
    print("WHEEZEDETECT: LOCAL INFERENCE vs COLAB GROUND TRUTH BENCHMARK")
    print("=" * 80)
    
    for fname, ref in COLAB_BENCHMARK.items():
        if not os.path.exists(fname):
            print(f"File {fname} not found in workspace.")
            continue
            
        print(f"\nEvaluating: {fname}")
        print(f"  Colab Saved Score: {ref['expected_score']:.6f} | Pred: {ref['expected_pred']} | Threshold: 0.225")
        
        # Test configurations
        configs = [
            ("Current Pipeline (kaiser_fast, fl=512)", "kaiser_fast", 512),
            ("Initial Backend (soxr_hq, fl=2048)", "soxr_hq", 2048),
            ("Soxr VHQ (soxr_vhq, fl=1024)", "soxr_vhq", 1024)
        ]
        
        for name, rt, fl in configs:
            res = run_pipeline(fname, res_type=rt, zcr_rms_fl=fl)
            diff = res["top_score"] - ref["expected_score"]
            match_pred = res["pred"] == ref["expected_pred"]
            print(f"  [{name}]:")
            print(f"    SHA-256: {res['sha256'][:16]}... | Segments: {res['segments']} | Features: {res['feature_count']}")
            print(f"    Score: {res['top_score']:.6f} (Diff: {diff:+.6f}) | Pred: {res['pred']} (Matches Colab Pred: {match_pred})")
            print(f"    Top Segment Index: {np.argmax(res['segment_scores']) + 1} with score {max(res['segment_scores']):.6f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
