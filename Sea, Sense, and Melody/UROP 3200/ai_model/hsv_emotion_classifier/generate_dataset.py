#!/usr/bin/env python3
"""
generate_dataset.py — Synthetic Dataset Generator for HSV Emotion Classifier
=============================================================================

Author: KUANG, Haozhe
Affiliation: The Hong Kong University of Science and Technology
Email: hkuangab@connect.ust.hk
Date: October 2026

Description:
    Generates synthetic training sequences by sampling from parametric color
    profiles defined in emotion_profiles.py. Each sequence simulates 12
    consecutive frames of dominant color observations with realistic noise,
    transitions, and temporal coherence.

Output:
    toy_dataset/X_seq.npy  — Feature sequences (N, 12, 17)
    toy_dataset/y_labels.npy — Integer labels (N,)
"""

import os
import sys
import math
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotion_profiles import EMOTION_COLOR_PROFILES, FEATURE_COLS, NUM_CLASSES, INPUT_DIM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    'num_sequences': 62500,
    'sequence_length': 27,
    'noise_level': 0.07,
    'valid_window': 15,
    'extract_length': 12,
    'random_state': 42,
}


def print_dataset_info():
    """Print basic information about the dataset configuration."""
    print("\n" + "=" * 65)
    print("  DATASET GENERATION INFORMATION")
    print("=" * 65)
    print(f"  Number of emotion classes:     {NUM_CLASSES}")
    print(f"  Feature dimensions:            {INPUT_DIM}")
    print(f"  Sequence length (raw):         {DATASET_CONFIG['sequence_length']}")
    print(f"  Valid window start:            {DATASET_CONFIG['valid_window']}")
    print(f"  Extract length:                {DATASET_CONFIG['extract_length']}")
    print(f"  Noise level:                   {DATASET_CONFIG['noise_level']}")
    print(f"  Random seed:                   {DATASET_CONFIG['random_state']}")
    print("=" * 65)

    print("\n  Emotion Classes:")
    for pid, info in EMOTION_COLOR_PROFILES.items():
        colors_str = ", ".join([f"H={c['h']:.0f} S={c['s']:.0f} V={c['v']:.0f}"
                               for c in info['colors']])
        print(f"    [{pid:2d}] {info['name']:<22s}  {colors_str}")
    print("=" * 65 + "\n")


def sample_profile(profile_id):
    """
    Sample a feature vector from a given emotion profile with noise.

    Args:
        profile_id: Integer index of the emotion profile

    Returns:
        17-element numpy array of feature values
    """
    profile = EMOTION_COLOR_PROFILES[profile_id]
    features = np.zeros(INPUT_DIM, dtype=np.float32)

    idx = 0
    for color_def in profile['colors']:
        # Add Gaussian noise to each channel
        h_noise = np.random.normal(0, 15)  # Hue noise: ±15 degrees
        s_noise = np.random.normal(0, 20)  # Saturation noise: ±20
        v_noise = np.random.normal(0, 20)  # Value noise: ±20

        h = np.clip(color_def['h'] + h_noise, 0, 179)
        s = np.clip(color_def['s'] + s_noise, 0, 255)
        v = np.clip(color_def['v'] + v_noise, 0, 255)

        # Ratio: Dirichlet-like distribution using Beta
        # Default to equal proportions if not specified
        r = np.random.beta(3, 5)  # alpha=3, beta=5 gives mean ~0.375

        features[idx] = h
        features[idx + 1] = s
        features[idx + 2] = v
        features[idx + 3] = np.clip(r, 0.05, 0.95)
        idx += 4

    return features


def generate_base_sequence(profile_id, length):
    """
    Generate a single base sequence for one emotion profile.

    Args:
        profile_id: Integer index of the emotion profile
        length: Number of frames in the sequence

    Returns:
        Array of shape (length, INPUT_DIM)
    """
    seq = np.zeros((length, INPUT_DIM), dtype=np.float32)
    for t in range(length):
        vec = sample_profile(profile_id)

        # Add temporal drift (small changes between consecutive frames)
        if t > 0:
            drift = np.random.normal(0, 3, INPUT_DIM)
            vec = vec + drift
            # Clamp values to valid ranges
            for c in range(3):  # For each color
                base_idx = c * 4
                vec[base_idx] = np.clip(vec[base_idx], 0, 179)      # H
                vec[base_idx + 1] = np.clip(vec[base_idx + 1], 0, 255)  # S
                vec[base_idx + 2] = np.clip(vec[base_idx + 2], 0, 255)  # V
                vec[base_idx + 3] = np.clip(vec[base_idx + 3], 0.05, 0.95)  # ratio

        seq[t] = vec
    return seq


def add_transition_noise(seq):
    """
    Add temporal noise simulating camera jitter and color fluctuation.

    Args:
        seq: Clean sequence array of shape (T, D)

    Returns:
        Noisy sequence array of same shape
    """
    noise = np.random.normal(0, DATASET_CONFIG['noise_level'] * 50, seq.shape)
    noisy = seq + noise

    # Clamp to valid ranges
    for c in range(3):
        base_idx = c * 4
        noisy[:, base_idx] = np.clip(noisy[:, base_idx], 0, 179)
        noisy[:, base_idx + 1] = np.clip(noisy[:, base_idx + 1], 0, 255)
        noisy[:, base_idx + 2] = np.clip(noisy[:, base_idx + 2], 0, 255)
        noisy[:, base_idx + 3] = np.clip(noisy[:, base_idx + 3], 0.05, 0.95)

    return noisy


def extract_valid_subsequence(seq, target_len):
    """
    Extract a valid subsequence from the stable region of a longer sequence.

    Args:
        seq: Full sequence array of shape (T, D)
        target_len: Desired output length

    Returns:
        Subsequence array of shape (target_len, D)
    """
    start = DATASET_CONFIG['valid_window']
    end = start + target_len
    if end <= seq.shape[0]:
        return seq[start:end]
    return seq[-target_len:]


def generate_dataset():
    """Main dataset generation routine."""
    print("[INFO] Generating synthetic dataset...")
    start_time = time.time()

    np.random.seed(DATASET_CONFIG['random_state'])

    num_base = DATASET_CONFIG['num_sequences']
    extract_len = DATASET_CONFIG['extract_length']

    print(f"[INFO] Total sequences to generate: {num_base}")
    print(f"[INFO] Extracted sequence length: {extract_len}")

    all_X = []
    all_y = []

    for i in range(num_base):
        profile_id = i % NUM_CLASSES
        seq = generate_base_sequence(profile_id, DATASET_CONFIG['sequence_length'])
        seq = add_transition_noise(seq)
        core = extract_valid_subsequence(seq, extract_len)

        all_X.append(core)
        all_y.append(profile_id)

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / num_base * 100
            rate = (i + 1) / elapsed
            eta = (num_base - i - 1) / rate if rate > 0 else 0
            print(f"[INFO] Generated {i + 1}/{num_base} sequences "
                  f"({progress:.1f}%) "
                  f"[{rate:.0f} seq/s, ETA: {eta:.0f}s]")

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    elapsed = time.time() - start_time
    print(f"\n[INFO] Generation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[INFO] Average rate: {num_base/elapsed:.0f} sequences/second")
    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] y shape: {y.shape}")
    print(f"[INFO] Feature range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"[INFO] NaN count: {np.isnan(X).sum()}")

    # Class distribution
    label_counts = np.bincount(y, minlength=NUM_CLASSES)
    print(f"[INFO] Samples per class: {label_counts}")
    print(f"[INFO] Min class count: {label_counts.min()}, Max class count: {label_counts.max()}")
    print(f"[INFO] Class balance ratio: {label_counts.min() / label_counts.max():.3f}")

    return X, y


def save_dataset(X, y, output_dir="toy_dataset"):
    """Save dataset to disk."""
    os.makedirs(output_dir, exist_ok=True)

    x_path = os.path.join(output_dir, "X_seq.npy")
    y_path = os.path.join(output_dir, "y_labels.npy")

    save_start = time.time()
    np.save(x_path, X)
    np.save(y_path, y)
    save_time = time.time() - save_start

    x_size_mb = os.path.getsize(x_path) / (1024 * 1024)
    y_size_kb = os.path.getsize(y_path) / 1024

    print(f"\n[INFO] Saving took {save_time:.2f}s")
    print(f"[INFO] Saved X to: {x_path} ({x_size_mb:.1f} MB)")
    print(f"[INFO] Saved y to: {y_path} ({y_size_kb:.1f} KB)")
    print(f"[INFO] Total sequences: {X.shape[0]:,}")

    return x_path, y_path


def main():
    """Entry point."""
    print("=" * 75)
    print("  HSV EMOTION CLASSIFIER — SYNTHETIC DATASET GENERATOR")
    print("  Author: KUANG, Haozhe | HKUST")
    print("=" * 75)

    print_dataset_info()

    X, y = generate_dataset()
    save_dataset(X, y)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toy_dataset')
    print(f"\n[INFO] Dataset generation completed successfully.")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Ready for training: run 'python3 train_model.py'")


if __name__ == "__main__":
    main()