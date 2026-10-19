#!/usr/bin/env python3
"""
emotion_profiles.py — 16 Composite Color-Emotion Profiles
==========================================================

Author: KUANG, Haozhe
Affiliation: The Hong Kong University of Science and Technology
Email: hkuangab@connect.ust.hk
Date: October 2026

Description:
    Defines 16 composite emotion profiles with HSV anchor colors,
    instability values, and the 17-dimensional feature vector specification.
    Used as the target classes for the GRU emotion classifier.

    THIS IS NOT REAL EMOTIONAL RECOGNITION.
    These are synthetic color-emotion mappings for demonstration purposes.

Notes:
    - Each profile defines a single anchor HSV color plus temporal noise.
    - Instability controls how much the color drifts between frames.
    - FEATURE_COLS defines the 17 extracted features per video frame.
"""

EMOTION_COLOR_PROFILES = {
    0:  {"name": "calm_serenity",      "colors": [{"h": 200, "s": 120, "v": 220}], "instability": 0.05},
    1:  {"name": "warm_affection",     "colors": [{"h": 10,  "s": 170, "v": 230}], "instability": 0.07},
    2:  {"name": "melancholy_blue",    "colors": [{"h": 220, "s": 140, "v": 150}], "instability": 0.09},
    3:  {"name": "anxious_tension",    "colors": [{"h": 30,  "s": 190, "v": 180}], "instability": 0.18},
    4:  {"name": "excited_joy",        "colors": [{"h": 45,  "s": 210, "v": 240}], "instability": 0.14},
    5:  {"name": "dark_despair",       "colors": [{"h": 240, "s": 120, "v": 70}],  "instability": 0.12},
    6:  {"name": "romantic_blush",     "colors": [{"h": 350, "s": 150, "v": 225}], "instability": 0.08},
    7:  {"name": "cold_detachment",    "colors": [{"h": 190, "s": 60,  "v": 200}], "instability": 0.06},
    8:  {"name": "fiery_anger",        "colors": [{"h": 0,   "s": 235, "v": 220}], "instability": 0.22},
    9:  {"name": "mysterious_depth",   "colors": [{"h": 270, "s": 160, "v": 130}], "instability": 0.11},
    10: {"name": "fresh_renewal",      "colors": [{"h": 110, "s": 170, "v": 215}], "instability": 0.07},
    11: {"name": "nostalgic_warmth",   "colors": [{"h": 25,  "s": 150, "v": 205}], "instability": 0.08},
    12: {"name": "dreamy_softness",    "colors": [{"h": 300, "s": 110, "v": 220}], "instability": 0.09},
    13: {"name": "tense_alertness",    "colors": [{"h": 60,  "s": 200, "v": 225}], "instability": 0.16},
    14: {"name": "peaceful_glow",      "colors": [{"h": 170, "s": 120, "v": 230}], "instability": 0.05},
    15: {"name": "stormy_turmoil",     "colors": [{"h": 210, "s": 210, "v": 120}], "instability": 0.20},
}

NUM_CLASSES = len(EMOTION_COLOR_PROFILES)

# 17-dimensional feature vector per frame
# Extracted from top-3 dominant HSV colors plus derived metrics
FEATURE_COLS = [
    "c1_h", "c1_s", "c1_v", "c1_ratio",
    "c2_h", "c2_s", "c2_v", "c2_ratio",
    "c3_h", "c3_s", "c3_v", "c3_ratio",
    "color_entropy", "saturation_avg", "area_imbalance",
    "hue_shift", "motion_color_change",
]
INPUT_DIM = len(FEATURE_COLS)
WINDOW_SIZE = 12