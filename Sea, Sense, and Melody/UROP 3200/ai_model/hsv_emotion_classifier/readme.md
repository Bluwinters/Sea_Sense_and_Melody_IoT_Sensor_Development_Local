HSV Emotion Classifier

Real-time HSV color sequence → composite emotion classifier for Raspberry Pi 5.

This classifies color patterns, NOT facial expressions.  

It maps temporal sequences of dominant HSV colors to 16 composite mood profiles based on color psychology principles.

Author: KUANG, Haozhe  
Affiliation: The Hong Kong University of Science and Technology (HKUST)  
Email: hkuangab@connect.ust.hk  
Date: October 2026

Table of Contents

1. #overview
2. #project-structure
3. #how-it-works
4. #the-16-composite-emotions
5. #system-requirements
6. #installation
7. #usage
8. #camera-controls
9. #feature-vector-17-dimensions
10. #model-architecture
11. #performance-notes
12. #troubleshooting
13. #license--disclaimer

Overview

The HSV Emotion Classifier is a computer vision project developed for the UROP 3200 program at HKUST. It uses a Raspberry Pi 5 with a PiCamera2 module to capture live video, extracts dominant HSV colors from each frame, and feeds a temporal sequence of color features into a Gated Recurrent Unit (GRU) neural network. The network predicts one of 16 predefined composite emotional states.

Unlike conventional emotion recognition systems that analyze facial expressions, this system interprets ambient or scene color palettes as emotional indicators. Applications include:

• Mood-responsive ambient lighting

• Art installation feedback systems

• Color psychology research tools

• Interactive media installations

Project Structure


hsv_emotion_classifier/
│
├── emotion_profiles.py          # 16 composite emotion profiles with HSV anchor colors
├── mood_model.py                # GRU neural network architecture definition
├── generate_dataset.py          # Generates 60,000 synthetic training sequences
├── train_model.py               # Trains GRU model and saves weights + scaler
├── camera_infer.py              # Real-time PiCamera2 inference script
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── trained_model/               # Contains trained model artifacts
│   ├── gru_mood_model.pth       # PyTorch model weights
│   └── scaler.pkl               # StandardScaler fitted on training data
│
├── toy_dataset/                 # Pre-generated sample dataset
│   ├── X_seq.npy                # Feature sequences (shape: N×12×17)
│   └── y_labels.npy             # Ground truth labels (shape: N,)
│
└── captured_frames/             # Saved camera frames (created on first use)
    └── (frames saved by pressing 's' during inference)


How It Works

The processing pipeline consists of the following stages:

1. Frame Capture

PiCamera2 captures 640×480 RGB frames at approximately 15 FPS.

2. Preprocessing

Each frame is resized to 160×120 pixels and converted from RGB to HSV color space. A Gaussian blur (5×5 kernel, σ=1.8) reduces sensor noise.

3. Dominant Color Extraction

Low-saturation pixels (saturation < 25) are masked out to exclude grays. The remaining pixels are quantized into 4 Hue bins × 8 Saturation bins × 8 Value bins = 256 possible color bins. The three most frequently occurring bins are selected as the dominant colors for that frame.

4. Feature Computation

For each frame, a 17-dimensional feature vector is computed from the three dominant colors:

Feature Group Dimensions Description

Raw HSV + ratio 12 (3 colors × 4 values) H, S, V, and proportion for each dominant color

Color entropy 1 Shannon entropy of color proportions (0–1)

Saturation average 1 Weighted mean saturation normalized to 0–1

Area imbalance 1 Ratio of largest color area to sum of smaller two

Hue shift 1 Mean absolute hue difference from previous frame

Motion color change 1 Normalized hue shift (0–1)

5. Temporal Buffering

Features are accumulated in a sliding window of 12 consecutive frames (~0.8 seconds of video).

6. Inference

When the buffer reaches 12 frames, the sequence is normalized using a pre-fitted StandardScaler and passed through the GRU classifier. The output is a probability distribution over 16 emotion classes.

7. Prediction Smoothing

The raw prediction is smoothed over the last 5 inference cycles (majority vote). A prediction is only displayed when confidence exceeds 35%.

The 16 Composite Emotions

Each emotion profile defines 3 anchor HSV colors plus variance parameters used for synthetic data generation.

ID Emotion Name Primary Anchor Hue Psychological Basis

0 Joyful Energy 45° (Yellow) Warm, high-saturation stimulation

1 Calm Serenity 210° (Blue) Cool, low-saturation relaxation

2 Melancholy Blue 220° (Deep Blue) Dark, desaturated introspection

3 Angry Red 0° (Red) High-saturation arousal

4 Peaceful Green 130° (Green) Balanced, mid-value harmony

5 Anxious Yellow 55° (Yellow-Green) High-value, high-hue contrast tension

6 Romantic Pink 330° (Pink/Magenta) Soft, warm affection

7 Neutral Gray Undefined Low-saturation equilibrium

8 Energetic Orange 20° (Orange) High-saturation activation

9 Sad Purple 280° (Purple) Dark, cool withdrawal

10 Fearful Gray 170° (Teal-Gray) Desaturated, cold avoidance

11 Surprised Gold 40° (Gold) Bright, sudden contrast

12 Disgusted Green 100° (Yellow-Green) Unbalanced, sickly tone

13 Trustworthy Blue 200° (Sky Blue) Stable, medium-value reliability

14 Anticipation Teal 175° (Teal) Moderate saturation curiosity

15 Contempt Violet 270° (Violet) Cool, dark superiority

System Requirements

Hardware

• Raspberry Pi 5 (recommended 4GB+ RAM)

• PiCamera2 module (CSI interface)

• Monitor or HDMI display (optional but recommended)

• Power supply (5V 5A USB-C recommended)

Software

• Raspberry Pi OS Bookworm (64-bit) or later

• Python 3.11+

• System packages (installed via apt):

  • python3-picamera2

  • python3-opencv

  • python3-numpy

  • python3-sklearn

  • python3-pandas

Installation

Step 1: Install System Dependencies

sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-numpy python3-sklearn python3-pandas


Step 2: Set Up Python Virtual Environment (Recommended)

cd ~/Desktop/UROP\ 3200/ai_model/hsv_emotion_classifier
python3 -m venv venv --system-site-packages
source venv/bin/activate


Step 3: Install PyTorch

PyTorch must be installed separately because it is not available in the Raspberry Pi OS repositories:
pip install torch --index-url https://download.pytorch.org/whl/cpu


Step 4: Create Required Directories

mkdir -p trained_model toy_dataset captured_frames


Usage

Step 1: Generate Training Data

Generate 60,000 synthetic sequences (each sequence = 12 frames × 17 features):
source venv/bin/activate
python3 generate_dataset.py


This creates:
• toy_dataset/X_seq.npy — feature sequences (60,000 × 12 × 17)

• toy_dataset/y_labels.npy — corresponding emotion labels (60,000,)

Expected runtime: approximately 30–60 seconds on Raspberry Pi 5.

Step 2: Train the Model

Train the GRU classifier on the generated dataset:
python3 train_model.py


Training progress is displayed in the terminal. On completion, the script saves:
• trained_model/gru_mood_model.pth — trained PyTorch model weights

• trained_model/scaler.pkl — fitted StandardScaler for normalization

Expected runtime: approximately 2–5 minutes on Raspberry Pi 5 (CPU-only).

Step 3: Run Real-Time Inference

Launch the live camera classifier:
python3 camera_infer.py


The application opens a window showing the camera feed with overlays:
• Top bar: Predicted emotion name and confidence percentage

• Bottom swatches: Three dominant colors with their proportions

• Debug panel (toggle with 'd'): Probability bars for all 16 emotions

Camera Controls

While the camera window is active, the following keyboard shortcuts are available:

Key Action

q Quit the application

r Reset the temporal buffer (clears accumulated frames)

d Toggle debug probability overlay on/off

s Save current frame to captured_frames/ directory

Saved frames are named with timestamps, predicted emotion, and confidence level, e.g.:
frame_20261012_143021_123_Calm_Serenity_conf0p82.jpg

Feature Vector (17 Dimensions)

Each frame produces a 17-element feature vector. The full specification:

Index Feature Range Description

0 C1_H 0–179 Hue of dominant color 1

1 C1_S 0–255 Saturation of dominant color 1

2 C1_V 0–255 Value (brightness) of dominant color 1

3 C1_ratio 0–1 Proportion of dominant color 1

4 C2_H 0–179 Hue of dominant color 2

5 C2_S 0–255 Saturation of dominant color 2

6 C2_V 0–255 Value of dominant color 2

7 C2_ratio 0–1 Proportion of dominant color 2

8 C3_H 0–179 Hue of dominant color 3

9 C3_S 0–255 Saturation of dominant color 3

10 C3_V 0–255 Value of dominant color 3

11 C3_ratio 0–1 Proportion of dominant color 3

12 color_entropy 0–1 Shannon entropy of color proportions

13 saturation_avg 0–1 Weighted average saturation (normalized)

14 area_imbalance 0+ Largest color area / sum of smaller two

15 hue_shift 0–180 Mean absolute hue change from previous frame

16 motion_color_change 0–1 Normalized hue shift magnitude

Model Architecture

The GRU classifier (GRUMoodClassifier in mood_model.py) has the following structure:

Input: (batch, 12, 17)

GRU Layer (hidden_dim=128, num_layers=2, bidirectional=False, batch_first=True)
Dropout between layers: 0.3

Take last hidden state (output[:,-1,:])

Dropout(0.4)

Linear(128 → 64)

ReLU

Dropout(0.3)

Linear(64 → 16)

Output: (batch, 16) — logits for 16 emotion classes

Total parameters: approximately 135,000 (lightweight enough for real-time CPU inference).

Training hyperparameters:
• Loss function: CrossEntropyLoss

• Optimizer: Adam (lr=0.001)

• Batch size: 512

• Epochs: 150

• Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=10)

• Early stopping: patience=25 epochs

Performance Notes

Expected Frame Rate

• With debug panel off: ~8–12 FPS on Raspberry Pi 5

• With debug panel on: ~6–9 FPS on Raspberry Pi 5

• Limiting factor: HSV histogram binning and numpy operations on CPU

Memory Usage

• Peak memory: approximately 350–500 MB

• Model size: approximately 540 KB (gru_mood_model.pth)

Accuracy Considerations

• The model is trained exclusively on synthetic data generated from parametric color profiles

• Real-world accuracy depends on lighting conditions, camera white balance, and scene composition

• The system performs best under consistent, neutral-white illumination

• Strong color casts (e.g., sunset, fluorescent green lighting) may bias predictions toward related emotion classes

Tips for Better Results

1. Ensure adequate, even lighting (avoid strong shadows)
2. Allow 2–3 seconds after launching for auto white balance to stabilize
3. Hold static color samples for at least 1 second for reliable classification
4. Use the 'r' key to reset the buffer when changing scenes abruptly

Troubleshooting

Problem: Camera fails to initialize

Error message: [ERROR] Failed to initialize camera: ...

Solution: Ensure the camera ribbon cable is properly seated. Run libcamera-hello to verify camera functionality. If using a third-party case, check that the camera port is not obstructed.

Problem: OpenCV window not displaying

Error message: qt.qpa.plugin: Could not load the Qt platform plugin

Solution: Install missing GUI libraries:
sudo apt install -y libqt5gui5 qt5-qmake

Problem: Model file not found

Error message: [ERROR] Model not found: trained_model/gru_mood_model.pth

Solution: Run python3 train_model.py before running inference. Ensure the trained_model/ directory exists.

Problem: Low FPS (< 5)

Possible causes and solutions:
• Reduce camera resolution in camera_infer.py: change CONFIG['camera_resolution'] to (320, 240)

• Increase frame_interval to 0.05 (sacrifices responsiveness for stability)

• Close other applications consuming CPU

Problem: All predictions show "Neutral Gray" or low confidence

Possible causes:
• Poor lighting or overexposed image

• Camera white balance not stabilized (wait 3–5 seconds)

• Scene contains mostly gray/white/black surfaces with low saturation

• Press 'r' to reset the buffer and force a fresh accumulation

Problem: Permission denied for camera

Error message: mmal: mmal_vc_component_enable: failed to enable component: ENOSPC

Solution: Reboot the Raspberry Pi. If persistent, check that no other process is using the camera (sudo fuser /dev/video0).

License & Disclaimer

License

This project is developed as part of UROP 3200 at the Hong Kong University of Science and Technology. Educational and research use is permitted with proper attribution.

Disclaimer

• This system does NOT perform facial expression recognition or any form of biometric identification.

• Emotional classifications are based on generalized color psychology associations and synthetic training data.

• The system should not be used for clinical diagnosis, psychological assessment, or any application where misclassification could cause harm.

• Color perception varies across individuals (including color vision deficiencies), display devices, and lighting conditions.

• The author and HKUST assume no liability for misuse or misinterpretation of the system's outputs.

Last updated: October 2026