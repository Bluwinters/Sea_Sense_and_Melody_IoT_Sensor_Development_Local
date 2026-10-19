#!/usr/bin/env python3
"""
run_pipeline.py — One‑Click Pipeline Launcher
==============================================

Author:       KUANG, Haozhe
Affiliation:  The Hong Kong University of Science and Technology
Email:        hkuangab@connect.ust.hk
Date:         October 2026

Description:
    Parent process that orchestrates the entire sensor+vision pipeline:
      1. Creates the named pipe /tmp/sb_features.fifo if missing.
      2. Starts the C data collection binary as a subprocess.
      3. Starts the Python camera inference script as a subprocess.
      4. Monitors both; terminates all on Ctrl+C.

Usage:
    sudo python3 run_pipeline.py
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIFO_PATH = "/tmp/sb_features.fifo"
SENSOR_BOX_DIR = os.path.expanduser("~/Desktop/UROP 3200/sensor_box")
CAMERA_DIR = os.path.expanduser("~/Desktop/UROP 3200/ai_model/hsv_emotion_classifier")
C_BINARY = "data_collection_pipeline"
PY_SCRIPT = "camera_infer_fifo.py"


def main():
    print("=" * 60)
    print("  UROP 3200 — Full Pipeline Launcher")
    print("  Author: KUANG, Haozhe (HKUST)")
    print("=" * 60)

    # 1. Create FIFO if needed
    if not os.path.exists(FIFO_PATH):
        os.mkfifo(FIFO_PATH)
        print(f"[OK] Created FIFO: {FIFO_PATH}")
    else:
        print(f"[OK] FIFO already exists: {FIFO_PATH}")

    # 2. Start C binary
    c_path = os.path.join(SENSOR_BOX_DIR, C_BINARY)
    if not os.path.isfile(c_path):
        print(f"[ERROR] C binary not found: {c_path}")
        print("        Please compile first:")
        print("        gcc -std=c11 -O2 -pthread -o data_collection_pipeline \\")
        print("            data_collection_pipeline.c fifo_bridge.c label_assigner.c \\")
        print("            csv_writer.c -lgpiod -lm")
        sys.exit(1)

    print(f"[INFO] Starting C pipeline: {C_BINARY}")
    c_proc = subprocess.Popen([c_path], cwd=SENSOR_BOX_DIR)

    # 3. Start Python camera inference
    py_path = os.path.join(CAMERA_DIR, PY_SCRIPT)
    if not os.path.isfile(py_path):
        print(f"[ERROR] Python script not found: {py_path}")
        c_proc.terminate()
        sys.exit(1)

    print(f"[INFO] Starting camera inference: {PY_SCRIPT}")
    py_proc = subprocess.Popen([sys.executable, py_path], cwd=CAMERA_DIR)

    # 4. Monitor and handle Ctrl+C
    try:
        while True:
            if c_proc.poll() is not None:
                print("[WARN] C pipeline exited unexpectedly.")
                break
            if py_proc.poll() is not None:
                print("[WARN] Python inference exited unexpectedly.")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down pipeline...")

    # 5. Cleanup
    for proc in (py_proc, c_proc):
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    print("[INFO] All processes terminated. Goodbye.")


if __name__ == "__main__":
    main()