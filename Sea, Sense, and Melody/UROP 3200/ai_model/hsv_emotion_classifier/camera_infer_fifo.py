#!/usr/bin/env python3
"""
camera_infer_fifo.py — Camera inference via rpicam-vid subprocess + FIFO + PREVIEW
Author: KUANG, Haozhe (HKUST)

Uses rpicam-vid (MJPEG stdout pipe) — no picamera2 needed.
Shows a live camera window while extracting top 3 vibrant colors.
Press 'q' to quit.
"""
import cv2
import numpy as np
import time
import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(__file__))
from fifo_writer import FeatureFifoWriter

# ─── Configuration ────────────────────────────────────────────────────────────
CAMERA_RESOLUTION = (640, 480)
FPS_TARGET = 15
FIFO_ENABLED = True
DEBUG = True
WINDOW_NAME = "Camera Preview — Press 'q' to quit"

# ─── Helper Functions ─────────────────────────────────────────────────────────
def extract_top_colors_from_hsv(hsv_frame, top_n=3):
    sat_mask = (hsv_frame[:,:,1] > 50) & (hsv_frame[:,:,2] > 50)
    if not np.any(sat_mask):
        return [(0,0,0,0.0)] * top_n

    filtered_hsv = hsv_frame[sat_mask]
    hue_bins = 18
    hist = cv2.calcHist([filtered_hsv[:,0].reshape(-1,1)], [0], None, [hue_bins], [0,180])
    total_pixels = len(filtered_hsv)
    top_indices = np.argsort(hist.flatten())[::-1][:top_n]

    colors = []
    for idx in top_indices:
        hue_center = int(idx * (180.0 / hue_bins) + (180.0 / hue_bins / 2))
        bin_mask = (filtered_hsv[:,0] >= idx*(180/hue_bins)) & \
                   (filtered_hsv[:,0] < (idx+1)*(180/hue_bins))
        if np.any(bin_mask):
            avg_s = np.mean(filtered_hsv[bin_mask, 1])
            avg_v = np.mean(filtered_hsv[bin_mask, 2])
            ratio = len(filtered_hsv[bin_mask]) / total_pixels
        else:
            avg_s = avg_v = 0
            ratio = 0.0
        colors.append((hue_center, avg_s, avg_v, ratio))

    while len(colors) < top_n:
        colors.append((0,0,0,0.0))

    return colors[:top_n]

def compute_derived_features(top_colors):
    features = []
    for h, s, v, r in top_colors:
        features.extend([float(h), float(s), float(v), float(r)])
    return features

def draw_color_bars(frame, top_colors):
    h_frame, w_frame = frame.shape[:2]
    bar_height = 40
    bar_width = w_frame // 3

    for i, (hue, sat, val, ratio) in enumerate(top_colors):
        bgr_color = cv2.cvtColor(
            np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR
        )[0][0].tolist()
        x_start = i * bar_width
        cv2.rectangle(frame,
                      (x_start, h_frame - bar_height),
                      (x_start + bar_width - 2, h_frame),
                      bgr_color, -1)
        text = f"c{i+1}: {ratio:.2f}"
        cv2.putText(frame, text,
                    (x_start + 5, h_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255) if sum(bgr_color) < 384 else (0, 0, 0),
                    1, cv2.LINE_AA)
    return frame

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  HSV EMOTION CLASSIFIER — Camera + FIFO + PREVIEW")
    print("  Backend: rpicam-vid (MJPEG pipe)")
    print("=" * 60)
    print("  Author: KUANG, Haozhe (HKUST)")
    print("  FIFO:   ", "/tmp/sb_features.fifo" if FIFO_ENABLED else "DISABLED")
    print("  Preview: ON")
    print("=" * 60)

    # ── Find rpicam-vid ──
    rpicam_cmd = None
    for candidate in ["rpicam-vid", "/usr/bin/rpicam-vid"]:
        try:
            result = subprocess.run(["which", candidate], capture_output=True, text=True)
            if result.returncode == 0:
                rpicam_cmd = candidate if "/" in candidate else result.stdout.strip()
                break
        except:
            pass

    if not rpicam_cmd:
        print("ERROR: rpicam-vid not found. Install: sudo apt install rpicam-apps")
        return

    # ── TURN ON CAMERA via rpicam-vid subprocess ──
    cmd = [
        rpicam_cmd,
        "-t", "0",
        "--codec", "mjpeg",
        "--width", str(CAMERA_RESOLUTION[0]),
        "--height", str(CAMERA_RESOLUTION[1]),
        "--framerate", str(FPS_TARGET),
        "-o", "-"
    ]

    print(f"[camera] Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           bufsize=0)

    # Give camera time to start
    time.sleep(3)

    if proc.poll() is not None:
        stderr_output = proc.stderr.read().decode(errors='replace')
        print(f"ERROR: rpicam-vid failed to start")
        print(f"Stderr: {stderr_output[:500]}")
        return

    print(f"[camera] Camera TURNED ON: {CAMERA_RESOLUTION} @ {FPS_TARGET} FPS")

    # Create preview window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    print(f"[preview] Window opened")

    fifo_writer = FeatureFifoWriter() if FIFO_ENABLED else None
    if fifo_writer and fifo_writer.fd is None:
        print("[WARN] FIFO not connected; will retry each frame")

    # ── Read MJPEG stream ──
    frame_count = 0
    prev_time = time.time()
    buffer = b""

    try:
        while True:
            # Read chunk from stdout
            chunk = proc.stdout.read(65536)
            if not chunk:
                print("ERROR: Camera stream ended")
                break
            buffer += chunk

            # Find JPEG boundaries (SOI = FF D8, EOI = FF D9)
            start = buffer.find(b'\xff\xd8')
            end = buffer.find(b'\xff\xd9')
            if start != -1 and end != -1 and end > start:
                jpeg_data = buffer[start:end+2]
                buffer = buffer[end+2:]

                # Decode JPEG
                frame_bgr = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue

                # ── PROCESS HSV ──
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                top_colors = extract_top_colors_from_hsv(hsv, top_n=3)
                features_12 = compute_derived_features(top_colors)

                # Send to FIFO
                if fifo_writer:
                    success = fifo_writer.write_features(features_12)
                    if DEBUG and frame_count % 30 == 0:
                        elapsed = time.time() - prev_time
                        actual_fps = 30.0 / elapsed if elapsed > 0 else 0
                        prev_time = time.time()
                        print(f"[fifo] Sent 12 dims | "
                              f"c1_ratio={top_colors[0][3]:.3f} | "
                              f"c2_ratio={top_colors[1][3]:.3f} | "
                              f"c3_ratio={top_colors[2][3]:.3f} | "
                              f"~{actual_fps:.1f} Hz")

                # ── DRAW PREVIEW ──
                preview_frame = frame_bgr.copy()
                preview_frame = draw_color_bars(preview_frame, top_colors)
                status = f"Frame {frame_count} | FIFO: {'OK' if fifo_writer and fifo_writer.fd else 'WAIT'}"
                cv2.putText(preview_frame, status,
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 1, cv2.LINE_AA)

                # ── SHOW WINDOW ──
                cv2.imshow(WINDOW_NAME, preview_frame)

                frame_count += 1

                # Check for 'q' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[shutdown] 'q' pressed, exiting...")
                    break

    except KeyboardInterrupt:
        print("\n[shutdown] User interrupted (Ctrl+C)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        cv2.destroyAllWindows()
        if fifo_writer:
            fifo_writer.close()
        print("[done] Camera OFF, resources released")

if __name__ == "__main__":
    main()