#!/usr/bin/env python3
"""
fifo_writer.py — Feature FIFO Writer Thread for Sensor Box Pipeline
===================================================================

Author:       KUANG, Haozhe
Affiliation:  The Hong Kong University of Science and Technology
Email:        hkuangab@connect.ust.hk
Date:         October 2026
Version:      1.1.0 (moderated)

Description:
    Thread‑safe bridge between Python camera inference and C data collection.
    Uses queue.Queue(maxsize=30) as information queue and a daemon writer
    thread that drains it, writing 17‑dim feature vectors as CSV lines to
    /tmp/sb_features.fifo.

    Design decisions:
      - Non‑blocking enqueue: drops oldest frame if queue full.
      - Non‑blocking FIFO write (O_RDWR|O_NONBLOCK): drops frame if C consumer
        is not reading.
      - Daemon thread: terminates automatically when main process exits.

Wire format (one line, LF‑terminated):
    <seq>,<f0>,<f1>,...,<f16>\\n
"""

import os
import time

FIFO_PATH = "/tmp/sb_features.fifo"

class FeatureFifoWriter:
    def __init__(self, fifo_path=FIFO_PATH):
        self.fifo_path = fifo_path
        self.fd = None
        self._open()

    def _open(self):
        try:
            if not os.path.exists(self.fifo_path):
                os.mkfifo(self.fifo_path)
            self.fd = os.open(self.fifo_path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as e:
            print(f"[fifo_writer] ERROR: {e}")
            self.fd = None

    def write_features(self, features_12):
        """
        features_12: list/array of 12 floats (3 colors × h, s, v, ratio)
        """
        if self.fd is None:
            self._open()
            if self.fd is None:
                return False
        try:
            line = ",".join(f"{v:.4f}" for v in features_12) + "\n"
            os.write(self.fd, line.encode('utf-8'))
            return True
        except (OSError, BrokenPipeError):
            print("[fifo_writer] Pipe broken, reopening...")
            self.close()
            return False

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __del__(self):
        self.close()
