import cv2
import threading
import time
from collections import deque
from core.config import CAMERA_SOURCES, FRAME_WIDTH, FRAME_HEIGHT

# === Shared frame storage — maxlen=1 keeps only the freshest frame ===
frames = {name: deque(maxlen=1) for name in CAMERA_SOURCES.keys()}

def capture_frames(cam_name, src):
    cap = cv2.VideoCapture(src)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source: {cam_name} ({src})")
        return

    # Rate-limit capture to the stream's actual FPS so we don't spin the CPU.
    # Cap at 30fps floor; RTSP usually reports 15 for Tapo stream2.
    stream_fps = cap.get(cv2.CAP_PROP_FPS)
    if stream_fps <= 0 or stream_fps > 60:
        stream_fps = 15.0
    frame_interval = 1.0 / stream_fps  # seconds between captures
    print(f"[INFO] Started capture for {cam_name} @ {stream_fps:.1f} fps (interval {frame_interval*1000:.0f}ms)")

    consecutive_failures = 0

    while True:
        t0 = time.perf_counter()

        # Grab once to flush the oldest buffered frame, then retrieve the next.
        # One grab is enough with CAP_PROP_BUFFERSIZE=1 — we don't need 2+ grabs
        # since the buffer only holds 1 frame. Extra grabs just waste CPU.
        cap.grab()
        ret, frame = cap.retrieve()

        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 20:
                print(f"[WARN] Connection lost for {cam_name}. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(src)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                consecutive_failures = 0
            time.sleep(0.05)
            continue

        consecutive_failures = 0
        frames[cam_name].append(frame)  # No .copy() — deque owns it, AI workers read-only

        # Sleep for remainder of frame interval so this thread yields CPU to AI workers
        elapsed = time.perf_counter() - t0
        sleep_t = frame_interval - elapsed
        if sleep_t > 0.002:
            time.sleep(sleep_t)

# === Background camera capture threads ===
for name, src in CAMERA_SOURCES.items():
    t = threading.Thread(target=capture_frames, args=(name, src), daemon=True)
    t.start()