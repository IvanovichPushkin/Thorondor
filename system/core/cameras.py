import cv2
import threading
import time
from collections import deque
from core.config import CAMERA_SOURCES, FRAME_WIDTH, FRAME_HEIGHT

# === Shared frame storage for low latency ===
# deque maxlen=1 drops old RTSP packets automatically — always fresh frame.
frames = {name: deque(maxlen=1) for name in CAMERA_SOURCES.keys()}

def capture_frames(cam_name, src):
    """
    Background thread to capture and flush RTSP/USB camera buffers.
    """
    # For RTSP (Tapo), OpenCV works best with ffmpeg backend (default)
    cap = cv2.VideoCapture(src)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Critical for RTSP: minimize internal OpenCV buffering
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source: {cam_name} ({src})")
        return

    print(f"[INFO] Started capture thread for: {cam_name}")

    consecutive_failures = 0

    while True:
        # FLUSH: Grab multiple frames without decoding to drain the RTSP buffer.
        # Tapo cams buffer several frames server-side — 1 grab isn't enough.
        # 4 grabs ensures we decode the absolute latest frame, not a stale one.
        for _ in range(4):
            cap.grab()

        # RETRIEVE: Decode only the freshest frame
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
            time.sleep(0.01)
            continue

        consecutive_failures = 0
        # .copy() prevents thread-tearing when WebRTC reads simultaneously
        frames[cam_name].append(frame.copy())

# === Background camera capture threads ===
for name, src in CAMERA_SOURCES.items():
    t = threading.Thread(target=capture_frames, args=(name, src), daemon=True)
    t.start()