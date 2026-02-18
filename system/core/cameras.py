import cv2
import threading
import time
from collections import deque
from core.config import CAMERA_SOURCES, FRAME_WIDTH, FRAME_HEIGHT

# === Shared frame storage for low latency ===
# We use deque maxlen=1 to drop old RTSP packets automatically.
# This is the "secret" to turning RTSP into real-time WebRTC.
frames = {name: deque(maxlen=1) for name in CAMERA_SOURCES.keys()}

def capture_frames(cam_name, src):
    """
    Background thread to capture and flush RTSP/USB camera buffers.
    """
    # For RTSP (Tapo), OpenCV works best with ffmpeg or no specific backend defined
    cap = cv2.VideoCapture(src)
    
    # Set resolution based on your config
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Critical for RTSP: minimize internal buffering
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source: {cam_name} ({src})")
        return

    print(f"[INFO] Started capture thread for: {cam_name}")

    while True:
        # 1. FLUSH: Grab frames without decoding them to reach the latest one.
        # This prevents the Tapo cam from "bursting" or playing in fast-forward.
        cap.grab()
        
        # 2. RETRIEVE: Decode only the absolute freshest frame
        ret, frame = cap.retrieve()
        
        if not ret:
            print(f"[WARN] Connection lost for {cam_name}. Reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(src)
            continue

        # 3. THREAD SAFETY: Use .copy() so WebRTC doesn't grab a half-written frame
        frames[cam_name].append(frame.copy())

# === Background camera capture threads ===
for name, src in CAMERA_SOURCES.items():
    t = threading.Thread(target=capture_frames, args=(name, src), daemon=True)
    t.start()