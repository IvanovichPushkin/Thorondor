import asyncio
import sys
import threading
import time
import atexit
import os
from collections import deque # Added for latency fix

# CRITICAL: Must be set before importing aiortc on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from core.vision import generate_frames, latest_annotated
from core.config import FRAME_HEIGHT, FRAME_WIDTH, LOG_FILE, CSV_FILE
from core.record_video import VideoRecorder
from core.record_logs import LogRecorder
from core.routes import register_routes
from core.recorders import init_recorders

app = Flask(__name__, 
            template_folder='screens', 
            static_folder='screens/static', 
            static_url_path='/static')

# -----------------------------
# Persistent event loop
# -----------------------------
_loop = asyncio.new_event_loop()

def _start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=_start_loop, args=(_loop,), daemon=True).start()

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result(timeout=15)

# -----------------------------
# FPS Detection (Literal Match)
# -----------------------------
CAMERA_SOURCES = {"cam1": 0}

# Open briefly to grab hardware speed before initializing recorders
cap_init = cv2.VideoCapture(CAMERA_SOURCES["cam1"], cv2.CAP_DSHOW)
actual_fps = cap_init.get(cv2.CAP_PROP_FPS)
if actual_fps <= 0 or actual_fps > 120:
    actual_fps = 30.0  # Fallback only if driver lies
cap_init.release()

print(f"[INFO] Hardware FPS detected: {actual_fps}")

# -----------------------------
# Recorders
# -----------------------------
recorder, log_recorder = init_recorders(fps=actual_fps)

# -----------------------------
# Camera Setup (Optimized for Latency)
# -----------------------------
# Using deque maxlen=1 to ensure we always have the freshest frame
frames = {name: deque(maxlen=1) for name in CAMERA_SOURCES.keys()}

def capture_frames(cam_name, src):
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, actual_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # These prevent "glitching" and blur during fast movement
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam: {src}")
        return
        
    consecutive_failures = 0
    while True:
        # Flush the driver buffer to stop the "burst" effect
        for _ in range(2):
            cap.grab()
            
        ret, frame = cap.retrieve()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 30:
                print(f"[WARN] Webcam {cam_name} failing — reopening...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                consecutive_failures = 0
            time.sleep(0.01)
            continue
            
        consecutive_failures = 0
        # .copy() prevents thread-tearing glitches
        frames[cam_name].append(frame.copy())

for name, src in CAMERA_SOURCES.items():
    threading.Thread(target=capture_frames, args=(name, src), daemon=True).start()

# Wait up to 5s for webcam (Updated check for deque)
print("[INFO] Waiting for webcam...")
for _ in range(50):
    if len(frames.get("cam1", [])) > 0:
        print("[INFO] Webcam ready.")
        break
    time.sleep(0.1)
else:
    print("[WARN] Webcam not ready after 5s — continuing anyway.")

# -----------------------------
# Start MJPEG/AI loop
# -----------------------------
def ai_processing_loop(cam_name):
    for _ in generate_frames(cam_name, frames_override=frames, recorder=recorder):
        # Prevent CPU starvation
        time.sleep(0.001)

for cam_name in CAMERA_SOURCES:
    threading.Thread(target=ai_processing_loop, args=(cam_name,), daemon=True).start()

# -----------------------------
# WebRTC Video Track
# -----------------------------
class ArgusVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, cam_name):
        super().__init__()
        self.cam_name = cam_name

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        # Read the raw annotated numpy frame
        frame = latest_annotated.get(self.cam_name)
        if frame is None:
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for camera...", (30, FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Thread-safe copy for WebRTC encoding
        frame_to_send = frame.copy()
        frame_rgb   = cv2.cvtColor(frame_to_send, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts       = pts
        video_frame.time_base = time_base
        
        # Keep track steady
        await asyncio.sleep(0.01)
        return video_frame

# -----------------------------
# WebRTC offer handler
# -----------------------------
async def handle_offer(cam_name, sdp, type_):
    pc = RTCPeerConnection()

    @pc.on("connectionstatechange")
    async def on_state():
        if pc.connectionState in ("failed", "closed"):
            await pc.close()

    pc.addTrack(ArgusVideoTrack(cam_name))
    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return pc.localDescription

# -----------------------------
# Log Streaming
# -----------------------------
def follow(logfile):
    logfile.seek(0, 2)
    while True:
        line = logfile.readline()
        if not line:
            time.sleep(0.1)
            continue
        if log_recorder.recording:
            log_recorder.write(line.strip())
        yield f"data: {line}\n\n"

# -----------------------------
# Routes
# -----------------------------
register_routes(app, recorder, log_recorder, generate_frames, frames, 
                CAMERA_SOURCES, run_async, handle_offer, LOG_FILE, follow, 
                template_name='wcapp.html')

# -----------------------------
# Run
# -----------------------------
def cleanup_files():
    with open(LOG_FILE, "w") as f: f.truncate(0)
    with open(CSV_FILE, "w") as f: f.truncate(0)

atexit.register(cleanup_files)

if __name__ == "__main__":
    cleanup_files()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)