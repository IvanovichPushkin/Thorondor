import asyncio
import sys
import threading
import time
import atexit
import os

# CRITICAL: Must be set before importing aiortc on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from core.vision import generate_frames, latest_annotated
from core.cameras import frames
from core.config import FRAME_HEIGHT, FRAME_WIDTH, LOG_FILE, CSV_FILE, CAMERA_SOURCES
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
# FPS Detection (Literal Match for RTSP)
# -----------------------------
# Grab the FPS from the first camera in your CAMERA_SOURCES to sync the recorder
first_cam_name = list(CAMERA_SOURCES.keys())[0]
first_cam_url = CAMERA_SOURCES[first_cam_name]

cap_init = cv2.VideoCapture(first_cam_url)
actual_fps = cap_init.get(cv2.CAP_PROP_FPS)

# RTSP sometimes takes a second to report metadata, or defaults to 0
if actual_fps <= 0 or actual_fps > 120:
    actual_fps = 15.0 # Typical Tapo default if detection fails
cap_init.release()

print(f"[INFO] Detected Stream FPS: {actual_fps}. Syncing recorders...")

# -----------------------------
# Recorders
# -----------------------------
recorder, log_recorder = init_recorders(fps=actual_fps)

# -----------------------------
# NO CAMERA SETUP — uses core.cameras.frames
# -----------------------------

# -----------------------------
# Start AI loop per camera so latest_annotated gets populated
# -----------------------------
def ai_processing_loop(cam_name):
    # This keeps the AI running in the background for each Tapo cam
    gen = generate_frames(cam_name, frames_override=frames, recorder=recorder)
    while True:
        try:
            next(gen) # Just trigger the next frame processing
            # Optimization: Tiny breather to keep the RTSP stream thread stable
            time.sleep(0.001) 
        except StopIteration:
            break
        except Exception as e:
            print(f"AI Loop Error ({cam_name}): {e}")
            time.sleep(1)

for cam_name in CAMERA_SOURCES:
    threading.Thread(target=ai_processing_loop, args=(cam_name,), daemon=True).start()

# -----------------------------
# WebRTC Video Track — reads raw annotated frame, no JPEG roundtrip
# -----------------------------
class ArgusVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, cam_name):
        super().__init__()
        self.cam_name = cam_name

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = None
        for _ in range(10): 
            frame = latest_annotated.get(self.cam_name)
            if frame is not None:
                break
            await asyncio.sleep(0.02) 

        if frame is None:
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, f"Connecting to {self.cam_name}...", (50, FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # GLITCH FIX: Use a copy so the conversion doesn't happen while AI is drawing
        frame_to_send = frame.copy()
        frame_rgb = cv2.cvtColor(frame_to_send, cv2.COLOR_BGR2RGB)
        
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        
        # Keep the event loop yielding smoothly
        await asyncio.sleep(0.005) 
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
                template_name='app.html')

# -----------------------------
# Run
# -----------------------------
def cleanup_files():
    # Defensive truncate to ensure file handles are cleared
    try:
        with open(LOG_FILE, "w") as f: f.truncate(0)
        with open(CSV_FILE, "w") as f: f.truncate(0)
    except:
        pass

atexit.register(cleanup_files)

if __name__ == "__main__":
    cleanup_files()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)