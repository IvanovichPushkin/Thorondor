import asyncio
import sys
import threading
import time
import atexit
from contextlib import asynccontextmanager

# CRITICAL: Must be set before importing aiortc on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from core.vision import generate_frames, latest_annotated
from core.cameras import frames
from core.config import FRAME_HEIGHT, FRAME_WIDTH, LOG_FILE, CSV_FILE, CAMERA_SOURCES
from core.recorders import init_recorders
from core.routes import register_routes

# ─────────────────────────────────────────────
# Global state — populated inside lifespan only
# ─────────────────────────────────────────────
recorder     = None
log_recorder = None


def ai_processing_loop(cam_name):
    gen = generate_frames(cam_name, frames_override=frames, recorder=recorder)
    while True:
        try:
            next(gen)
            time.sleep(0.001)
        except StopIteration:
            break
        except Exception as e:
            print(f"AI Loop Error ({cam_name}): {e}")
            time.sleep(1)


# ─────────────────────────────────────────────
# Lifespan — recorder init + AI loops live here
# so they only run once in the real server process
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recorder, log_recorder

    # Detect FPS from first RTSP camera
    first_cam_url = list(CAMERA_SOURCES.values())[0]
    cap_init = cv2.VideoCapture(first_cam_url)
    actual_fps = cap_init.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0 or actual_fps > 120:
        actual_fps = 15.0  # Tapo default fallback
    cap_init.release()
    print(f"[INFO] Detected Stream FPS: {actual_fps}. Syncing recorders...")

    # Init recorders
    recorder, log_recorder = init_recorders(fps=actual_fps)

    # Start AI loop per camera
    for cam_name in CAMERA_SOURCES:
        threading.Thread(
            target=ai_processing_loop, args=(cam_name,), daemon=True
        ).start()

    # Register routes now that recorder is ready
    register_routes(app, recorder, log_recorder, generate_frames, frames,
                    CAMERA_SOURCES, handle_offer, LOG_FILE, follow,
                    templates=templates, template_name="app.html")

    yield  # ← server is running

    # Cleanup on shutdown
    try:
        with open(LOG_FILE, "w") as f: f.truncate(0)
        with open(CSV_FILE, "w") as f: f.truncate(0)
    except Exception:
        pass


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="screens/static"), name="static")
templates = Jinja2Templates(directory="screens")


# ─────────────────────────────────────────────
# WebRTC Track
# ─────────────────────────────────────────────
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

        frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


# ─────────────────────────────────────────────
# WebRTC Offer Handler
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Log Streaming
# ─────────────────────────────────────────────
def follow(logfile):
    logfile.seek(0, 2)
    while True:
        line = logfile.readline()
        if not line:
            time.sleep(0.1)
            continue
        if log_recorder and log_recorder.recording:
            log_recorder.write(line.strip())
        yield f"data: {line}\n\n"


# ─────────────────────────────────────────────
# Entry point — always use uvicorn directly,
# NEVER "fastapi dev" for camera apps
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        log_level="warning",
        reload=False        # reload=False is critical — reloader breaks camera lock
    )