import os
# ── CRITICAL: Set BEFORE any numpy/cv2/onnxruntime/torch imports ──────────────
# ONNX Runtime spawns OMP_NUM_THREADS per session. With 3 sessions (pose, obj,
# desk) on default settings = 3 × N_CPU_CORES threads all fighting each other
# after ~5s warmup, starving the uvicorn/aiortc event loop → WebRTC lag.
# 2 threads per session is the sweet spot: fast inference, no thread thrash.
os.environ.setdefault("OMP_NUM_THREADS",      "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS",      "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "2")
os.environ.setdefault("OMP_WAIT_POLICY",      "PASSIVE")  # idle-wait not spin-wait
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import sys
import threading
import time
import atexit
from contextlib import asynccontextmanager
from collections import deque

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
from core.config import FRAME_HEIGHT, FRAME_WIDTH, LOG_FILE, CSV_FILE
from core.recorders import init_recorders
from core.routes import register_routes

# ─────────────────────────────────────────────
# Global state — populated inside lifespan only
# ─────────────────────────────────────────────
CAMERA_SOURCES = {}
frames         = {}
recorder       = None
log_recorder   = None
_cam_backend   = 0

# ─────────────────────────────────────────────
# Camera probe
# ─────────────────────────────────────────────
def _find_working_camera():
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, 0] if sys.platform == "win32" else [0]
    for idx in range(6):
        for backend in backends:
            try:
                cap = (cv2.VideoCapture(idx, backend)
                       if backend != 0 else cv2.VideoCapture(idx))
                if not cap.isOpened():
                    cap.release()
                    continue
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"[INFO] Camera found: index={idx}, backend={backend}")
                    return idx, backend
            except Exception:
                pass
    print("[WARN] No working camera found — defaulting index=0, backend=default.")
    return 0, 0

def _open_cap(src, backend):
    if backend != 0:
        cap = cv2.VideoCapture(src, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(src)

# ─────────────────────────────────────────────
# Capture thread
# ─────────────────────────────────────────────
def capture_frames(cam_name, src, backend, fps):
    if sys.platform == "win32":
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = _open_cap(src, backend)
    else:
        cap = _open_cap(src, backend)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {actual_w}x{actual_h} (requested {FRAME_WIDTH}x{FRAME_HEIGHT})")

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam: {src}")
        return

    consecutive_failures = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 30:
                print(f"[WARN] Webcam {cam_name} failing — re-probing...")
                cap.release()
                time.sleep(1)
                new_idx, new_backend = _find_working_camera()
                cap = _open_cap(new_idx, new_backend)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                consecutive_failures = 0
            time.sleep(0.005)
            continue
        consecutive_failures = 0
        frames[cam_name].append(frame.copy())

def ai_processing_loop(cam_name):
    for _ in generate_frames(cam_name, frames_override=frames, recorder=recorder):
        time.sleep(0.001)

# ─────────────────────────────────────────────
# WebRTC frame-ready signal
# ─────────────────────────────────────────────
# Per-camera asyncio.Event — set by the AI pipeline the instant a new annotated
# frame lands. recv() waits on this instead of sending duplicate frames, which
# is the main cause of WebRTC jitter-buffer lag.
_frame_events: dict[str, asyncio.Event] = {}

def _get_frame_event(cam_name: str) -> asyncio.Event:
    if cam_name not in _frame_events:
        _frame_events[cam_name] = asyncio.Event()
    return _frame_events[cam_name]

def signal_new_frame(cam_name: str):
    """Called from AI pipeline thread when latest_annotated is updated."""
    ev = _frame_events.get(cam_name)
    if ev:
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(ev.set)
        except Exception:
            pass

# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global CAMERA_SOURCES, frames, recorder, log_recorder, _cam_backend

    # Clear stale log/csv from previous run at startup
    try:
        open(LOG_FILE, "w").close()
        open(CSV_FILE, "w").close()
    except Exception:
        pass

    # Probe camera
    cam_idx, _cam_backend = _find_working_camera()
    CAMERA_SOURCES = {"cam1": cam_idx}
    frames = {"cam1": deque(maxlen=1)}

    # Detect FPS
    cap_init = _open_cap(cam_idx, _cam_backend)
    actual_fps = cap_init.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0 or actual_fps > 120:
        actual_fps = 30.0
    cap_init.release()
    print(f"[INFO] Hardware FPS detected: {actual_fps}")

    # Init recorders
    recorder, log_recorder = init_recorders(fps=actual_fps)

    # Start capture thread
    threading.Thread(
        target=capture_frames,
        args=("cam1", cam_idx, _cam_backend, actual_fps),
        daemon=True
    ).start()

    # Wait up to 5s for first frame
    print("[INFO] Waiting for webcam...")
    for _ in range(50):
        if len(frames.get("cam1", [])) > 0:
            print("[INFO] Webcam ready.")
            break
        await asyncio.sleep(0.1)
    else:
        print("[WARN] Webcam not ready after 5s — continuing anyway.")

    # Wire pipeline → WebRTC signal
    import core.vision as _vision
    _vision.on_frame_ready = signal_new_frame

    # Start AI loop
    threading.Thread(
        target=ai_processing_loop, args=("cam1",), daemon=True
    ).start()

    # Register routes
    register_routes(app, recorder, log_recorder, generate_frames, frames,
                    CAMERA_SOURCES, handle_offer, LOG_FILE, follow,
                    templates=templates, template_name="wcapp.html")

    yield  # ← server is running


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

        # Ensure event exists (must be created inside the event loop)
        _get_frame_event(self.cam_name)
        ev = _frame_events[self.cam_name]

        # Wait for a genuinely new frame (up to 66ms = 15fps floor).
        # Prevents flooding the browser with duplicate frames — the primary
        # cause of WebRTC jitter-buffer lag.
        try:
            await asyncio.wait_for(ev.wait(), timeout=0.066)
        except asyncio.TimeoutError:
            pass
        ev.clear()

        raw = latest_annotated.get(self.cam_name)
        if raw is None:
            raw = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(raw, "Waiting for camera...", (30, FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Run BGR→RGB in thread pool — keeps the event loop free
        frame_bgr = raw.copy()
        loop = asyncio.get_event_loop()
        frame_rgb = await loop.run_in_executor(
            None, lambda: cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )

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

    # Munge answer SDP to set high bitrate (4Mbps) for crisp 720p
    sdp_lines = answer.sdp.split("\r\n")
    new_lines  = []
    for line in sdp_lines:
        new_lines.append(line)
        if line.startswith("m=video"):
            new_lines.append("b=AS:4000")
    answer = RTCSessionDescription(sdp="\r\n".join(new_lines), type=answer.type)

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
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "wcapp:app",
        host="0.0.0.0",
        port=5000,
        log_level="warning",
        reload=False
    )