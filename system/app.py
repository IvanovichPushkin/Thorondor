import os
# ── CRITICAL: Set BEFORE any numpy/cv2/onnxruntime imports ───────────────────
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
os.environ.setdefault("OMP_WAIT_POLICY",      "PASSIVE")
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import sys
import threading
import time
import atexit
from contextlib import asynccontextmanager

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.vision import generate_frames, latest_annotated
from core.cameras import frames
from core.config import FRAME_HEIGHT, FRAME_WIDTH, LOG_FILE, CSV_FILE, CAMERA_SOURCES
from core.recorders import init_recorders
from core.routes import register_routes
import core.vision as _vision

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
recorder     = None
log_recorder = None


# ─────────────────────────────────────────────
# AI processing loop (one per camera)
# ─────────────────────────────────────────────
def ai_processing_loop(cam_name):
    gen = generate_frames(cam_name, frames_override=frames, recorder=recorder)
    while True:
        try:
            next(gen)
        except StopIteration:
            break
        except Exception as e:
            print(f"AI Loop Error ({cam_name}): {e}")
            time.sleep(1)


# ─────────────────────────────────────────────
# MJPEG frame generator
# ─────────────────────────────────────────────
def mjpeg_generator(cam_name):
    """Yield annotated frames as a multipart MJPEG stream."""
    last_id = None
    blank   = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(blank, f"Connecting to {cam_name}...",
                (50, FRAME_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)
    _, blank_jpg = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 85])
    blank_bytes  = blank_jpg.tobytes()

    while True:
        frame = latest_annotated.get(cam_name)

        if frame is None:
            payload = blank_bytes
        else:
            fid = id(frame)
            if fid == last_id:
                # No new frame yet — yield nothing and sleep briefly
                time.sleep(0.005)
                continue
            last_id = fid
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            payload  = jpg.tobytes() if ok else blank_bytes

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            payload +
            b"\r\n"
        )


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recorder, log_recorder

    # Detect FPS from first RTSP camera
    first_cam_url = list(CAMERA_SOURCES.values())[0]
    cap_init = cv2.VideoCapture(first_cam_url)
    actual_fps = cap_init.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0 or actual_fps > 120:
        actual_fps = 15.0
    cap_init.release()
    print(f"[INFO] Detected Stream FPS: {actual_fps}. Syncing recorders...")

    recorder, log_recorder = init_recorders(fps=actual_fps)

    # Pre-warm AI workers for all cameras in parallel
    print("[INFO] Pre-warming AI workers for all cameras...")
    warm_events = []
    for cam_name in CAMERA_SOURCES:
        ev = threading.Event()
        warm_events.append(ev)
        def _warm(cn=cam_name, done=ev):
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            try:
                from core.detections.pose   import predict as pp
                from core.detections.object import predict as op
                from core.detections.desk   import predict as dp
                pp(dummy, cn); op(dummy, cn); dp(dummy, cn)
            except Exception as e:
                print(f"[WARN] Warmup failed for {cn}: {e}")
            finally:
                done.set()
        threading.Thread(target=_warm, daemon=True).start()

    for ev in warm_events:
        ev.wait(timeout=10)
    print("[INFO] AI workers warmed up — starting camera loops")

    # Start AI loop per camera
    for cam_name in CAMERA_SOURCES:
        threading.Thread(
            target=ai_processing_loop, args=(cam_name,), daemon=True
        ).start()

    # Register routes
    register_routes(app, recorder, log_recorder, generate_frames, frames,
                    CAMERA_SOURCES, None, LOG_FILE, follow,
                    templates=templates, template_name="app.html")

    # MJPEG video feed routes
    @app.get("/video_feed/{cam_name}")
    async def video_feed(cam_name: str):
        return StreamingResponse(
            mjpeg_generator(cam_name),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    yield

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
# Log streaming
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


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        log_level="warning",
        reload=False
    )