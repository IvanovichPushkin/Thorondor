import cv2
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from core.detections.pose import predict as pose_predict, draw as pose_draw
from core.detections.desk import process as desk_process
from core.detections.object import predict as obj_predict, draw as obj_draw

latest_annotated = {}
executor = ThreadPoolExecutor(max_workers=4)

# ── Pose background thread ────────────────────────────────────────────────────
# Pose runs ~75-120ms. Instead of blocking the main pipeline waiting for it,
# a dedicated thread loops pose continuously and caches the latest result.
# Main pipeline grabs the cache instantly (0ms wait) and moves on.
_pose_cache        = {}   # cam_name -> matched
_pose_input        = {}   # cam_name -> latest frame for pose to process
_pose_input_lock   = threading.Lock()
_pose_cache_lock   = threading.Lock()

def _pose_worker(cam_name):
    last_frame_id = None
    while True:
        with _pose_input_lock:
            frame = _pose_input.get(cam_name)
        if frame is None or id(frame) == last_frame_id:
            time.sleep(0.005)
            continue
        last_frame_id = id(frame)
        matched = pose_predict(frame, cam_name)
        with _pose_cache_lock:
            _pose_cache[cam_name] = matched
        # Rate-limit pose to ~10fps — frees CPU for obj/desk in between
        time.sleep(0.08)

def _ensure_pose_thread(cam_name):
    key = f"_pose_thread_{cam_name}"
    if not globals().get(key):
        t = threading.Thread(target=_pose_worker, args=(cam_name,), daemon=True)
        t.start()
        globals()[key] = t

# ── Object background thread ──────────────────────────────────────────────────
# Obj (YOLO) runs 77-400ms — the main pipeline bottleneck. Same pattern as pose:
# a dedicated thread runs inference continuously and caches matched instances.
# Main pipeline calls obj_draw() on the cached result — pure OpenCV, ~0ms.
_obj_cache        = {}   # cam_name -> matched instances dict
_obj_input        = {}   # cam_name -> latest frame for obj to process
_obj_input_lock   = threading.Lock()
_obj_cache_lock   = threading.Lock()

def _obj_worker(cam_name):
    last_frame_id = None
    while True:
        with _obj_input_lock:
            frame = _obj_input.get(cam_name)
        if frame is None or id(frame) == last_frame_id:
            time.sleep(0.005)
            continue
        last_frame_id = id(frame)
        matched = obj_predict(frame, cam_name)
        with _obj_cache_lock:
            _obj_cache[cam_name] = matched

def _ensure_obj_thread(cam_name):
    key = f"_obj_thread_{cam_name}"
    if not globals().get(key):
        t = threading.Thread(target=_obj_worker, args=(cam_name,), daemon=True)
        t.start()
        globals()[key] = t

# ── FPS / timing tracker ─────────────────────────────────────────────────────
_frame_times = {}
_fps_log_interval = 5.0
_timing_accum = {}


def _log_timings(cam_name, timings):
    if cam_name not in _timing_accum:
        _timing_accum[cam_name] = []
    _timing_accum[cam_name].append(timings)

    now = time.time()
    if cam_name not in _frame_times:
        _frame_times[cam_name] = now
        return

    elapsed = now - _frame_times[cam_name]
    if elapsed < _fps_log_interval:
        return

    samples = _timing_accum[cam_name]
    n = len(samples)
    fps = n / elapsed

    def avg(key):
        return sum(s[key] for s in samples) / n * 1000

    print(
        f"\n[PERF] [{cam_name}] last {elapsed:.1f}s | "
        f"FPS: {fps:.1f} | "
        f"pose: {avg('pose'):.0f}ms | "
        f"obj: {avg('obj'):.0f}ms | "
        f"desk: {avg('desk'):.0f}ms | "
        f"overlay: {avg('overlay'):.0f}ms | "
        f"total: {avg('total'):.0f}ms"
    )

    _frame_times[cam_name] = now
    _timing_accum[cam_name] = []


# ─────────────────────────────────────────────────────────────────────────────

def run_ai_pipeline(cam_name, frame):
    try:
        t0 = time.perf_counter()
        original = frame.copy()

        # Feed latest frame to background threads (non-blocking)
        _ensure_pose_thread(cam_name)
        _ensure_obj_thread(cam_name)
        with _pose_input_lock:
            _pose_input[cam_name] = original.copy()
        with _obj_input_lock:
            _obj_input[cam_name] = original.copy()

        # Run desk in background (fast, kept as-is)
        future_desk = executor.submit(desk_process, original.copy(), cam_name, [])

        # Grab cached pose detections — draw fresh on current frame, no ghosting
        t_pose_start = time.perf_counter()
        with _pose_cache_lock:
            pose_matched = _pose_cache.get(cam_name)
        canvas = original.copy()
        if pose_matched:
            canvas, _ = pose_draw(canvas, pose_matched)
        t_pose = time.perf_counter() - t_pose_start

        # Grab cached obj detections — draw fresh on current frame, ~0ms
        t_obj_start = time.perf_counter()
        with _obj_cache_lock:
            obj_matched = _obj_cache.get(cam_name)
        obj_canvas = original.copy()
        if obj_matched:
            obj_canvas, _ = obj_draw(obj_canvas, obj_matched)
        t_obj = time.perf_counter() - t_obj_start

        desk_frame = future_desk.result()

        # Full opacity composite — hard pixel copy, no additive blending
        t_overlay_start = time.perf_counter()
        m_obj  = (obj_canvas  != original).any(axis=2)
        m_desk = (desk_frame  != original).any(axis=2)
        canvas[m_obj]  = obj_canvas[m_obj]
        canvas[m_desk] = desk_frame[m_desk]
        t_overlay = time.perf_counter() - t_overlay_start

        latest_annotated[cam_name] = canvas

        t_total = time.perf_counter() - t0
        _log_timings(cam_name, {
            "pose":    t_pose,
            "obj":     t_obj,
            "desk":    0,
            "overlay": t_overlay,
            "total":   t_total,
        })

        return canvas

    except Exception as e:
        print(f"AI Error: {e}")
        return frame


def generate_frames(cam_name, frames_override=None, recorder=None):
    """Drive the AI pipeline. WebRTC reads from latest_annotated directly — no MJPEG needed."""
    last_frame_id = None

    while True:
        try:
            frame_deque = frames_override.get(cam_name)
            if not frame_deque or len(frame_deque) == 0:
                time.sleep(0.01)
                continue

            frame = frame_deque[0]

            current_frame_id = id(frame)
            if current_frame_id == last_frame_id:
                time.sleep(0.001)
                continue
            last_frame_id = current_frame_id

        except (IndexError, KeyError):
            time.sleep(0.01)
            continue

        run_ai_pipeline(cam_name, frame.copy())
        yield  # keeps it a generator so wcapp.py loop still works