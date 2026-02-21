import cv2
import time
import threading
import numpy as np
from core.detections.pose import predict as pose_predict, draw as pose_draw
from core.detections.desk import predict as desk_predict, draw as desk_draw
from core.detections.object import predict as obj_predict, draw as obj_draw

latest_annotated = {}
latest_raw       = {}   # raw camera frame with no AI annotations

# Optional callback — wcapp.py sets this to signal_new_frame so WebRTC recv()
# wakes immediately when a fresh frame is ready instead of polling.
on_frame_ready = None  # signature: (cam_name: str) -> None

# ── Pose background thread ────────────────────────────────────────────────────
_pose_cache      = {}
_pose_input      = {}
_pose_input_lock = threading.Lock()
_pose_cache_lock = threading.Lock()

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
        time.sleep(0.08)  # rate-limit to ~10fps

def _ensure_pose_thread(cam_name):
    key = f"_pose_thread_{cam_name}"
    if not globals().get(key):
        t = threading.Thread(target=_pose_worker, args=(cam_name,), daemon=True)
        t.start()
        globals()[key] = t

# ── Object background thread ──────────────────────────────────────────────────
_obj_cache      = {}
_obj_input      = {}
_obj_input_lock = threading.Lock()
_obj_cache_lock = threading.Lock()

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

# ── Desk background thread ────────────────────────────────────────────────────
# Desk YOLO also runs ~70ms but was logged as 0ms (hardcoded) while actually
# blocking silently on future_desk.result(). Same background-thread fix as obj.
_desk_cache      = {}
_desk_input      = {}
_desk_input_lock = threading.Lock()
_desk_cache_lock = threading.Lock()

def _desk_worker(cam_name):
    last_frame_id = None
    while True:
        with _desk_input_lock:
            frame = _desk_input.get(cam_name)
        if frame is None or id(frame) == last_frame_id:
            time.sleep(0.005)
            continue
        last_frame_id = id(frame)
        matched = desk_predict(frame, cam_name)
        with _desk_cache_lock:
            _desk_cache[cam_name] = matched

def _ensure_desk_thread(cam_name):
    key = f"_desk_thread_{cam_name}"
    if not globals().get(key):
        t = threading.Thread(target=_desk_worker, args=(cam_name,), daemon=True)
        t.start()
        globals()[key] = t

# ── FPS / timing tracker ─────────────────────────────────────────────────────
_frame_times      = {}
_fps_log_interval = 5.0
_timing_accum     = {}

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

        # One copy shared read-only across all three inference threads.
        original = frame.copy()

        # Store raw frame for clean (no-annotation) video recording
        latest_raw[cam_name] = original

        _ensure_pose_thread(cam_name)
        _ensure_obj_thread(cam_name)
        _ensure_desk_thread(cam_name)
        with _pose_input_lock:
            _pose_input[cam_name] = original
        with _obj_input_lock:
            _obj_input[cam_name] = original
        with _desk_input_lock:
            _desk_input[cam_name] = original

        # Grab all three caches — dict lookups only, ~0ms each.
        t_pose_start = time.perf_counter()
        with _pose_cache_lock:
            pose_matched = _pose_cache.get(cam_name)
        t_pose = time.perf_counter() - t_pose_start

        t_obj_start = time.perf_counter()
        with _obj_cache_lock:
            obj_matched = _obj_cache.get(cam_name)
        t_obj = time.perf_counter() - t_obj_start

        t_desk_start = time.perf_counter()
        with _desk_cache_lock:
            desk_matched = _desk_cache.get(cam_name)
        t_desk = time.perf_counter() - t_desk_start

        # Draw directly — no pixel-diff comparison.
        # Old: (frame != original).any(axis=2) on 1280x720 = 39-49ms per frame.
        # New: OpenCV draw on cached coords = ~1ms.
        t_overlay_start = time.perf_counter()
        canvas = original.copy()
        if pose_matched:
            canvas, _ = pose_draw(canvas, pose_matched)
        if obj_matched:
            canvas, _ = obj_draw(canvas, obj_matched)
        if desk_matched:
            canvas = desk_draw(canvas, desk_matched)
        t_overlay = time.perf_counter() - t_overlay_start

        latest_annotated[cam_name] = canvas

        # Wake WebRTC recv() immediately — no polling delay
        if on_frame_ready:
            on_frame_ready(cam_name)

        t_total = time.perf_counter() - t0
        _log_timings(cam_name, {
            "pose":    t_pose,
            "obj":     t_obj,
            "desk":    t_desk,
            "overlay": t_overlay,
            "total":   t_total,
        })

        return canvas

    except Exception as e:
        print(f"AI Error: {e}")
        return frame


def generate_frames(cam_name, frames_override=None, recorder=None):
    """Drive the AI pipeline. WebRTC reads from latest_annotated directly."""
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
                # Webcam = 30fps = new frame every ~33ms.
                # 5ms poll = ~6x less CPU waste than old 1ms poll.
                time.sleep(0.005)
                continue
            last_frame_id = current_frame_id

        except (IndexError, KeyError):
            time.sleep(0.01)
            continue

        run_ai_pipeline(cam_name, frame)
        yield