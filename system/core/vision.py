import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from core.detections.pose import process as pose_process
from core.detections.desk import process as desk_process
from core.detections.object import process as obj_process

latest_annotated = {}
executor = ThreadPoolExecutor(max_workers=6)

# ── FPS / timing tracker ─────────────────────────────────────────────────────
_frame_times = {}   # cam_name -> last frame timestamp
_fps_log_interval = 5.0   # print summary every N seconds
_timing_accum = {}  # cam_name -> list of timing dicts


def _log_timings(cam_name, timings):
    """Accumulate timings and print a summary every FPS_LOG_INTERVAL seconds."""
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
        return sum(s[key] for s in samples) / n * 1000  # ms

    print(
        f"\n[PERF] [{cam_name}] last {elapsed:.1f}s | "
        f"FPS: {fps:.1f} | "
        f"pose: {avg('pose'):.0f}ms | "
        f"obj: {avg('obj'):.0f}ms | "
        f"desk: {avg('desk'):.0f}ms | "
        f"overlay: {avg('overlay'):.0f}ms | "
        f"jpeg: {avg('jpeg'):.0f}ms | "
        f"total: {avg('total'):.0f}ms"
    )

    _frame_times[cam_name] = now
    _timing_accum[cam_name] = []


# ─────────────────────────────────────────────────────────────────────────────

def overlay_annotations(base, annotated, original, alpha=0.7):
    """Extract annotations drawn on `annotated` vs `original`, apply at `alpha` onto `base`."""
    diff = cv2.absdiff(annotated, original)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    ann_only  = cv2.bitwise_and(annotated, annotated, mask=mask)
    base_only = cv2.bitwise_and(base, base, mask=mask)
    blended   = cv2.addWeighted(ann_only, alpha, base_only, 1 - alpha, 0)

    untouched = cv2.bitwise_and(base, base, mask=mask_inv)
    return cv2.add(untouched, blended)


def run_ai_pipeline(cam_name, frame):
    try:
        t0 = time.perf_counter()
        original = frame.copy()

        future_pose = executor.submit(pose_process, original.copy(), cam_name)
        future_obj  = executor.submit(obj_process,  original.copy(), cam_name)

        t_pose_start = time.perf_counter()
        annotated_frame, person_boxes = future_pose.result()
        t_pose = time.perf_counter() - t_pose_start

        t_obj_start = time.perf_counter()
        obj_frame, _ = future_obj.result()
        t_obj = time.perf_counter() - t_obj_start

        t_desk_start = time.perf_counter()
        future_desk = executor.submit(desk_process, original.copy(), cam_name, person_boxes)
        desk_frame  = future_desk.result()
        t_desk = time.perf_counter() - t_desk_start

        t_overlay_start = time.perf_counter()
        canvas = original.copy()
        canvas = overlay_annotations(canvas, annotated_frame, original, alpha=0.7)
        canvas = overlay_annotations(canvas, desk_frame,      original, alpha=0.7)
        canvas = overlay_annotations(canvas, obj_frame,       original, alpha=0.7)
        t_overlay = time.perf_counter() - t_overlay_start

        latest_annotated[cam_name] = canvas

        t_total = time.perf_counter() - t0
        _log_timings(cam_name, {
            "pose":    t_pose,
            "obj":     t_obj,
            "desk":    t_desk,
            "overlay": t_overlay,
            "jpeg":    0,   # filled below
            "total":   t_total,
        })

        return canvas

    except Exception as e:
        print(f"AI Error: {e}")
        return frame


def generate_frames(cam_name, frames_override=None, recorder=None):
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

        processed = run_ai_pipeline(cam_name, frame.copy())

        t_jpeg = time.perf_counter()
        ret, buffer = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
        t_jpeg = time.perf_counter() - t_jpeg

        # Patch jpeg time into last timing sample
        if cam_name in _timing_accum and _timing_accum[cam_name]:
            _timing_accum[cam_name][-1]["jpeg"] = t_jpeg

        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')