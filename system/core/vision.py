import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from core.detections.pose import process as pose_process
from core.detections.desk import process as desk_process
from core.detections.object import process as obj_process

latest_annotated = {}
executor = ThreadPoolExecutor(max_workers=4)  # bumped to 4 since desk now runs async too


def overlay_annotations(base, annotated, original, alpha=0.7):
    """Extract annotations drawn on `annotated` vs `original`, apply at `alpha` onto `base`."""
    diff = cv2.absdiff(annotated, original)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Where annotations exist: blend annotation color over base
    ann_only  = cv2.bitwise_and(annotated, annotated, mask=mask)
    base_only = cv2.bitwise_and(base, base, mask=mask)
    blended   = cv2.addWeighted(ann_only, alpha, base_only, 1 - alpha, 0)

    # Where no annotations: keep base untouched
    untouched = cv2.bitwise_and(base, base, mask=mask_inv)
    return cv2.add(untouched, blended)


def run_ai_pipeline(cam_name, frame):
    try:
        original = frame.copy()

        # Run pose + object in parallel
        future_pose = executor.submit(pose_process, original.copy(), cam_name)
        future_obj  = executor.submit(obj_process,  original.copy(), cam_name)

        # Pose must finish first so we get person_boxes for desk overlap rejection
        annotated_frame, person_boxes = future_pose.result()
        obj_frame, _                  = future_obj.result()

        # FIX #1: Submit desk_process to executor (was blocking main thread before)
        future_desk = executor.submit(desk_process, original.copy(), cam_name, person_boxes)
        desk_frame  = future_desk.result()

        # Start from the untouched camera frame, layer each at 70%
        canvas = original.copy()
        canvas = overlay_annotations(canvas, annotated_frame, original, alpha=0.7)
        canvas = overlay_annotations(canvas, desk_frame,      original, alpha=0.7)
        canvas = overlay_annotations(canvas, obj_frame,       original, alpha=0.7)

        latest_annotated[cam_name] = canvas
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

        ret, buffer = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')