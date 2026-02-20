import cv2
import csv
from datetime import datetime

from core.yolo_desk_models import yolo_desk
from core.config import (
    YOLO_DESK_CONF_THRESHOLD,
    LOG_FILE,
    CSV_FILE,
)

DESK_LABELS = {
    0: "Desk",
}

DESK_COLOR = (0, 128, 255)

_SNAP_GRID = 10
IOU_THRESHOLD = 0.3

_last_desk_state: dict[str, dict] = {}  # cam_name -> { instance_id -> box_tuple }
_next_instance_id = 0


def _get_label(cls):
    return DESK_LABELS.get(cls, f"Class{cls}")


def _snap(value: int, grid: int = _SNAP_GRID) -> int:
    return (value // grid) * grid


def _new_id():
    global _next_instance_id
    _next_instance_id += 1
    return _next_instance_id


def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)


def _snap_box(box):
    """Snap all four coords to grid to absorb jitter before IoU matching."""
    x1, y1, x2, y2 = box
    return (_snap(x1), _snap(y1), _snap(x2), _snap(y2))


def _match_desks(prev_instances, current_boxes):
    """
    Match current desk boxes to previous instances via IoU.

    prev_instances: dict { instance_id -> box_tuple }
    current_boxes:  list of box_tuples

    Returns:
        matched:      dict { instance_id -> box_tuple }  (surviving instances)
        new_boxes:    list of box_tuples                 (no match, newly appeared)
        lost_ids:     list of instance_ids that disappeared
    """
    used_prev = set()
    matched = {}
    new_boxes = []

    for box in current_boxes:
        best_id, best_iou = None, IOU_THRESHOLD
        for inst_id, prev_box in prev_instances.items():
            if inst_id in used_prev:
                continue
            iou = _iou(box, prev_box)
            if iou > best_iou:
                best_iou = iou
                best_id = inst_id

        if best_id is not None:
            matched[best_id] = box
            used_prev.add(best_id)
        else:
            new_boxes.append(box)

    lost_ids = [i for i in prev_instances if i not in used_prev]
    return matched, new_boxes, lost_ids


def process(frame, cam_name, person_boxes=None):
    if person_boxes is None:
        person_boxes = []

    desk_results = yolo_desk.predict(
        frame, imgsz=640, conf=YOLO_DESK_CONF_THRESHOLD, verbose=False
    )
    current_boxes = []

    for r in desk_results:
        for box in r.boxes:
            cls             = int(box.cls[0].item())
            conf_val        = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_area        = (x2 - x1) * (y2 - y1)

            if box_area < 1000:
                continue

            overlaps_person = False
            for px1, py1, px2, py2 in person_boxes:
                ix1 = max(x1, px1); iy1 = max(y1, py1)
                ix2 = min(x2, px2); iy2 = min(y2, py2)
                if ix1 < ix2 and iy1 < iy2:
                    if (ix2 - ix1) * (iy2 - iy1) / (box_area + 1e-6) > 0.6:
                        overlaps_person = True
                        break

            if overlaps_person:
                continue

            label = _get_label(cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), DESK_COLOR, 2)
            cv2.putText(
                frame, f"{label} {conf_val:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, DESK_COLOR, 2
            )

            # Snap before storing to absorb jitter in IoU comparisons
            current_boxes.append(_snap_box((x1, y1, x2, y2)))

    prev_instances = _last_desk_state.get(cam_name, {})
    matched, new_boxes, lost_ids = _match_desks(prev_instances, current_boxes)

    for box in new_boxes:
        inst_id = _new_id()
        matched[inst_id] = box
        timestamp = datetime.now()
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: Desk Detected (id={inst_id})\n")
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, cam_name, "Desk Detected", "desk", inst_id])

    for inst_id in lost_ids:
        timestamp = datetime.now()
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: Desk Left (id={inst_id})\n")
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, cam_name, "Desk Left", "desk", inst_id])

    _last_desk_state[cam_name] = matched
    return frame