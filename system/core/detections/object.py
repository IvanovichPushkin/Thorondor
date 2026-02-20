import cv2
import csv
import time
from datetime import datetime

from core.yolo_models import yolo
from core.config import (
    YOLO_CONF_THRESHOLD,
    SUSPICIOUS_LABELS,
    ALERT_COOLDOWN,
    PHONE_NUMBERS,
    LOG_FILE,
    CSV_FILE,
)
from core.gsm import gsm

OBJECT_LABELS = {
    0: "Phone",
    1: "Calculator",
    2: "Smartwatch",
    3: "Watch",
}

OBJECT_COLORS = {
    0: (0, 225, 255),
    1: (0, 225, 255),
    2: (0, 225, 255),
    3: (0, 225, 255),
}

def _get_label(cls):
    return OBJECT_LABELS.get(cls, f"Class{cls}")

def _get_color(cls):
    return OBJECT_COLORS.get(cls, (0, 225, 255))


_last_object_state = {}  # cam_name -> dict of { instance_key -> label }
_last_alert_time   = {}

IOU_THRESHOLD = 0.3  # Minimum IoU to consider two boxes the same object


def _iou(boxA, boxB):
    """Compute Intersection over Union between two (x1,y1,x2,y2) boxes."""
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


def _match_instances(prev_instances, current_detections):
    """
    Match current detections to previous instances using IoU.

    prev_instances: dict { instance_id -> (label, box) }
    current_detections: list of (label, box, conf)

    Returns:
        matched_instances: dict { instance_id -> (label, box) }  (carried-over IDs)
        new_detections:    list of (label, box, conf)             (no match found)
        lost_ids:          list of instance_ids that disappeared
    """
    used_prev = set()
    matched_instances = {}
    unmatched_current = []

    for label, box, conf in current_detections:
        best_id, best_iou = None, IOU_THRESHOLD
        for inst_id, (prev_label, prev_box) in prev_instances.items():
            if inst_id in used_prev:
                continue
            if prev_label != label:
                continue
            iou = _iou(box, prev_box)
            if iou > best_iou:
                best_iou = iou
                best_id = inst_id

        if best_id is not None:
            matched_instances[best_id] = (label, box)
            used_prev.add(best_id)
        else:
            unmatched_current.append((label, box, conf))

    lost_ids = [i for i in prev_instances if i not in used_prev]
    return matched_instances, unmatched_current, lost_ids


_next_instance_id = 0

def _new_id():
    global _next_instance_id
    _next_instance_id += 1
    return _next_instance_id


def process(frame, cam_name):
    results = yolo.predict(frame, imgsz=640, conf=YOLO_CONF_THRESHOLD, verbose=False)
    current_detections = []  # list of (label, box_tuple, conf)

    for r in results:
        for box in r.boxes:
            cls             = int(box.cls[0].item())
            conf_val        = float(box.conf[0].item())
            label           = _get_label(cls)
            color           = _get_color(cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf_val:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            current_detections.append((label, (x1, y1, x2, y2), conf_val))

    prev_instances = _last_object_state.get(cam_name, {})
    matched, new_detections, lost_ids = _match_instances(prev_instances, current_detections)

    # Assign new IDs to unmatched detections (newly appeared objects)
    for label, box, conf in new_detections:
        inst_id = _new_id()
        matched[inst_id] = (label, box)

        timestamp = datetime.now()
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: Object Detected: {label} (id={inst_id})\n")
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, cam_name, "Object Detected", label, inst_id])

        if label.lower() in SUSPICIOUS_LABELS:
            alert_key = f"{cam_name}_{label}"
            now = time.time()
            if alert_key not in _last_alert_time or now - _last_alert_time[alert_key] > ALERT_COOLDOWN:
                for number in PHONE_NUMBERS:
                    gsm.send_sms(number, f"ALERT: {label} detected on {cam_name}")
                _last_alert_time[alert_key] = now

    # Log objects that have left the frame
    for inst_id in lost_ids:
        label, _ = prev_instances[inst_id]
        timestamp = datetime.now()
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: Object Left: {label} (id={inst_id})\n")
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, cam_name, "Object Left", label, inst_id])

    _last_object_state[cam_name] = matched

    current_labels = {label for label, _ in matched.values()}
    return frame, current_labels