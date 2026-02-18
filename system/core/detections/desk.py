import cv2
import csv
from datetime import datetime

from core.yolo_desk_models import yolo_desk
from core.config import (
    YOLO_DESK_CONF_THRESHOLD,
    LOG_FILE,
    CSV_FILE,
)

# Confirmed from best.pt: yolov9t, 1 class stored as numeric '0' = desk
DESK_LABELS = {
    0: "Desk",
}

DESK_COLOR = (0, 128, 255)  # orange-blue

def _get_label(cls):
    """Always use override — model name is numeric string '0'."""
    return DESK_LABELS.get(cls, f"Class{cls}")

# Accept all class indices — only 1 class anyway
DESK_CLASS_INDICES = set(yolo_desk.names.keys())

_last_desk_state = {}


def process(frame, cam_name, person_boxes=None):
    if person_boxes is None:
        person_boxes = []

    desk_results = yolo_desk.predict(
        frame, imgsz=640, conf=YOLO_DESK_CONF_THRESHOLD, verbose=False
    )
    current_desks = set()

    for r in desk_results:
        for box in r.boxes:
            cls             = int(box.cls[0].item())
            conf_val        = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_area        = (x2 - x1) * (y2 - y1)

            # Reject tiny noise boxes
            if box_area < 1000:
                continue

            # Reject if >60% of box overlaps a detected person
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
            current_desks.add(f"{x1}_{y1}_{x2}_{y2}")

    prev_desks = _last_desk_state.get(cam_name, set())

    for _ in current_desks - prev_desks:
        timestamp = datetime.now()
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: Desk Detected\n")
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, cam_name, "Desk Detected", "desk", ""])

    for _ in prev_desks - current_desks:
        timestamp = datetime.now()
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: Desk Left\n")
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, cam_name, "Desk Left", "desk", ""])

    _last_desk_state[cam_name] = current_desks
    return frame