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

# Confirmed from best.pt: model stores numeric strings '0','1','2','3'
# Correct mapping from /content/Argus_Object_Detection-1/data.yaml
OBJECT_LABELS = {
    0: "Phone",
    1: "Calculator",
    2: "Smartwatch",
    3: "Watch",
}

# Box colors per class
OBJECT_COLORS = {
    0: (0, 165, 255),   # orange - Phone
    1: (255, 0, 255),   # magenta - Calculator
    2: (0, 255, 255),   # yellow - Smartwatch
    3: (255, 255, 0),   # cyan - Watch
}

def _get_label(cls):
    """Always use override — model names are numeric strings."""
    return OBJECT_LABELS.get(cls, f"Class{cls}")

def _get_color(cls):
    return OBJECT_COLORS.get(cls, (255, 0, 0))


_last_object_state = {}
_last_alert_time   = {}


def process(frame, cam_name):
    results = yolo.predict(frame, imgsz=640, conf=YOLO_CONF_THRESHOLD, verbose=False)
    current_objects = set()

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
            current_objects.add(label)

    prev_objects = _last_object_state.get(cam_name, set())

    for obj in current_objects - prev_objects:
        timestamp = datetime.now()
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: Object Detected: {obj}\n")
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, cam_name, "Object Detected", obj, ""])
        if obj.lower() in SUSPICIOUS_LABELS:
            alert_key = f"{cam_name}_{obj}"
            now = time.time()
            if alert_key not in _last_alert_time or now - _last_alert_time[alert_key] > ALERT_COOLDOWN:
                for number in PHONE_NUMBERS:
                    gsm.send_sms(number, f"ALERT: {obj} detected on {cam_name}")
                _last_alert_time[alert_key] = now

    for obj in prev_objects - current_objects:
        timestamp = datetime.now()
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] {cam_name}: Object Left: {obj}\n")
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, cam_name, "Object Left", obj, ""])

    _last_object_state[cam_name] = current_objects
    return frame, current_objects