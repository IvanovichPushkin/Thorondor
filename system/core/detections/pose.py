import cv2
import csv
import numpy as np
import time
from datetime import datetime

from core.pose_models import pose_model
from core.config import (
    POSE_CONF_THRESHOLD,
    LOG_FILE,
    CSV_FILE,
    ALERT_COOLDOWN,
    PHONE_NUMBERS,
    SUSPICIOUS_LABELS,
)
from core.gsm import gsm

# Confirmed from best.pt: {0: 'cheating', 1: 'normal'}
POSE_LABELS = {
    0: "Cheating",
    1: "Normal",
}

LABEL_COLORS = {
    "Cheating": (0, 0, 255),   # red
    "Normal":   (0, 255, 0),   # green
}

# COCO 17-keypoint skeleton connections
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (0, 5), (0, 6), (5, 7), (7, 9),        # left arm
    (6, 8), (8, 10),                         # right arm
    (5, 6),                                  # shoulders
    (5, 11), (6, 12), (11, 12),             # torso
    (11, 13), (13, 15),                      # left leg
    (12, 14), (14, 16),                      # right leg
]

# Minimum keypoint confidence to draw (model gives [17,3] = x,y,conf per point)
KPT_CONF_THRESHOLD = 0.3

_last_behavior_state = {}
_last_alert_time = {}


def _get_label(cls):
    name = pose_model.names.get(cls, str(cls))
    # If stored as numeric string, use override
    if name == str(cls):
        return POSE_LABELS.get(cls, f"Class{cls}")
    return name.capitalize()


def _draw_skeleton(frame, keypoints_xy, keypoints_conf, color):
    """
    Draw skeleton using keypoint xy coords + per-point confidence.
    keypoints_xy:   shape [17, 2]
    keypoints_conf: shape [17]   — skip points below KPT_CONF_THRESHOLD
    """
    # Draw connections
    for i, j in SKELETON:
        if i >= len(keypoints_xy) or j >= len(keypoints_xy):
            continue
        # Skip if either endpoint is low confidence or at origin
        if keypoints_conf[i] < KPT_CONF_THRESHOLD or keypoints_conf[j] < KPT_CONF_THRESHOLD:
            continue
        pt1 = tuple(keypoints_xy[i].astype(int))
        pt2 = tuple(keypoints_xy[j].astype(int))
        if pt1[0] > 1 and pt1[1] > 1 and pt2[0] > 1 and pt2[1] > 1:
            cv2.line(frame, pt1, pt2, color, 2)

    # Draw keypoint dots
    for idx, (kpt, conf) in enumerate(zip(keypoints_xy, keypoints_conf)):
        if conf < KPT_CONF_THRESHOLD:
            continue
        x, y = int(kpt[0]), int(kpt[1])
        if x > 1 and y > 1:
            cv2.circle(frame, (x, y), 3, color, -1)


def process(frame, cam_name):
    """
    Run pose model on frame.
    - Draws bounding box + cheating/normal label per person
    - Draws skeleton using high-confidence keypoints only
    - Logs state changes
    Returns (annotated frame, person_boxes)
    """
    pose_results = pose_model.predict(
        frame, imgsz=320, conf=POSE_CONF_THRESHOLD, verbose=False
    )
    person_boxes = []

    for r in pose_results:
        boxes     = r.boxes     if (hasattr(r, "boxes")     and r.boxes     is not None) else []
        kpts_obj  = r.keypoints if (hasattr(r, "keypoints") and r.keypoints is not None) else None

        # xy:   [N, 17, 2]
        # conf: [N, 17]
        kpts_xy   = kpts_obj.xy.cpu().numpy()   if kpts_obj is not None else []
        kpts_conf = kpts_obj.conf.cpu().numpy() if kpts_obj is not None else []

        for idx, box in enumerate(boxes):
            cls             = int(box.cls[0].item())
            conf_val        = float(box.conf[0].item())
            label           = _get_label(cls)
            color           = LABEL_COLORS.get(label, (0, 255, 0))
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            person_boxes.append((x1, y1, x2, y2))

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label + confidence above box
            cv2.putText(
                frame,
                f"{label} {conf_val:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )

            # Skeleton — only if keypoints available for this person
            if idx < len(kpts_xy) and len(kpts_xy[idx]) > 0:
                kp_conf = kpts_conf[idx] if idx < len(kpts_conf) else np.ones(17)
                _draw_skeleton(frame, kpts_xy[idx], kp_conf, color)

            # --- LOGGING & ALERT LOGIC ---
            state_key = f"{cam_name}_{idx}"
            prev = _last_behavior_state.get(state_key, "Normal")
            
            if label != prev:
                _last_behavior_state[state_key] = label
                timestamp = datetime.now()
                
                # Write to detections.log (for Live Log View)
                with open(LOG_FILE, "a") as f:
                    f.write(
                        f"[{timestamp.strftime('%H:%M:%S')}] "
                        f"{cam_name}: Person {idx + 1} behavior: {label}\n"
                    )
                
                # Write to detections.csv
                with open(CSV_FILE, "a", newline="") as f:
                    csv.writer(f).writerow([
                        timestamp, cam_name, "Behavior Changed",
                        label, f"person_{idx + 1}"
                    ])

                # SMS Alert Logic
                if label.lower() in [s.lower() for s in SUSPICIOUS_LABELS]:
                    alert_key = f"{cam_name}_{idx}_{label}"
                    now = time.time()
                    if alert_key not in _last_alert_time or now - _last_alert_time[alert_key] > ALERT_COOLDOWN:
                        for number in PHONE_NUMBERS:
                            gsm.send_sms(number, f"ALERT: {label} behavior detected on {cam_name} (Person {idx+1})")
                        _last_alert_time[alert_key] = now

    return frame, person_boxes