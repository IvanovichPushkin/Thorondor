# === LOGGING/CSV ===
LOG_FILE = "detections.log"
CSV_FILE  = "detections.csv"

# === CAMERAS ===
CAMERA_SOURCES = {
    # Format: "rtsp://username:password@ip:554/stream1" (stream1=HD, stream2=low)
    # All cameras must be on the same 2.4 GHz network
    "Camera 1": "rtsp://FINALBOSS:FINALBOSS@192.168.1.9:554/stream2",
    "Camera 2": "rtsp://FINALBOSS:FINALBOSS@192.168.1.9:554/stream2",
    "Camera 3": "rtsp://FINALBOSS:FINALBOSS@192.168.1.9:554/stream2",
}

# === RESOLUTION ===
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# === MODEL PATHS ===
YOLO_MODEL_PATH      = "../machine_learning/runs/argus_object_detection/weights/best.pt"
YOLO_DESK_MODEL_PATH = "../machine_learning/runs/argus_desk_detection/weights/best.pt"
POSE_MODEL_PATH      = "../machine_learning/runs/pose/argus_pose_estimation/weights/best.pt"

# === DETECTION THRESHOLDS ===
YOLO_CONF_THRESHOLD      = 0.75
YOLO_DESK_CONF_THRESHOLD = 0.45
POSE_CONF_THRESHOLD      = 0.15

# # === GSM / ALERTS ===
# PHONE_NUMBERS    = ["+639XXXXXXXXX", "+639YYYYYYYYY"]
# ALERT_COOLDOWN   = 10

# SUSPICIOUS_LABELS = ["phone", "smartwatch", "watch", "calculator", "cheating"]