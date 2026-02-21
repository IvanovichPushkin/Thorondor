from ultralytics import YOLO
import torch
from core.config import YOLO_DESK_MODEL_PATH

# AUTO GPU / CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] YOLO desk detection using device: {DEVICE}")

yolo_desk = YOLO(YOLO_DESK_MODEL_PATH)
yolo_desk.fuse()
yolo_desk.to(DEVICE)