from ultralytics import YOLO
import torch
from core.config import YOLO_MODEL_PATH

# AUTO GPU / CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] YOLO object detection using device: {DEVICE}")

yolo = YOLO(YOLO_MODEL_PATH)
yolo.fuse()
yolo.to(DEVICE)