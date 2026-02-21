from ultralytics import YOLO
import torch
from core.config import POSE_MODEL_PATH

# AUTO GPU / CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] YOLO pose detection using device: {DEVICE}")

pose_model = YOLO(POSE_MODEL_PATH)
pose_model.fuse()
pose_model.to(DEVICE)