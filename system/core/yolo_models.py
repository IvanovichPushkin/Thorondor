import os
import torch
from ultralytics import YOLO
from core.config import YOLO_MODEL_PATH

_onnx_path = YOLO_MODEL_PATH.replace(".pt", ".onnx")
_model_path = _onnx_path if os.path.exists(_onnx_path) else YOLO_MODEL_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Object model: {os.path.basename(_model_path)} | device: {DEVICE}")

yolo = YOLO(_model_path, task="detect")
if _model_path.endswith(".pt"):
    yolo.fuse()
    yolo.to(DEVICE)