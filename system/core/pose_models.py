import os
import torch
from ultralytics import YOLO
from core.config import POSE_MODEL_PATH

_onnx_path = POSE_MODEL_PATH.replace(".pt", ".onnx")
_model_path = _onnx_path if os.path.exists(_onnx_path) else POSE_MODEL_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_mode = "ONNX ✓ fast mode" if _model_path.endswith(".onnx") else "PyTorch (run export_onnx.py for speedup)"
print(f"[INFO] Pose model: {os.path.basename(_model_path)} | {_mode} | device: {DEVICE}")

pose_model = YOLO(_model_path, task="pose")
if _model_path.endswith(".pt"):
    pose_model.fuse()
    pose_model.to(DEVICE)