import os
import numpy as np
from ultralytics import YOLO
from core.config import POSE_MODEL_PATH
from core.gpu_provider import get_device, get_ort_providers, configure_onnx_session

_onnx_path  = POSE_MODEL_PATH.replace(".pt", ".onnx")
_model_path = _onnx_path if os.path.exists(_onnx_path) else POSE_MODEL_PATH
_is_onnx    = _model_path.endswith(".onnx")

DEVICE    = get_device()
PROVIDERS = get_ort_providers(DEVICE)

_mode = "ONNX ✓ fast mode" if _is_onnx else "PyTorch (run export_onnx.py for speedup)"
print(f"[INFO] Pose model   : {os.path.basename(_model_path)} | {_mode} | device: {DEVICE}")

pose_model = YOLO(_model_path, task="pose")

if not _is_onnx:
    try:
        import torch
        pose_model.fuse()
        pose_model.to("cuda" if DEVICE == "cuda" else "cpu")
    except Exception:
        pass
else:
    _dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    try:
        pose_model.predict(_dummy, imgsz=256, verbose=False)
        print(f"[INFO] Pose model warmed up")
    except Exception as e:
        print(f"[WARN] Pose model warmup failed: {e}")

    configure_onnx_session(pose_model, _model_path, PROVIDERS)