import os
import numpy as np
from ultralytics import YOLO
from core.config import YOLO_DESK_MODEL_PATH
from core.gpu_provider import get_device, get_ort_providers, configure_onnx_session

_onnx_path  = YOLO_DESK_MODEL_PATH.replace(".pt", ".onnx")
_model_path = _onnx_path if os.path.exists(_onnx_path) else YOLO_DESK_MODEL_PATH
_is_onnx    = _model_path.endswith(".onnx")

DEVICE    = get_device()
PROVIDERS = get_ort_providers(DEVICE)

print(f"[INFO] Desk model   : {os.path.basename(_model_path)} | device: {DEVICE}")

yolo_desk = YOLO(_model_path, task="detect")

if not _is_onnx:
    try:
        import torch
        yolo_desk.fuse()
        yolo_desk.to("cuda" if DEVICE == "cuda" else "cpu")
    except Exception:
        pass
else:
    _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    try:
        yolo_desk.predict(_dummy, imgsz=320, verbose=False)
        print(f"[INFO] Desk model warmed up")
    except Exception as e:
        print(f"[WARN] Desk model warmup failed: {e}")

    configure_onnx_session(yolo_desk, _model_path, PROVIDERS)