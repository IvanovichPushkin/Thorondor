import os
import numpy as np
from ultralytics import YOLO
from core.config import YOLO_MODEL_PATH
from core.gpu_provider import get_device, get_ort_providers, configure_onnx_session

_onnx_path  = YOLO_MODEL_PATH.replace(".pt", ".onnx")
_model_path = _onnx_path if os.path.exists(_onnx_path) else YOLO_MODEL_PATH
_is_onnx    = _model_path.endswith(".onnx")

DEVICE    = get_device()
PROVIDERS = get_ort_providers(DEVICE)

print(f"[INFO] Object model : {os.path.basename(_model_path)} | device: {DEVICE}")

yolo = YOLO(_model_path, task="detect")

if not _is_onnx:
    try:
        import torch
        yolo.fuse()
        yolo.to("cuda" if DEVICE == "cuda" else "cpu")
    except Exception:
        pass
else:
    # Warmup with a valid-sized frame to force Ultralytics to initialize
    # the predictor + ONNX session NOW (in the main thread), so worker
    # threads don't trigger a second "Loading..." on their first call.
    _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    try:
        yolo.predict(_dummy, imgsz=320, verbose=False)
        print(f"[INFO] Object model warmed up")
    except Exception as e:
        print(f"[WARN] Object model warmup failed: {e}")

    # Swap session to GPU provider + 1 thread per session
    configure_onnx_session(yolo, _model_path, PROVIDERS)