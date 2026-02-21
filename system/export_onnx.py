# This is to maximize speed for potato pcs

"""
Run this ONCE from the system/ directory to convert all .pt models to ONNX.
After running, restart the app — it will automatically use the faster ONNX models.

Usage:
    cd system
    python export_onnx.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import POSE_MODEL_PATH, YOLO_MODEL_PATH, YOLO_DESK_MODEL_PATH
from ultralytics import YOLO

def export(pt_path, imgsz, task_label):
    onnx_path = pt_path.replace(".pt", ".onnx")
    if os.path.exists(onnx_path):
        print(f"[SKIP] Already exported: {onnx_path}")
        return
    print(f"[EXPORT] {task_label}: {pt_path} -> {onnx_path} (imgsz={imgsz})")
    model = YOLO(pt_path)
    model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,   # graph simplification = faster inference
        opset=17,
        dynamic=False,   # static shape = faster on CPU
    )
    print(f"[DONE] {onnx_path}")

if __name__ == "__main__":
    export(POSE_MODEL_PATH,      imgsz=256, task_label="Pose")
    export(YOLO_MODEL_PATH,      imgsz=320, task_label="Object")
    export(YOLO_DESK_MODEL_PATH, imgsz=320, task_label="Desk")
    print("\n[ALL DONE] Restart the app to use ONNX models.")