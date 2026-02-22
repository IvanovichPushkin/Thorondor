"""
GPU/provider detection for ONNX Runtime and PyTorch.
Priority: CUDA (NVIDIA) → DirectML (Windows AMD/Intel GPU) → CPU

NOTE FOR AMD GPU USERS:
  DirectML requires a different package than standard onnxruntime.
  Run:  pip uninstall onnxruntime
        pip install onnxruntime-directml
  Then restart the app. Your AMD Radeon GPU will be used automatically.
"""
import sys
import numpy as np


def get_device() -> str:
    """Returns 'cuda', 'directml', or 'cpu'."""
    # 1. Try CUDA (NVIDIA)
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[GPU] CUDA available: {name}")
            return "cuda"
    except Exception:
        pass

    # 2. Try DirectML (AMD / Intel on Windows)
    if sys.platform == "win32":
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "DmlExecutionProvider" in providers:
                print("[GPU] DirectML available — AMD/Intel GPU will be used")
                return "directml"
            else:
                # Check if the standard (non-DML) package is installed
                try:
                    import importlib.metadata
                    pkg = importlib.metadata.packages_distributions()
                    has_dml = any("onnxruntime-directml" in str(v) for v in pkg.values())
                    if not has_dml:
                        print(
                            "[GPU] AMD/Intel GPU detected but DirectML is not available.\n"
                            "      To enable GPU acceleration, run:\n"
                            "        pip uninstall onnxruntime\n"
                            "        pip install onnxruntime-directml\n"
                            "      Then restart the app."
                        )
                except Exception:
                    print(
                        "[GPU] No DirectML provider found. If you have an AMD/Intel GPU, run:\n"
                        "        pip uninstall onnxruntime\n"
                        "        pip install onnxruntime-directml"
                    )
        except Exception:
            pass

    print("[GPU] Using CPU — no GPU acceleration active")
    return "cpu"


def get_ort_providers(device: str) -> list[str]:
    """Ordered list of ONNX Runtime providers for the detected device."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        return ["CPUExecutionProvider"]

    if device == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "directml" and "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def make_session_options(num_threads: int = 1):
    """
    Create ORT SessionOptions with conservative thread counts.
    With 3 parallel sessions on a 4-core CPU, 1 thread per session
    leaves cores free for the draw path and event loop.
    """
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = num_threads
        opts.intra_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return opts
    except Exception:
        return None


def configure_onnx_session(model, model_path: str, providers: list[str]):
    """
    Replace the ONNX Runtime session inside a loaded Ultralytics YOLO model
    with one using the correct GPU providers and thread settings.

    Ultralytics stores the session at model.model.session (AutoBackend).
    We recreate it with our providers and SessionOptions, then write it back.
    This prevents worker threads from triggering a second session load.
    """
    try:
        import onnxruntime as ort

        opts = make_session_options(num_threads=1)

        new_session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers,
        )

        # Ultralytics AutoBackend stores the session at model.model.session
        backend = getattr(model, "model", None)
        if backend is not None and hasattr(backend, "session"):
            backend.session = new_session
            print(f"[GPU] Session configured: {providers[0]} | threads=1")
            return True

        # Fallback paths for different Ultralytics versions
        for attr_path in ["session", "predictor.model.session"]:
            obj = model
            for part in attr_path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "run"):
                # Found a session — replace it
                parent = model
                parts = attr_path.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_session)
                print(f"[GPU] Session configured (fallback path): {providers[0]}")
                return True

        print(f"[GPU] Could not locate ONNX session in model — providers may not apply")
        return False

    except Exception as e:
        print(f"[GPU] Session configuration failed: {e}")
        return False