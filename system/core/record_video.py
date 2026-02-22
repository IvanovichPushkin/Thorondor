import os
import subprocess
import threading
import platform
import time
from datetime import datetime

from core.config import FRAME_WIDTH, FRAME_HEIGHT


class VideoRecorder:
    def __init__(self, fps=15.0):
        self.fps           = fps
        self.recording     = False
        self.finalizing    = False
        self.directory_set = True
        self.output_dir    = os.path.join(os.getcwd(), "recordings")
        os.makedirs(self.output_dir, exist_ok=True)

        current_dir     = os.path.dirname(os.path.abspath(__file__))
        self.ffmpeg_exe = os.path.normpath(
            os.path.join(current_dir, "..", "bin", "ffmpeg.exe")
        )

        self.current_file = "None"
        self.status_msg   = "Ready"

        self._cameras      = {}
        self._feed_threads = []

    def _kill_zombies(self):
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/IM", "ffmpeg.exe", "/T"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    def _spawn_ffmpeg(self, output_path, width, height):
        cmd = [
            self.ffmpeg_exe, "-y",
            "-f",        "rawvideo",
            "-vcodec",   "rawvideo",
            "-pix_fmt",  "bgr24",
            "-s",        f"{width}x{height}",
            "-r",        str(self.fps),
            "-use_wallclock_as_timestamps", "1",
            "-i",        "pipe:0",
            "-c:v",      "libx264",
            "-preset",   "ultrafast",
            "-tune",     "zerolatency",
            "-pix_fmt",  "yuv420p",
            "-vsync",    "vfr",
            "-movflags", "+faststart",
            output_path,
        ]
        si = None
        if platform.system() == "Windows":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=si,
        )

    def _get_frame_size(self, cam_name):
        from core.vision import latest_annotated, latest_raw
        for _ in range(30):
            ann   = latest_annotated.get(cam_name)
            raw   = latest_raw.get(cam_name)
            frame = ann if ann is not None else raw
            if frame is not None:
                h, w = frame.shape[:2]
                print(f"[INFO] Recorder [{cam_name}] frame size: {w}x{h}")
                return w, h
            time.sleep(0.1)
        print(f"[WARN] Recorder [{cam_name}] no frame found — using config {FRAME_WIDTH}x{FRAME_HEIGHT}")
        return FRAME_WIDTH, FRAME_HEIGHT

    def start(self, cam_name=None, cam_names=None):
        if self.recording or self.finalizing:
            return
        self._kill_zombies()
        self._cameras      = {}
        self._feed_threads = []

        if cam_names:
            targets = cam_names
        elif cam_name:
            targets = [cam_name]
        else:
            from core.config import CAMERA_SOURCES
            targets = list(CAMERA_SOURCES.keys())

        print(f"[INFO] Recorder starting for cameras: {targets}")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = os.path.join(self.output_dir, f"Argus_Record_{ts}.mp4")

        try:
            for cn in targets:
                w, h      = self._get_frame_size(cn)
                safe_name = cn.replace(" ", "_")
                ann_path  = os.path.join(self.output_dir, f"Argus_{safe_name}_{ts}.mp4")
                raw_path  = os.path.join(self.output_dir, f"Argus_{safe_name}_{ts}_raw.mp4")

                proc_ann = self._spawn_ffmpeg(ann_path, w, h)
                proc_raw = self._spawn_ffmpeg(raw_path, w, h)

                time.sleep(0.3)
                if proc_ann.poll() is not None or proc_raw.poll() is not None:
                    print(f"[ERROR] Recorder [{cn}] ffmpeg failed to start")
                    continue

                self._cameras[cn] = {
                    "proc_ann":       proc_ann,
                    "proc_raw":       proc_raw,
                    "ann_path":       ann_path,
                    "raw_path":       raw_path,
                    "frames_written": 0,
                    "start_time":     None,
                    "end_time":       None,
                }
                print(f"[INFO] Recorder [{cn}] ffmpeg started OK")

            if not self._cameras:
                self.status_msg = "Failed to start any camera recorder"
                return

            self.recording  = True
            self.status_msg = f"Recording {len(self._cameras)} camera(s)..."

            for cn in self._cameras:
                t = threading.Thread(target=self._feed_frames, args=(cn,), daemon=True)
                t.start()
                self._feed_threads.append(t)

        except Exception as e:
            import traceback
            print(f"[ERROR] Recorder failed to start: {e}")
            traceback.print_exc()
            self.status_msg = f"Error: {e}"

    def _feed_frames(self, cam_name):
        from core.vision import latest_annotated, latest_raw

        cam        = self._cameras[cam_name]
        proc_ann   = cam["proc_ann"]
        proc_raw   = cam["proc_raw"]
        last_ann_id    = None
        last_raw_id    = None
        frames_written = 0

        print(f"[INFO] Recorder [{cam_name}] feed thread started")

        while self.recording:
            annotated = latest_annotated.get(cam_name)
            raw       = latest_raw.get(cam_name)

            wrote = False

            if annotated is not None and id(annotated) != last_ann_id:
                last_ann_id = id(annotated)
                try:
                    proc_ann.stdin.write(annotated.tobytes())
                    if frames_written == 0:
                        cam["start_time"] = time.perf_counter()
                        print(f"[INFO] Recorder [{cam_name}] first frame written: shape={annotated.shape}")
                    frames_written += 1
                    cam["frames_written"] = frames_written
                    cam["end_time"]       = time.perf_counter()
                    wrote = True
                except Exception as e:
                    print(f"[ERROR] Recorder [{cam_name}] annotated write failed: {e}")
                    break

            if raw is not None and id(raw) != last_raw_id:
                last_raw_id = id(raw)
                try:
                    proc_raw.stdin.write(raw.tobytes())
                    wrote = True
                except Exception as e:
                    print(f"[ERROR] Recorder [{cam_name}] raw write failed: {e}")
                    break

            if not wrote:
                time.sleep(0.002)

        print(f"[INFO] Recorder [{cam_name}] feed thread done. Frames: {frames_written}")

    def stop(self):
        if not self.recording:
            return
        self.recording  = False
        self.finalizing = True
        self.status_msg = "Finalizing..."

        def finalize():
            for t in self._feed_threads:
                if t.is_alive():
                    t.join(timeout=3)
            self._feed_threads.clear()

            for cn, cam in self._cameras.items():
                for proc in (cam["proc_ann"], cam["proc_raw"]):
                    try:
                        proc.stdin.close()
                        proc.wait(timeout=15)
                    except Exception:
                        proc.kill()

            self._cleanup()
            self.finalizing = False

        threading.Thread(target=finalize).start()

    def _cleanup(self):
        time.sleep(0.5)
        saved = []
        for cn, cam in self._cameras.items():
            frames_written = cam.get("frames_written", 0)
            start_time     = cam.get("start_time")
            end_time       = cam.get("end_time")

            if frames_written > 1 and start_time and end_time:
                elapsed    = end_time - start_time
                actual_fps = round(frames_written / elapsed, 3) if elapsed > 0 else self.fps
                actual_fps = max(1.0, actual_fps)  # no upper cap — use true measured rate
            else:
                actual_fps = self.fps

            print(f"[INFO] Recorder [{cn}] actual FPS: {actual_fps:.2f} ({frames_written} frames)")

            for path in (cam["ann_path"], cam["raw_path"]):
                if not os.path.exists(path):
                    print(f"[WARN] Recorder [{cn}] file missing: {path}")
                    continue

                size = os.path.getsize(path)
                print(f"[INFO] Recorder [{cn}] {os.path.basename(path)} | {size} bytes")

                if size < 5000:
                    print(f"[WARN] Recorder [{cn}] deleting empty file: {os.path.basename(path)}")
                    os.remove(path)
                    continue

                fixed_path = path.replace(".mp4", "_fixed.mp4")
                try:
                    si = None
                    if platform.system() == "Windows":
                        si = subprocess.STARTUPINFO()
                        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                    result = subprocess.run(
                        [
                            self.ffmpeg_exe, "-y",
                            "-i",       path,
                            "-vf",      f"setpts=N/{actual_fps}/TB",
                            "-r",       str(actual_fps),
                            "-c:v",     "libx264",
                            "-preset",  "ultrafast",
                            "-pix_fmt", "yuv420p",
                            fixed_path,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        startupinfo=si,
                        timeout=120,
                    )
                    if result.returncode == 0 and os.path.exists(fixed_path) and os.path.getsize(fixed_path) > 5000:
                        os.remove(path)
                        os.rename(fixed_path, path)
                        print(f"[INFO] Recorder [{cn}] re-encoded at {actual_fps:.2f} fps: {os.path.basename(path)}")
                    else:
                        if os.path.exists(fixed_path):
                            os.remove(fixed_path)
                        print(f"[WARN] Recorder [{cn}] re-encode failed, keeping original: {os.path.basename(path)}")
                except Exception as e:
                    print(f"[WARN] Recorder [{cn}] re-encode error: {e}")

                saved.append(os.path.basename(path))

        self.status_msg = f"Saved {len(saved)} file(s)" if saved else "Recording failed (Empty)"