import os
import subprocess
import threading
import platform
import time
from datetime import datetime
from collections import deque

from core.config import FRAME_WIDTH, FRAME_HEIGHT


class VideoRecorder:
    def __init__(self, fps=15.0):
        self.fps           = fps
        self.recording     = False
        self.finalizing    = False
        self.directory_set = True
        self.output_dir    = os.path.join(os.getcwd(), "recordings")
        os.makedirs(self.output_dir, exist_ok=True)

        current_dir      = os.path.dirname(os.path.abspath(__file__))
        self.ffmpeg_exe  = os.path.normpath(
            os.path.join(current_dir, "..", "bin", "ffmpeg.exe")
        )

        self.current_file      = "None"
        self.current_file_raw  = "None"
        self.status_msg        = "Ready"
        self._proc_annotated   = None
        self._proc_raw         = None
        self._feed_thread      = None
        self._cam_name         = "cam1"

        # Measured FPS tracking — rolling window of frame timestamps
        self._fps_window = deque(maxlen=30)

    def _kill_zombies(self):
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/IM", "ffmpeg.exe", "/T"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    def _measured_fps(self):
        """Return FPS measured from actual pipeline output, capped to hw fps."""
        if len(self._fps_window) < 2:
            return self.fps
        elapsed = self._fps_window[-1] - self._fps_window[0]
        if elapsed <= 0:
            return self.fps
        measured = (len(self._fps_window) - 1) / elapsed
        # Clamp: never exceed hardware fps, never go below 1
        return max(1.0, min(measured, self.fps))

    def _spawn_ffmpeg(self, output_path, width, height):
        """Spawn one FFmpeg process that reads rawvideo from stdin."""
        cmd = [
            self.ffmpeg_exe, "-y",
            "-f",        "rawvideo",
            "-vcodec",   "rawvideo",
            "-pix_fmt",  "bgr24",
            "-s",        f"{width}x{height}",   # actual frame size, not config assumption
            "-r",        str(self.fps),
            "-i",        "pipe:0",
            "-c:v",      "libx264",
            "-preset",   "ultrafast",
            "-tune",     "zerolatency",
            "-pix_fmt",  "yuv420p",
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

    def start(self, cam_name="cam1"):
        if self.recording or self.finalizing:
            return
        self._kill_zombies()
        self._cam_name = cam_name
        self._fps_window.clear()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file     = os.path.join(self.output_dir, f"Argus_Record_{ts}.mp4")
        self.current_file_raw = os.path.join(self.output_dir, f"Argus_Record_{ts}_raw.mp4")

        try:
            # Detect actual frame dimensions from the live feed.
            # Wait up to 3s in case the pipeline hasn't produced a frame yet.
            from core.vision import latest_annotated, latest_raw
            actual_frame = None
            for _ in range(30):
                _ann = latest_annotated.get(cam_name)
                _raw = latest_raw.get(cam_name)
                actual_frame = _ann if _ann is not None else _raw
                if actual_frame is not None:
                    break
                time.sleep(0.1)

            if actual_frame is not None:
                h, w = actual_frame.shape[:2]
            else:
                w, h = FRAME_WIDTH, FRAME_HEIGHT
                print(f"[WARN] No frame available for {cam_name} — falling back to config size {w}x{h}")
            print(f"[INFO] Recorder using frame size: {w}x{h} for {cam_name}")

            self._proc_annotated = self._spawn_ffmpeg(self.current_file,     w, h)
            self._proc_raw       = self._spawn_ffmpeg(self.current_file_raw, w, h)

            time.sleep(0.5)
            if self._proc_annotated.poll() is not None or self._proc_raw.poll() is not None:
                self.status_msg = "Failed to start FFmpeg"
                self._cleanup()
                return

            self.recording  = True
            self.status_msg = "Recording..."

            self._feed_thread = threading.Thread(target=self._feed_frames, daemon=True)
            self._feed_thread.start()

        except Exception as e:
            import traceback
            print(f"[ERROR] Recorder failed to start: {e}")
            traceback.print_exc()
            self.status_msg = f"Error: {e}"

    def _feed_frames(self):
        """Push annotated + raw frames into their respective ffmpeg processes."""
        from core.vision import latest_annotated, latest_raw

        last_annotated_id = None
        last_raw_id       = None
        interval          = 1.0 / self.fps
        frames_written    = 0

        while self.recording:
            t_start = time.perf_counter()

            annotated = latest_annotated.get(self._cam_name)
            raw       = latest_raw.get(self._cam_name)

            if annotated is not None and id(annotated) != last_annotated_id:
                last_annotated_id = id(annotated)
                try:
                    self._proc_annotated.stdin.write(annotated.tobytes())
                    self._fps_window.append(t_start)
                    interval = 1.0 / self._measured_fps()
                    frames_written += 1
                    if frames_written == 1:
                        print(f"[INFO] Recorder writing frames: shape={annotated.shape}")
                except Exception as e:
                    print(f"[ERROR] Recorder annotated write failed: {e}")
                    break

            if raw is not None and id(raw) != last_raw_id:
                last_raw_id = id(raw)
                try:
                    self._proc_raw.stdin.write(raw.tobytes())
                except Exception as e:
                    print(f"[ERROR] Recorder raw write failed: {e}")
                    break

            elapsed = time.perf_counter() - t_start
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        print(f"[INFO] Recorder _feed_frames exited. Total frames written: {frames_written}")

    def stop(self):
        if not self.recording:
            return

        self.recording  = False
        self.finalizing = True
        self.status_msg = "Finalizing..."

        def finalize():
            if self._feed_thread and self._feed_thread.is_alive():
                self._feed_thread.join(timeout=3)

            for proc in (self._proc_annotated, self._proc_raw):
                if proc:
                    try:
                        proc.stdin.close()
                        proc.wait(timeout=15)
                    except Exception:
                        proc.kill()

            self._proc_annotated = None
            self._proc_raw       = None
            self._cleanup()
            self.finalizing = False

        threading.Thread(target=finalize).start()

    def _cleanup(self):
        time.sleep(0.5)
        saved = []
        for path in (self.current_file, self.current_file_raw):
            if path and os.path.exists(path):
                size = os.path.getsize(path)
                print(f"[INFO] Recorder file: {os.path.basename(path)} | size: {size} bytes")
                if size < 5000:
                    print(f"[WARN] Recorder deleting empty file: {os.path.basename(path)}")
                    os.remove(path)
                else:
                    saved.append(os.path.basename(path))
            else:
                print(f"[WARN] Recorder file not found: {path}")

        if saved:
            self.status_msg = "Saved: " + ", ".join(saved)
        else:
            self.status_msg = "Recording failed (Empty)"