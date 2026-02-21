import os
import subprocess
import threading
import platform
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import time

class VideoRecorder:
    def __init__(self, fps=15.0):
        self.fps = fps
        self.process = None
        self.recording = False
        self.finalizing = False
        self.directory_set = True
        self.output_dir = os.path.join(os.getcwd(), "recordings")  # ← relative to project
        os.makedirs(self.output_dir, exist_ok=True)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.ffmpeg_exe = os.path.normpath(os.path.join(current_dir, "..", "bin", "ffmpeg.exe"))

        self.current_file = "None"
        self.status_msg = "Ready"

    def _kill_zombies(self):
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", "ffmpeg.exe", "/T"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def start(self):
        if self.recording or self.finalizing: return
        self._kill_zombies()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = os.path.join(self.output_dir, f"Argus_Record_{ts}.mp4")

        cmd = [
            f'"{self.ffmpeg_exe}"', '-y',
            '-f', 'gdigrab',
            '-framerate', str(self.fps),
            '-i', 'desktop',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            f'"{self.current_file}"'
        ]

        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            self.process = subprocess.Popen(
                " ".join(cmd),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, startupinfo=si, shell=True
            )

            time.sleep(1)
            if self.process.poll() is not None:
                self.status_msg = "Failed to start FFmpeg"
                self._cleanup()
            else:
                self.recording = True
                self.status_msg = "Recording..."
        except Exception as e:
            self.status_msg = f"Error: {e}"

    def stop(self):
        if not self.recording or not self.process: return

        self.recording = False
        self.finalizing = True
        self.status_msg = "Finalizing..."

        def finalize():
            try:
                self.process.stdin.write('q')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except:
                self.process.kill()

            self._cleanup()
            self.finalizing = False
            self.process = None

        threading.Thread(target=finalize).start()

    def _cleanup(self):
        time.sleep(1)
        if self.current_file and os.path.exists(self.current_file):
            if os.path.getsize(self.current_file) < 5000:
                os.remove(self.current_file)
                self.status_msg = "Recording failed (Empty)"
            else:
                self.status_msg = "Success: " + os.path.basename(self.current_file)

    # def set_directory_popup(self):
    #     root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    #     path = filedialog.askdirectory(); root.destroy()
    #     if path:
    #         self.output_dir = os.path.normpath(path)
    #         self.directory_set = True
    #         return path
    #     return None