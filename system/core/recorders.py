import atexit
from core.record_video import VideoRecorder
from core.record_logs import LogRecorder

def init_recorders(fps=15):
    recorder = VideoRecorder(fps=fps)
    # directory_set stays True — VideoRecorder already defaults to ./recordings/
    # which is created on init. No need for a popup to unlock recording.

    log_recorder = LogRecorder()

    atexit.register(recorder.stop)
    atexit.register(log_recorder.stop)

    return recorder, log_recorder