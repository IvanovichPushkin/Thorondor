import atexit
from core.record_video import VideoRecorder
from core.record_logs import LogRecorder

def init_recorders(fps=15):
    recorder = VideoRecorder(fps=fps)
    recorder.directory_set = False

    log_recorder = LogRecorder()

    atexit.register(recorder.stop)
    atexit.register(log_recorder.stop)

    return recorder, log_recorder