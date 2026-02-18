import atexit
from core.record_video import VideoRecorder
from core.record_logs import LogRecorder

def init_recorders(fps=15):
    # Initialize Video Recorder
    recorder = VideoRecorder(fps=fps)
    recorder.directory_set = False
    
    # Initialize Log Recorder
    log_recorder = LogRecorder()
    
    # Register clean shutdown to prevent corrupted video files
    atexit.register(recorder.stop)
    atexit.register(log_recorder.stop)
    
    return recorder, log_recorder