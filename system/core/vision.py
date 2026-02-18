import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from core.detections.pose import process as pose_process
from core.detections.desk import process as desk_process
from core.detections.object import process as obj_process

latest_annotated = {}
# Parallel processing pool
executor = ThreadPoolExecutor(max_workers=3)

def run_ai_pipeline(cam_name, frame):
    try:
        # Run Pose and Objects in parallel
        future_pose = executor.submit(pose_process, frame.copy(), cam_name)
        future_obj  = executor.submit(obj_process, frame.copy(), cam_name)

        # Wait for results
        annotated_frame, person_boxes = future_pose.result()
        obj_frame, _ = future_obj.result()

        # Run Desk (depends on Pose boxes)
        final_frame = desk_process(annotated_frame, cam_name, person_boxes=person_boxes)

        # Merge object annotations
        final_frame = cv2.addWeighted(final_frame, 0.8, obj_frame, 0.2, 0)
        
        latest_annotated[cam_name] = final_frame
        return final_frame
    except Exception as e:
        print(f"AI Error: {e}")
        return frame

def generate_frames(cam_name, frames_override=None, recorder=None):
    last_frame_id = None
    
    while True:
        try:
            frame_deque = frames_override.get(cam_name)
            if not frame_deque or len(frame_deque) == 0:
                time.sleep(0.01)
                continue
            
            frame = frame_deque[0]
            
            # Optimization: Only process if the camera has provided a NEW frame
            current_frame_id = id(frame)
            if current_frame_id == last_frame_id:
                time.sleep(0.001)
                continue
            last_frame_id = current_frame_id
                
        except (IndexError, KeyError):
            time.sleep(0.01)
            continue

        processed = run_ai_pipeline(cam_name, frame.copy())
        
        ret, buffer = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')