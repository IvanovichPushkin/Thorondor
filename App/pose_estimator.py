import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_pose_keypoints(frame, draw=False):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_model.process(frame_rgb)

    if result.pose_landmarks:
        if draw:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark
        keypoints = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
        return keypoints[:99]
    return None
