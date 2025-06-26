import os
import cv2
import numpy as np
from pose_estimator import extract_pose_keypoints
import time

# === Setup ===
labels = ["normal", "looking_away", "head_down", "talking", "phone"]
samples_per_label = 200
data = {"keypoints": [], "labels": []}

# === Webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

for label in labels:
    print(f"\n🎬 Recording: {label.upper()} in 3 seconds...")
    time.sleep(3)  # Give bro time to pose

    count = 0
    while count < samples_per_label:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_pose_keypoints(frame, draw=True)

        if keypoints is not None:
            data["keypoints"].append(keypoints)
            data["labels"].append(label)
            count += 1
            print(f"[{label}] Sample {count}/{samples_per_label}")

        # Display recording status
        cv2.putText(frame, f"Label: {label}  Sample: {count}/{samples_per_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Pose Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# === Save dataset ===
cap.release()
cv2.destroyAllWindows()

os.makedirs("dataset", exist_ok=True)
np.save("dataset/pose_sequences.npy", data)
print("✅ Recording complete. Saved to dataset/pose_sequences.npy")
