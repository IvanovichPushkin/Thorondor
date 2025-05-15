from ultralytics import YOLO
import os
import cv2

# Get absolute path to the current script directory (App folder)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Build full path to the best.pt model inside runs/aidetection/weights
model_path = os.path.join(base_dir, 'runs', 'aidetection', 'weights', 'best.pt')

model = YOLO(model_path)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.predict(source=frame, imgsz=640, conf=0.25, stream=True)

    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                # Get coordinates and confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]  # Get class name

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame with boxes
    cv2.imshow("Thorondor - AI Cheating Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
