from ultralytics import YOLO
import cv2
from pathlib import Path

# Load your trained model
path_to_model = Path('C:/Users/brand/Documents/College/2025/MARTIN/SYBIL/runs/final/yolov8m_best_full_retrain/weights/best.pt')
model = YOLO(path_to_model)  

# Open laptop camera (device 0) if only one camera is available
# Open external camera (device 0) if hooked up
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference; set confidence threshold at 0.531 for F1 recommended
    # set ct at 0.01 to capture all detections
    results = model(frame, conf=0.531)  # Adjust confidence threshold as needed

    # Annotate frame with detections
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("Litter Detection", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
