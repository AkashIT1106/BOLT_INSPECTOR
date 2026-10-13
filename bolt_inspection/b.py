import cv2
from ultralytics import YOLO

# =========================
# SETTINGS
# =========================
MODEL_PATH = r"C:\Users\scien\Desktop\RANE\.venv\Model_training\bolt1\best.pt"     # change if needed
CAMERA_INDEX = 1           # try 1 if DroidCam/external camera
CONFIDENCE = 0.2

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# OPEN CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Could not open camera")
    exit()

print("Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    # Run inference
    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        verbose=False
    )

    annotated_frame = frame.copy()

    # Draw detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            class_name = model.names[cls] if hasattr(model, "names") else str(cls)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"{class_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.imshow("YOLO Bolt Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()