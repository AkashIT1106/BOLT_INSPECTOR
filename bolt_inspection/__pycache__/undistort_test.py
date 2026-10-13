import cv2
import numpy as np

# Load calibration data
data = np.load("camera_calibration.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    h, w = frame.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted_cropped = undistorted[y:y+rh, x:x+rw]
    else:
        undistorted_cropped = undistorted

    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", undistorted_cropped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()