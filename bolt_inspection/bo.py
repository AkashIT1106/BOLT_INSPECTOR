import cv2
import numpy as np
from ultralytics import YOLO

class BoltInspector:
    def __init__(self, camera_index=1):
        # 1. Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam.")
            
        # Camera Settings for Sharpness
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 40) 
        
        # 2. Load YOLO model
        self.yolo_model = YOLO(r".venv\Model_training\bolt.pt") 
        
        # 3. Storage for the 4 captures
        self.captured_frames = []
        self.quadrants = ["0° (Front)", "90° (Right)", "180° (Back)", "270° (Left)"]
        
    def get_bolt_roi(self, frame):
        """Finds the bolt and crops a consistent window."""
        results = self.yolo_model.predict(frame, verbose=False, conf=0.3)
        h, w = frame.shape[:2]
        
        if len(results[0].boxes) > 0:
            box = max(results[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0]))
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            roi = frame[y1:y2, x1:x2]
            return cv2.resize(roi, (250, 500)) 
        
        fallback = frame[int(h*0.1):int(h*0.9), int(w*0.4):int(w*0.6)]
        return cv2.resize(fallback, (250, 500))

    def analyze_threads(self, roi):
        """Analyzes threads within a strict bounding box to ignore background."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        h, w = edges.shape
        defect_overlay = roi.copy()
        defects_found = 0
        
        # GRID ANALYSIS SETTINGS
        grid_h, grid_w = 25, 25 
        
        # --- THE FIX: Define a strict Bounding Box for the threads ---
        # Horizontal: Center 30% of the image (tighter than before)
        bolt_start_x = int(w * 0.35)
        bolt_end_x = int(w * 0.65)
        
        # Vertical: Ignore the top 20% (empty air) and bottom 35% (nut/base)
        bolt_start_y = int(h * 0.20)
        bolt_end_y = int(h * 0.65)

        # Only loop through the specific grid area
        for y in range(bolt_start_y, bolt_end_y - grid_h, grid_h):
            for x in range(bolt_start_x, bolt_end_x - grid_w, grid_w):
                cell = edges[y:y+grid_h, x:x+grid_w]
                density = np.count_nonzero(cell)
                
                # If density is too low inside the thread zone, flag it
                if density < 15: 
                    cv2.rectangle(defect_overlay, (x, y), (x+grid_w, y+grid_h), (0, 0, 255), 1)
                    defects_found += 1

        # Determine overall status
        status = "FAIL" if defects_found > 3 else "PASS"
        color = (0, 0, 255) if status == "FAIL" else (0, 255, 0)
        
        cv2.putText(defect_overlay, f"STATUS: {status}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Draw a Green Bounding Box on the Canny map to show the exact inspection zone
        cv2.rectangle(edge_bgr, (bolt_start_x, bolt_start_y), (bolt_end_x, bolt_end_y), (0, 255, 0), 1)
        
        return np.hstack((defect_overlay, edge_bgr))

    def run(self):
        print("--- BOLT INSPECTOR: 4-FRAME MODE ---")
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            display_frame = frame.copy()
            current_step = len(self.captured_frames)
            
            if current_step < 4:
                cv2.putText(display_frame, f"TARGET: {self.quadrants[current_step]}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, "Press 'C' to Capture", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Live Feed", display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and current_step < 4:
                roi = self.get_bolt_roi(frame)
                processed = self.analyze_threads(roi)
                self.captured_frames.append(processed)
                
                if len(self.captured_frames) == 4:
                    top = np.hstack((self.captured_frames[0], self.captured_frames[1]))
                    bottom = np.hstack((self.captured_frames[2], self.captured_frames[3]))
                    report = np.vstack((top, bottom))
                    report_scaled = cv2.resize(report, (1000, 800))
                    cv2.imshow("Inspection Report", report_scaled)

            if key == ord('r'):
                self.captured_frames = []
                print("Resetting...")

            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    inspector = BoltInspector()
    inspector.run()