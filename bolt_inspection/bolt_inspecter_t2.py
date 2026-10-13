import cv2
import numpy as np
from ultralytics import YOLO
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d  # <--- ADD THIS LINE

class MasterBoltInspector:
    def __init__(self, camera_index=0, slice_width=1):
        # 1. Initialize camera & Lock Settings to prevent motion blur
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam.")
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 255)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        
        # 2. Load YOLO
        self.yolo_model = YOLO(r"C:/Users/scien/Desktop/SSG PROJECTS/RANE/bolt.pt")        
        self.slice_width = slice_width 
        self.scanned_slices = []
        self.scanning = False
        
        # --- CALIBRATION CONSTANT ---
        self.mm_per_pixel = 0.271639
        # ----------------------------
        
        # 3. Settings
        self.grid_size = 20           
        self.std_dev_multiplier = 2.0 
        
    def get_rois(self, frame):
        """Returns TWO ROIs: A full one for 2D measurement, and a cropped one for 360 scanning."""
        results = self.yolo_model.predict(frame, verbose=False, conf=0.3)
        h, w = frame.shape[:2]
        
        # Fallbacks
        full_roi = frame[int(h*0.1):int(h*0.9), int(w*0.3):int(w*0.7)]
        scan_roi = frame[int(h*0.15):int(h*0.65), int(w*0.48):int(w*0.52)]
        
        if len(results[0].boxes) > 0:
            largest_box = max(results[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1]))
            x1, y1, x2, y2 = map(int, largest_box.xyxy[0].tolist())
            full_roi = frame[y1:y2, x1:x2]
            
            # The Scan ROI stays tightly centered on the threads, ignoring the head/base
            scan_y2 = y1 + int((y2-y1) * 0.7) 
            scan_x1 = x1 + int((x2-x1) * 0.45)
            scan_x2 = x1 + int((x2-x1) * 0.55)
            scan_roi = frame[y1:scan_y2, scan_x1:scan_x2]
            
        return full_roi, scan_roi

    def measure_2d_dimensions(self, full_roi):
        """Analyzes the 2D silhouette to find bolt length and head length."""
        gray = cv2.cvtColor(full_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        total_length_px = thresh.shape[0]
        widths = np.sum(thresh == 255, axis=1) 
        
        head_length_px = 0
        if len(widths) > 0:
            avg_stem_width = np.mean(widths[:int(total_length_px*0.5)])
            for y in range(total_length_px-1, -1, -1):
                if widths[y] > avg_stem_width * 1.5: 
                    head_length_px += 1
                elif head_length_px > 5: 
                    break
        
        # Convert pixels to real-world millimeters
        total_length_mm = total_length_px * self.mm_per_pixel
        head_length_mm = head_length_px * self.mm_per_pixel
                    
        return total_length_mm, head_length_mm

    def extract_center_slice(self, roi):
        h, w = roi.shape[:2]
        center_x = w // 2
        if self.slice_width == 1:
            return roi[:, center_x:center_x+1]
        half_slice = self.slice_width // 2
        return roi[:, max(0, center_x - half_slice):min(w, center_x + half_slice)]

    def analyze_360_scan(self, stitched_img):
        """Runs all 4 defect modules and generates a single Dashboard Report window."""
        output_img = stitched_img.copy()
        h, w = stitched_img.shape[:2]
        defect_list = []
        
        # --- 1. PREP ---
        gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Create a color version of the Canny map so we can draw colored lines on it
        canny_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
# --- 2. CANNY GEOMETRY (Pitch, Count) ---
        vertical_profile = np.sum(edges, axis=1)
        
        # We find ALL the lines (top and bottom edges)
        # Using dynamic prominence to ensure we still catch your faint/broken lines
        max_signal = np.max(vertical_profile) if len(vertical_profile) > 0 else 1
        dynamic_prom = max_signal * 0.15 
        
        peaks, _ = find_peaks(vertical_profile, distance=5, prominence=dynamic_prom)
        
        # --- YOUR RULE: 2 lines = 1 thread ---
        total_lines = len(peaks)
        no_of_threads = total_lines // 2  # Integer division by 2
        
        # Since the gap between two lines is only half a thread, 
        # we multiply the average gap by 2 to get the true thread pitch!
        pixel_pitch = (np.mean(np.diff(peaks)) * 2) if total_lines > 1 else 0
        
        # Convert pixel pitch to real-world millimeters
        mm_pitch = pixel_pitch * self.mm_per_pixel

        # VISUALLY DRAW THE THREADS ON THE CANNY VIEW
        for peak in peaks:
            cv2.line(canny_color, (0, peak), (w, peak), (0, 255, 0), 1) # Draw green line on each thread
            
        cv2.putText(canny_color, f"Threads Counted: {no_of_threads}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(canny_color, f"Measured Pitch: {mm_pitch:.2f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # --- 3. MISTHREAD DETECTION ---
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=int(w*0.3), maxLineGap=10)
        misthread_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                if 5 < angle < 175: 
                    cv2.line(output_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    misthread_count += 1
        if misthread_count > 0:
            defect_list.append(f"Cross-thread ({misthread_count} instances) [BLUE LINES]")

        # --- 4. CONTAMINANTS (Rust & Grease) ---
        hsv = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2HSV)
        rust_mask = cv2.inRange(hsv, np.array([10, 100, 50]), np.array([25, 255, 255]))
        grease_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 40]))
        
        output_img[rust_mask > 0] = [0, 165, 255]   # Orange for Rust
        output_img[grease_mask > 0] = [255, 0, 255] # Purple for Grease
        
        if np.sum(rust_mask) > 500: defect_list.append("Surface Rust Detected [BROWN HUE]")
        if np.sum(grease_mask) > 500: defect_list.append("Grease/Contaminants Detected [BLACK HUE]")

        # --- 5. EDGE DENSITY (Missing Threads) ---
        densities = []
        for y in range(0, h, self.grid_size):
            for x in range(0, w, self.grid_size):
                cell = edges[y:y+self.grid_size, x:x+self.grid_size]
                densities.append((x, y, np.count_nonzero(cell)))
                
        density_values = [d[2] for d in densities]
        if density_values:
            mean_density, std_density = np.mean(density_values), np.std(density_values)
            upper_bound = mean_density + (self.std_dev_multiplier * std_density)
            lower_bound = max(0, mean_density - (self.std_dev_multiplier * std_density))
            
            missing_thread_boxes = 0
            for x, y, density in densities:
                if density > upper_bound or density < lower_bound:
                    cv2.rectangle(output_img, (x, y), (x+self.grid_size, y+self.grid_size), (0, 0, 255), 2)
                    missing_thread_boxes += 1
            if missing_thread_boxes > 0:
                defect_list.append("Edge Anomaly [RED BOXES]")

        # --- 6. BUILD UI DASHBOARD (One Window Result) ---
        target_width = max(w, 600)
        dashboard_height = 200
        dashboard = np.zeros((dashboard_height, target_width, 3), dtype=np.uint8)
        
        if len(defect_list) == 0:
            cv2.putText(dashboard, "FINAL STATUS: BOLT OK", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(dashboard, f"Measurements: {no_of_threads} Threads | Pitch: {mm_pitch:.2f} mm", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(dashboard, "No surface anomalies or cross-threads detected.", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        else:
            cv2.putText(dashboard, "FINAL STATUS: DEFECTS FOUND", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            y_pos = 90
            for idx, defect in enumerate(defect_list):
                cv2.putText(dashboard, f"{idx+1}. {defect}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                y_pos += 30

        # Resize output image and canny map to match dashboard width seamlessly
        if w != target_width:
            aspect_ratio = h / w
            new_h = int(target_width * aspect_ratio)
            output_img = cv2.resize(output_img, (target_width, new_h))
            canny_color = cv2.resize(canny_color, (target_width, new_h))
            
        # Stack Dashboard on top of Image, on top of Canny Map
        final_report_img = np.vstack((dashboard, output_img, canny_color))
        return final_report_img

    def run(self):
        print("Starting Master Vision System...")
        print("Controls: 'Space' to start/stop scanning, 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
                
            display_frame = frame.copy()
            full_roi, scan_roi = self.get_rois(frame)
            
            # --- 2D Measurement (Live) - Now displaying in Real-World MM ---
            total_len_mm, head_len_mm = self.measure_2d_dimensions(full_roi)
            cv2.putText(display_frame, f"2D Bolt Length: {total_len_mm:.1f} mm", (10, h:=frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"2D Head Length: {head_len_mm:.1f} mm", (10, h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Laser Guide
            h, w = frame.shape[:2]
            cv2.line(display_frame, (w//2, int(h*0.15)), (w//2, int(h*0.65)), (0, 255, 0), 2)
            cv2.putText(display_frame, "Align bolt with GREEN LINE. Press SPACE to scan.", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if self.scanning:
                cv2.putText(display_frame, "SCANNING... Rotate steadily.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                slice_img = self.extract_center_slice(scan_roi)
                slice_standardized = cv2.resize(slice_img, (self.slice_width, 400), interpolation=cv2.INTER_LINEAR)
                self.scanned_slices.append(slice_standardized)
                
            cv2.imshow("Live Target Feed", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord(' '):
                self.scanning = not self.scanning
                if not self.scanning and len(self.scanned_slices) > 0:
                    stitched_strip = np.hstack(self.scanned_slices)
                    final_report = self.analyze_360_scan(stitched_strip)
                    cv2.imshow("Final Inspection Report", final_report)
                    self.scanned_slices = [] 

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    inspector = MasterBoltInspector(camera_index=0, slice_width=1)
    inspector.run()