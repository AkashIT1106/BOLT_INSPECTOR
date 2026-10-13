import cv2
import numpy as np

points = []
image = None

def click_event(event, x, y, flags, param):
    global points, image

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point selected: ({x}, {y})")

        # draw point
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        if len(points) == 2:
            # draw line
            cv2.line(image, points[0], points[1], (255, 0, 0), 2)

            # calculate pixel distance
            p1 = np.array(points[0])
            p2 = np.array(points[1])
            pixel_distance = np.linalg.norm(p1 - p2)

            print(f"Pixel distance: {pixel_distance:.2f} px")

            # input real length
            real_length_mm = float(input("Enter real-world length (mm): "))

            mm_per_pixel = real_length_mm / pixel_distance

            print(f"mm_per_pixel = {mm_per_pixel:.6f}")

            cv2.putText(
                image,
                f"{mm_per_pixel:.6f} mm/px",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Calibration", image)


def main():
    global image, points

    cap = cv2.VideoCapture(1)

    print("Press 's' to capture frame")
    print("Click 2 points on known reference")
    print("Press 'r' to reset")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        cv2.imshow("Live Camera", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            image = frame.copy()
            points = []
            cv2.imshow("Calibration", image)
            cv2.setMouseCallback("Calibration", click_event)

        elif key == ord('r'):
            if image is not None:
                image = image.copy()
                points = []
                cv2.imshow("Calibration", image)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()