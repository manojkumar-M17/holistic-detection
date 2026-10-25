import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from modules.visualization import Visualization


def run_visualization_demo():
    camera = cv2.VideoCapture(0)
    visual = Visualization()

    if not camera.isOpened():
        print("[VISUALIZATION] Camera 0 not available. Exiting demo.")
        return

    print("[VISUALIZATION] Camera opened. Press 'q' to quit.")
    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                break

            visual.draw_student_box(
                frame,
                (100, 100, 350, 400),
                1,
                82,
                "HIGH"
            )

            visual.draw_event(
                frame,
                (100, 100, 350, 400),
                "CELL PHONE"
            )

            visual.draw_fps(frame, 30)

            cv2.imshow("Visualization", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_visualization_demo()
