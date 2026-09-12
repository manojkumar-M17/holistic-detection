import cv2
from modules.visualization import Visualization


def run_visualization_demo():
    camera = cv2.VideoCapture(0)
    visual = Visualization()

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