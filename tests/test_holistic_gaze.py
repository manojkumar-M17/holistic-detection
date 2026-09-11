import cv2

from modules.holistic import HolisticDetector


def main():

    print("=" * 50)
    print("HOLISTIC + GAZE INTEGRATION TEST")
    print("=" * 50)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("[ERROR] Camera could not be opened.")

        return

    detector = HolisticDetector()

    print("[INFO] Camera started.")
    print("[INFO] Press Q to exit.")

    while True:

        ret, frame = camera.read()

        if not ret:

            print("[ERROR] Cannot read camera frame.")

            break

        height, width = frame.shape[:2]

        # Temporary full-frame student box
        bbox = (
            0,
            0,
            width,
            height
        )

        result = detector.process_student(

            frame=frame,

            bbox=bbox,

            student_id=1
        )

        if result:

            gaze = result["gaze"]

            direction = gaze["direction"]

            event = gaze["event"]

            cv2.putText(
                frame,
                f"Gaze: {direction}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Yaw: {gaze['yaw']}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Pitch: {gaze['pitch']}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if event:

                print(
                    f"[EVENT] {event}"
                )

                cv2.putText(
                    frame,
                    event,
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

        cv2.imshow(
            "Holistic Gaze Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    camera.release()

    detector.release_all()

    cv2.destroyAllWindows()

    print("[INFO] Test completed.")


if __name__ == "__main__":

    main()