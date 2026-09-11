from modules.behaviour import analyze_student_behaviour


def main():

    print("=" * 55)
    print("GAZE + BEHAVIOUR INTEGRATION TEST")
    print("=" * 55)

    # Simulated output from holistic.py
    landmark_data = {

        "crop_dims": (
            0,
            0,
            640,
            480
        ),

        "face_landmarks": None,
        "pose_landmarks": None,
        "left_hand_landmarks": None,
        "right_hand_landmarks": None,

        "gaze": {

            "yaw": 25.5,
            "pitch": 2.0,
            "roll": 1.0,

            "confidence": 1.0,

            "direction": "LOOKING_RIGHT",

            "event": "LOOKING_RIGHT"
        }
    }

    result = analyze_student_behaviour(
        landmark_data
    )

    print("\nBEHAVIOUR RESULT:\n")

    for key, value in result.items():

        print(f"{key}: {value}")

    print("\n" + "=" * 55)
    print("TEST COMPLETED")
    print("=" * 55)


if __name__ == "__main__":

    main()