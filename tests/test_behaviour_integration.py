from modules.behaviour import analyze_student_behaviour


def print_result(title, result):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")


def test_gaze_behaviour():
    """Test stable gaze integration."""

    landmark_data = {
        "crop_dims": (0, 0, 640, 480),

        "face_landmarks": None,
        "pose_landmarks": None,

        "left_hand_landmarks": None,
        "right_hand_landmarks": None,

        "gaze": {
            "yaw": 25.5,
            "pitch": 2.0,
            "roll": 1.0,
            "direction": "LOOKING_RIGHT",
            "event": "LOOKING_RIGHT",
            "confidence": 1.0
        }
    }

    result = analyze_student_behaviour(landmark_data)

    print_result(
        "TEST 1: STABLE GAZE EVENT",
        result
    )

    assert result["gaze_direction"] == "LOOKING_RIGHT"
    assert result["gaze_event"] == "LOOKING_RIGHT"
    assert result["is_looking_away"] is True


def test_normal_gaze():
    """Test normal forward gaze."""

    landmark_data = {
        "crop_dims": (0, 0, 640, 480),

        "face_landmarks": None,
        "pose_landmarks": None,

        "left_hand_landmarks": None,
        "right_hand_landmarks": None,

        "gaze": {
            "yaw": 2.0,
            "pitch": 1.0,
            "roll": 0.0,
            "direction": "FORWARD",
            "event": None,
            "confidence": 1.0
        }
    }

    result = analyze_student_behaviour(landmark_data)

    print_result(
        "TEST 2: NORMAL GAZE",
        result
    )

    assert result["gaze_direction"] == "FORWARD"
    assert result["gaze_event"] is None


def test_forbidden_object():
    """Test forbidden object detection."""

    landmark_data = {
        "crop_dims": (0, 0, 640, 480),

        "face_landmarks": None,
        "pose_landmarks": None,

        "left_hand_landmarks": None,
        "right_hand_landmarks": None
    }

    correlated_objects = [
        {
            "label": "Cell Phone"
        }
    ]

    result = analyze_student_behaviour(
        landmark_data,
        correlated_objects
    )

    print_result(
        "TEST 3: FORBIDDEN OBJECT",
        result
    )

    assert result["has_forbidden_object"] is True
    assert "Cell Phone" in result["forbidden_objects"]


def main():

    print("\n")
    print("=" * 60)
    print("BEHAVIOUR MODULE INTEGRATION TEST")
    print("=" * 60)

    try:

        test_gaze_behaviour()
        test_normal_gaze()
        test_forbidden_object()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY")
        print("=" * 60)

    except AssertionError as error:

        print("\nTEST FAILED")

        raise error


if __name__ == "__main__":
    main()