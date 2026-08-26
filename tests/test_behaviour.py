"""
Tests for Behaviour Analysis
-----------------------------
Tests for analyze_student_behaviour() and estimate_head_pose()
from modules.behaviour.

Run:
    python -m pytest tests/test_behaviour.py -v
"""

import sys
import os
import unittest
from unittest.mock import MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.behaviour import analyze_student_behaviour, estimate_head_pose


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_landmark(x=0.5, y=0.5, z=0.0, visibility=1.0):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    lm.visibility = visibility
    return lm


def make_face_landmarks(num=468):
    fl = MagicMock()
    fl.landmark = [make_landmark() for _ in range(num)]
    return fl


def make_pose_landmarks_sitting():
    """Shoulder y around 0.4 → NOT standing (threshold < 0.22)."""
    pl = MagicMock()
    pl.landmark = [make_landmark() for _ in range(33)]
    pl.landmark[11] = make_landmark(x=0.3, y=0.4)  # left shoulder
    pl.landmark[12] = make_landmark(x=0.7, y=0.4)  # right shoulder
    return pl


def make_pose_landmarks_standing():
    """Shoulder y around 0.1 → IS standing (< 0.22)."""
    pl = MagicMock()
    pl.landmark = [make_landmark() for _ in range(33)]
    pl.landmark[11] = make_landmark(x=0.3, y=0.1)
    pl.landmark[12] = make_landmark(x=0.7, y=0.1)
    return pl


def make_landmark_data(
    face=True,
    pose=True,
    left_hand=False,
    right_hand=False,
    standing=False,
    crop_dims=(0, 0, 320, 480)
):
    data = {
        "crop_dims": crop_dims,
        "face_landmarks": make_face_landmarks() if face else None,
        "pose_landmarks": (
            make_pose_landmarks_standing() if standing
            else make_pose_landmarks_sitting()
        ) if pose else None,
        "left_hand_landmarks": MagicMock() if left_hand else None,
        "right_hand_landmarks": MagicMock() if right_hand else None,
    }
    if left_hand:
        data["left_hand_landmarks"].landmark = [make_landmark(x=0.5, y=0.5) for _ in range(21)]
    if right_hand:
        data["right_hand_landmarks"].landmark = [make_landmark(x=0.5, y=0.5) for _ in range(21)]
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnalyseStudentBehaviourReturnStructure(unittest.TestCase):

    def test_returns_dict(self):
        result = analyze_student_behaviour(make_landmark_data())
        self.assertIsInstance(result, dict)

    def test_contains_all_expected_keys(self):
        result = analyze_student_behaviour(make_landmark_data())
        expected_keys = {
            "yaw", "pitch", "roll", "gaze_line",
            "shoulder_tilt", "hands_detected", "hand_near_face",
            "hand_near_face_val", "is_standing", "standing_val",
            "is_absent", "has_forbidden_object", "forbidden_objects"
        }
        self.assertEqual(set(result.keys()), expected_keys)


class TestAbsence(unittest.TestCase):

    def test_none_landmark_data_marks_absent(self):
        result = analyze_student_behaviour(None)
        self.assertTrue(result["is_absent"])

    def test_no_face_no_pose_marks_absent(self):
        data = make_landmark_data(face=False, pose=False)
        result = analyze_student_behaviour(data)
        self.assertTrue(result["is_absent"])

    def test_with_face_and_pose_not_absent(self):
        result = analyze_student_behaviour(make_landmark_data())
        self.assertFalse(result["is_absent"])


class TestStandingDetection(unittest.TestCase):

    def test_high_shoulders_marks_standing(self):
        data = make_landmark_data(standing=True)
        result = analyze_student_behaviour(data)
        self.assertTrue(result["is_standing"])

    def test_low_shoulders_marks_sitting(self):
        data = make_landmark_data(standing=False)
        result = analyze_student_behaviour(data)
        self.assertFalse(result["is_standing"])


class TestHandDetection(unittest.TestCase):

    def test_no_hands_detected(self):
        data = make_landmark_data(left_hand=False, right_hand=False)
        result = analyze_student_behaviour(data)
        self.assertFalse(result["hands_detected"])

    def test_left_hand_detected(self):
        data = make_landmark_data(left_hand=True)
        result = analyze_student_behaviour(data)
        self.assertTrue(result["hands_detected"])

    def test_right_hand_detected(self):
        data = make_landmark_data(right_hand=True)
        result = analyze_student_behaviour(data)
        self.assertTrue(result["hands_detected"])

    def test_both_hands_detected(self):
        data = make_landmark_data(left_hand=True, right_hand=True)
        result = analyze_student_behaviour(data)
        self.assertTrue(result["hands_detected"])


class TestForbiddenObjects(unittest.TestCase):

    def test_no_objects_no_forbidden(self):
        result = analyze_student_behaviour(make_landmark_data(), correlated_objects=None)
        self.assertFalse(result["has_forbidden_object"])
        self.assertEqual(result["forbidden_objects"], [])

    def test_correlated_objects_flagged(self):
        objects = [{"label": "Cell Phone"}, {"label": "Book"}]
        result = analyze_student_behaviour(make_landmark_data(), correlated_objects=objects)
        self.assertTrue(result["has_forbidden_object"])
        self.assertIn("Cell Phone", result["forbidden_objects"])
        self.assertIn("Book", result["forbidden_objects"])

    def test_forbidden_objects_with_none_landmarks(self):
        objects = [{"label": "Laptop"}]
        result = analyze_student_behaviour(None, correlated_objects=objects)
        self.assertTrue(result["has_forbidden_object"])
        self.assertTrue(result["is_absent"])


class TestHeadPoseDefaults(unittest.TestCase):

    def test_no_face_landmarks_yaw_pitch_roll_are_zero(self):
        data = make_landmark_data(face=False)
        result = analyze_student_behaviour(data)
        self.assertEqual(result["yaw"], 0.0)
        self.assertEqual(result["pitch"], 0.0)
        self.assertEqual(result["roll"], 0.0)

    def test_with_face_landmarks_yaw_is_float(self):
        data = make_landmark_data(face=True)
        result = analyze_student_behaviour(data)
        self.assertIsInstance(result["yaw"], float)
        self.assertIsInstance(result["pitch"], float)
        self.assertIsInstance(result["roll"], float)


class TestEstimateHeadPoseStandalone(unittest.TestCase):

    def test_returns_four_values(self):
        result = estimate_head_pose(None, 320, 240)
        self.assertEqual(len(result), 4)

    def test_returns_zeros_for_none_landmarks(self):
        yaw, pitch, roll, line = estimate_head_pose(None, 320, 240)
        self.assertEqual(yaw, 0.0)
        self.assertEqual(pitch, 0.0)
        self.assertEqual(roll, 0.0)
        self.assertIsNone(line)

    def test_returns_floats_for_valid_landmarks(self):
        face = make_face_landmarks()
        yaw, pitch, roll, line = estimate_head_pose(face, 320, 240)
        self.assertIsInstance(yaw, float)
        self.assertIsInstance(pitch, float)
        self.assertIsInstance(roll, float)


if __name__ == "__main__":
    unittest.main()
