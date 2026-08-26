"""
Tests for HolisticDetector
---------------------------
Tests for HolisticDetector: instance management, processing, and cleanup.
MediaPipe Holistic is mocked to avoid requiring a camera or GPU.

Run:
    python -m pytest tests/test_holistic.py -v
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.holistic import HolisticDetector


class TestHolisticDetectorInit(unittest.TestCase):

    def test_init_creates_empty_instances(self):
        detector = HolisticDetector()
        self.assertEqual(len(detector.instances), 0)

    def test_init_creates_empty_last_active(self):
        detector = HolisticDetector()
        self.assertEqual(len(detector.last_active), 0)

    def test_default_timeout(self):
        detector = HolisticDetector()
        self.assertEqual(detector.timeout, 10.0)


class TestGetOrCreateInstance(unittest.TestCase):

    def setUp(self):
        self.detector = HolisticDetector()

    def test_creates_new_instance_for_new_student(self):
        with patch.object(self.detector.mp_holistic, "Holistic") as MockHolistic:
            MockHolistic.return_value = MagicMock()
            self.detector._get_or_create_instance(1)
            self.assertIn(1, self.detector.instances)

    def test_reuses_existing_instance(self):
        with patch.object(self.detector.mp_holistic, "Holistic") as MockHolistic:
            mock_inst = MagicMock()
            MockHolistic.return_value = mock_inst
            inst1 = self.detector._get_or_create_instance(1)
            inst2 = self.detector._get_or_create_instance(1)
            self.assertIs(inst1, inst2)
            MockHolistic.assert_called_once()

    def test_updates_last_active_timestamp(self):
        with patch.object(self.detector.mp_holistic, "Holistic", return_value=MagicMock()):
            before = time.time()
            self.detector._get_or_create_instance(5)
            after = time.time()
            self.assertGreaterEqual(self.detector.last_active[5], before)
            self.assertLessEqual(self.detector.last_active[5], after)

    def test_cleans_up_stale_instances(self):
        with patch.object(self.detector.mp_holistic, "Holistic", return_value=MagicMock()):
            self.detector._get_or_create_instance(1)
            # Simulate stale instance
            self.detector.last_active[1] = time.time() - 100
            self.detector._get_or_create_instance(99)  # trigger cleanup
            self.assertNotIn(1, self.detector.instances)


class TestProcessStudent(unittest.TestCase):

    def setUp(self):
        self.detector = HolisticDetector()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_returns_none_for_small_bbox(self):
        result = self.detector.process_student(self.frame, (100, 100, 110, 115), 1)
        self.assertIsNone(result)

    def test_returns_none_when_bbox_too_narrow(self):
        result = self.detector.process_student(self.frame, (100, 100, 110, 400), 1)
        self.assertIsNone(result)

    def test_returns_none_when_bbox_too_short(self):
        result = self.detector.process_student(self.frame, (100, 100, 400, 110), 1)
        self.assertIsNone(result)

    def test_returns_dict_for_valid_bbox(self):
        with patch.object(self.detector.mp_holistic, "Holistic") as MockHolistic:
            mock_results = MagicMock()
            mock_results.face_landmarks = None
            mock_results.pose_landmarks = None
            mock_results.left_hand_landmarks = None
            mock_results.right_hand_landmarks = None
            mock_inst = MagicMock()
            mock_inst.process.return_value = mock_results
            MockHolistic.return_value = mock_inst

            result = self.detector.process_student(self.frame, (10, 10, 300, 400), 1)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, dict)

    def test_result_contains_expected_keys(self):
        with patch.object(self.detector.mp_holistic, "Holistic") as MockHolistic:
            mock_results = MagicMock()
            mock_results.face_landmarks = None
            mock_results.pose_landmarks = None
            mock_results.left_hand_landmarks = None
            mock_results.right_hand_landmarks = None
            mock_inst = MagicMock()
            mock_inst.process.return_value = mock_results
            MockHolistic.return_value = mock_inst

            result = self.detector.process_student(self.frame, (10, 10, 300, 400), 1)
            expected_keys = {
                "crop_dims", "face_landmarks", "pose_landmarks",
                "left_hand_landmarks", "right_hand_landmarks"
            }
            self.assertEqual(set(result.keys()), expected_keys)

    def test_bbox_clipped_to_frame_boundaries(self):
        """bbox exceeding frame dimensions should be clipped without error."""
        with patch.object(self.detector.mp_holistic, "Holistic") as MockHolistic:
            mock_results = MagicMock()
            mock_results.face_landmarks = None
            mock_results.pose_landmarks = None
            mock_results.left_hand_landmarks = None
            mock_results.right_hand_landmarks = None
            mock_inst = MagicMock()
            mock_inst.process.return_value = mock_results
            MockHolistic.return_value = mock_inst

            try:
                result = self.detector.process_student(self.frame, (-50, -50, 700, 600), 1)
            except Exception as e:
                self.fail(f"process_student with out-of-bounds bbox raised: {e}")


class TestReleaseAll(unittest.TestCase):

    def setUp(self):
        self.detector = HolisticDetector()

    def test_release_all_clears_instances(self):
        with patch.object(self.detector.mp_holistic, "Holistic") as MockHolistic:
            MockHolistic.return_value = MagicMock()
            self.detector._get_or_create_instance(1)
            self.detector._get_or_create_instance(2)
            self.detector.release_all()
            self.assertEqual(len(self.detector.instances), 0)

    def test_release_all_clears_last_active(self):
        with patch.object(self.detector.mp_holistic, "Holistic") as MockHolistic:
            MockHolistic.return_value = MagicMock()
            self.detector._get_or_create_instance(1)
            self.detector.release_all()
            self.assertEqual(len(self.detector.last_active), 0)

    def test_release_all_calls_close_on_instances(self):
        with patch.object(self.detector.mp_holistic, "Holistic") as MockHolistic:
            mock_inst = MagicMock()
            MockHolistic.return_value = mock_inst
            self.detector._get_or_create_instance(1)
            self.detector.release_all()
            mock_inst.close.assert_called_once()

    def test_release_all_on_empty_is_safe(self):
        try:
            self.detector.release_all()
        except Exception as e:
            self.fail(f"release_all on empty detector raised: {e}")


if __name__ == "__main__":
    unittest.main()
