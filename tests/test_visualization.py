"""
Tests for Visualization Module
-------------------------------
Tests for the Visualization class — color mapping and drawing methods.
cv2 drawing calls are verified via mocking to avoid needing a display.

Run:
    python -m pytest tests/test_visualization.py -v
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.visualization import Visualization


class TestVisualizationInit(unittest.TestCase):

    def test_init_creates_instance(self):
        vis = Visualization()
        self.assertIsNotNone(vis)
        self.assertIsNotNone(vis.mp_draw)
        self.assertIsNotNone(vis.mp_styles)
        self.assertIsNotNone(vis.mp_holistic)


class TestGetColor(unittest.TestCase):

    def setUp(self):
        self.vis = Visualization()

    def test_normal_color(self):
        color = self.vis.get_color("NORMAL")
        self.assertEqual(color, (0, 255, 0))

    def test_low_color(self):
        color = self.vis.get_color("LOW")
        self.assertEqual(color, (0, 255, 255))

    def test_medium_color(self):
        color = self.vis.get_color("MEDIUM")
        self.assertEqual(color, (0, 165, 255))

    def test_high_color(self):
        color = self.vis.get_color("HIGH")
        self.assertEqual(color, (0, 0, 255))

    def test_critical_color(self):
        color = self.vis.get_color("CRITICAL")
        self.assertEqual(color, (128, 0, 255))

    def test_unknown_level_returns_white(self):
        color = self.vis.get_color("UNKNOWN")
        self.assertEqual(color, (255, 255, 255))

    def test_all_five_levels_return_different_colors(self):
        levels = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        colors = [self.vis.get_color(l) for l in levels]
        self.assertEqual(len(set(colors)), 5)


class TestDrawStudentBox(unittest.TestCase):

    def setUp(self):
        self.vis = Visualization()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    @patch("cv2.rectangle")
    @patch("cv2.putText")
    def test_draw_student_box_calls_rectangle(self, mock_puttext, mock_rect):
        self.vis.draw_student_box(self.frame, (10, 20, 200, 300), 1, 45.0, "MEDIUM")
        mock_rect.assert_called_once()

    @patch("cv2.rectangle")
    @patch("cv2.putText")
    def test_draw_student_box_calls_puttext_twice(self, mock_puttext, mock_rect):
        self.vis.draw_student_box(self.frame, (10, 20, 200, 300), 1, 45.0, "MEDIUM")
        self.assertEqual(mock_puttext.call_count, 2)

    @patch("cv2.rectangle")
    @patch("cv2.putText")
    def test_draw_student_box_uses_correct_color(self, mock_puttext, mock_rect):
        self.vis.draw_student_box(self.frame, (10, 20, 200, 300), 1, 45.0, "HIGH")
        call_color = mock_rect.call_args[0][3]
        self.assertEqual(call_color, (0, 0, 255))

    @patch("cv2.rectangle")
    @patch("cv2.putText")
    def test_draw_student_box_shows_student_id(self, mock_puttext, mock_rect):
        self.vis.draw_student_box(self.frame, (10, 20, 200, 300), 7, 20.0, "LOW")
        all_text = [call[0][1] for call in mock_puttext.call_args_list]
        self.assertTrue(any("7" in t for t in all_text))

    @patch("cv2.rectangle")
    @patch("cv2.putText")
    def test_draw_student_box_shows_risk_score(self, mock_puttext, mock_rect):
        self.vis.draw_student_box(self.frame, (10, 20, 200, 300), 1, 33.5, "LOW")
        all_text = [call[0][1] for call in mock_puttext.call_args_list]
        self.assertTrue(any("33.5" in t for t in all_text))


class TestDrawEvent(unittest.TestCase):

    def setUp(self):
        self.vis = Visualization()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    @patch("cv2.putText")
    def test_draw_event_calls_puttext(self, mock_puttext):
        self.vis.draw_event(self.frame, (10, 20, 200, 300), "CELL_PHONE")
        mock_puttext.assert_called_once()

    @patch("cv2.putText")
    def test_draw_event_uses_red_color(self, mock_puttext):
        self.vis.draw_event(self.frame, (10, 20, 200, 300), "STANDING")
        call_color = mock_puttext.call_args[0][5]
        self.assertEqual(call_color, (0, 0, 255))

    @patch("cv2.putText")
    def test_draw_event_passes_event_text(self, mock_puttext):
        self.vis.draw_event(self.frame, (10, 20, 200, 300), "LOOK_LEFT")
        text_arg = mock_puttext.call_args[0][1]
        self.assertEqual(text_arg, "LOOK_LEFT")


class TestDrawFps(unittest.TestCase):

    def setUp(self):
        self.vis = Visualization()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    @patch("cv2.putText")
    def test_draw_fps_calls_puttext(self, mock_puttext):
        self.vis.draw_fps(self.frame, 29.5)
        mock_puttext.assert_called_once()

    @patch("cv2.putText")
    def test_draw_fps_shows_fps_value(self, mock_puttext):
        self.vis.draw_fps(self.frame, 30.0)
        text_arg = mock_puttext.call_args[0][1]
        self.assertIn("30.0", text_arg)


class TestDrawHolistic(unittest.TestCase):

    def setUp(self):
        self.vis = Visualization()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def _make_results(self, face=False, pose=False, left_hand=False, right_hand=False):
        results = MagicMock()
        results.pose_landmarks = MagicMock() if pose else None
        results.left_hand_landmarks = MagicMock() if left_hand else None
        results.right_hand_landmarks = MagicMock() if right_hand else None
        results.face_landmarks = MagicMock() if face else None
        return results

    def test_draw_holistic_no_landmarks_no_crash(self):
        results = self._make_results()
        try:
            self.vis.draw_holistic(self.frame, results)
        except Exception as e:
            self.fail(f"draw_holistic raised an exception with empty results: {e}")

    def test_draw_holistic_calls_draw_landmarks_for_pose(self):
        results = self._make_results(pose=True)
        with patch.object(self.vis.mp_draw, "draw_landmarks") as mock_draw:
            self.vis.draw_holistic(self.frame, results)
            self.assertGreaterEqual(mock_draw.call_count, 1)

    def test_draw_holistic_calls_draw_landmarks_for_face(self):
        results = self._make_results(face=True)
        with patch.object(self.vis.mp_draw, "draw_landmarks") as mock_draw:
            self.vis.draw_holistic(self.frame, results)
            self.assertGreaterEqual(mock_draw.call_count, 1)

    def test_draw_holistic_calls_all_for_full_results(self):
        results = self._make_results(face=True, pose=True, left_hand=True, right_hand=True)
        with patch.object(self.vis.mp_draw, "draw_landmarks") as mock_draw:
            self.vis.draw_holistic(self.frame, results)
            self.assertEqual(mock_draw.call_count, 4)


if __name__ == "__main__":
    unittest.main()
