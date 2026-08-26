"""
Tests for Detection Module
--------------------------
Tests for StudentDetector and find_student_objects().
YOLO model is mocked to avoid requiring GPU/model weights.

Run:
    python -m pytest tests/test_detection.py -v
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestFindStudentObjects(unittest.TestCase):
    """
    Tests for the pure function find_student_objects() which
    does not require any model — safe to test without mocking.
    """

    def _import(self):
        from modules.detection import find_student_objects
        return find_student_objects

    def test_object_inside_student_box_is_returned(self):
        find_student_objects = self._import()
        student_bbox = (100, 100, 300, 400)
        obj = {"label": "Cell Phone", "bbox": (150, 200, 180, 230)}
        result = find_student_objects(student_bbox, [obj])
        self.assertIn(obj, result)

    def test_object_outside_student_box_not_returned(self):
        find_student_objects = self._import()
        student_bbox = (100, 100, 200, 200)
        obj = {"label": "Book", "bbox": (400, 400, 450, 450)}
        result = find_student_objects(student_bbox, [obj])
        self.assertEqual(result, [])

    def test_object_within_margin_is_included(self):
        find_student_objects = self._import()
        # Object center just outside bbox but within margin=30
        student_bbox = (100, 100, 200, 200)
        # Center at (210, 150) — 10px outside right edge, within margin=30
        obj = {"label": "Cell Phone", "bbox": (205, 145, 215, 155)}
        result = find_student_objects(student_bbox, [obj], margin=30)
        self.assertIn(obj, result)

    def test_empty_objects_list_returns_empty(self):
        find_student_objects = self._import()
        student_bbox = (0, 0, 100, 100)
        result = find_student_objects(student_bbox, [])
        self.assertEqual(result, [])

    def test_multiple_objects_some_inside(self):
        find_student_objects = self._import()
        student_bbox = (0, 0, 200, 200)
        inside = {"label": "Book", "bbox": (50, 50, 100, 100)}
        outside = {"label": "Laptop", "bbox": (500, 500, 600, 600)}
        result = find_student_objects(student_bbox, [inside, outside])
        self.assertIn(inside, result)
        self.assertNotIn(outside, result)

    def test_object_center_exactly_on_boundary(self):
        find_student_objects = self._import()
        # margin=0, center exactly at x2/y2 border
        student_bbox = (0, 0, 100, 100)
        obj = {"label": "Cell Phone", "bbox": (95, 95, 105, 105)}  # center=100,100
        result = find_student_objects(student_bbox, [obj], margin=0)
        self.assertIn(obj, result)

    def test_returns_list(self):
        find_student_objects = self._import()
        result = find_student_objects((0, 0, 100, 100), [])
        self.assertIsInstance(result, list)


class TestStudentDetectorWithMock(unittest.TestCase):
    """
    Tests for StudentDetector using a mocked YOLO model.
    """

    def _make_mock_box(self, x1, y1, x2, y2, cls_id, conf, track_id=None):
        """Helper to build a fake YOLO box."""
        import torch
        box = MagicMock()
        box.xyxy = [MagicMock()]
        box.xyxy[0].cpu.return_value.numpy.return_value = np.array([x1, y1, x2, y2])
        box.cls = [MagicMock()]
        box.cls[0].item.return_value = cls_id
        box.conf = [MagicMock()]
        box.conf[0].item.return_value = conf
        if track_id is not None:
            box.id = [MagicMock()]
            box.id[0].item.return_value = track_id
        else:
            box.id = None
        return box

    def _make_mock_result(self, boxes):
        result = MagicMock()
        result.boxes = boxes
        mock_boxes = MagicMock()
        mock_boxes.__len__ = MagicMock(return_value=len(boxes))
        mock_boxes.__iter__ = MagicMock(return_value=iter(boxes))

        # Make result.boxes[i] work
        class BoxList:
            def __init__(self, items):
                self._items = items
            def __len__(self):
                return len(self._items)
            def __getitem__(self, i):
                return self._items[i]
            def __iter__(self):
                return iter(self._items)

        result.boxes = BoxList(boxes)
        return result

    @patch("modules.detection.YOLO")
    def test_detect_returns_students_and_objects_keys(self, MockYOLO):
        from modules.detection import StudentDetector
        mock_model = MagicMock()
        mock_model.track.return_value = []
        MockYOLO.return_value = mock_model

        detector = StudentDetector(model_path="fake_model.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_and_track(frame)

        self.assertIn("students", result)
        self.assertIn("objects", result)

    @patch("modules.detection.YOLO")
    def test_detect_returns_empty_on_none_frame(self, MockYOLO):
        from modules.detection import StudentDetector
        mock_model = MagicMock()
        MockYOLO.return_value = mock_model

        detector = StudentDetector(model_path="fake_model.pt")
        result = detector.detect_and_track(None)

        self.assertEqual(result["students"], [])
        self.assertEqual(result["objects"], [])

    @patch("modules.detection.YOLO")
    def test_detect_returns_empty_when_no_results(self, MockYOLO):
        from modules.detection import StudentDetector
        mock_model = MagicMock()
        mock_model.track.return_value = []
        MockYOLO.return_value = mock_model

        detector = StudentDetector(model_path="fake_model.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_and_track(frame)

        self.assertEqual(result["students"], [])
        self.assertEqual(result["objects"], [])


if __name__ == "__main__":
    unittest.main()
