"""
test_tracker.py
---------------
Tests for modules.object_tracker.ObjectTracker.

Run the live-camera demo directly:
    python tests/test_tracker.py

Run the unit tests via pytest (from project root):
    pytest tests/test_tracker.py
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestObjectTrackerInstantiation(unittest.TestCase):
    """Smoke test — verify ObjectTracker can be imported and instantiated."""

    @patch("modules.object_tracker.YOLO")          # mock out the heavy YOLO load
    def test_instantiation(self, mock_yolo):
        from modules.object_tracker import ObjectTracker
        tracker = ObjectTracker(model_path="weights/yolov8n.pt")
        mock_yolo.assert_called_once_with("weights/yolov8n.pt")
        self.assertIsNotNone(tracker)

    @patch("modules.object_tracker.YOLO")
    def test_track_returns_list(self, mock_yolo):
        """track() should always return a list (empty when no detections)."""
        from modules.object_tracker import ObjectTracker

        # Build a fake result where boxes is None
        fake_result = MagicMock()
        fake_result.boxes = None
        mock_yolo.return_value.track.return_value = [fake_result]

        tracker = ObjectTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        students = tracker.track(frame)

        self.assertIsInstance(students, list)
        self.assertEqual(len(students), 0)


# ---------------------------------------------------------------------------
# Live camera demo — only runs when executed directly, NOT during pytest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import cv2
    from modules.object_tracker import ObjectTracker

    tracker = ObjectTracker()
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        students = tracker.track(frame)

        for student in students:
            x1, y1, x2, y2 = student["bbox"]
            sid = student["student_id"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Student {sid}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Tracker", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()