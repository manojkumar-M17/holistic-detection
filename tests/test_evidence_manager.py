"""
Tests for Evidence Manager
---------------------------
Tests for EvidenceManager: screenshot saving, video recording,
and directory creation. cv2.imwrite and VideoWriter are mocked
to avoid requiring real camera frames or disk writes.

Run:
    python -m pytest tests/test_evidence_manager.py -v
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.evidence_manager import EvidenceManager


class TestEvidenceManagerInit(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.screenshot_dir = os.path.join(self.tmp_dir, "screenshots")
        self.video_dir = os.path.join(self.tmp_dir, "videos")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_creates_screenshot_directory(self):
        EvidenceManager(
            screenshot_dir=self.screenshot_dir,
            video_dir=self.video_dir
        )
        self.assertTrue(os.path.isdir(self.screenshot_dir))

    def test_creates_video_directory(self):
        EvidenceManager(
            screenshot_dir=self.screenshot_dir,
            video_dir=self.video_dir
        )
        self.assertTrue(os.path.isdir(self.video_dir))

    def test_stores_screenshot_dir(self):
        em = EvidenceManager(
            screenshot_dir=self.screenshot_dir,
            video_dir=self.video_dir
        )
        self.assertEqual(em.screenshot_dir, self.screenshot_dir)

    def test_stores_video_dir(self):
        em = EvidenceManager(
            screenshot_dir=self.screenshot_dir,
            video_dir=self.video_dir
        )
        self.assertEqual(em.video_dir, self.video_dir)

    def test_existing_directories_do_not_raise(self):
        os.makedirs(self.screenshot_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        try:
            EvidenceManager(
                screenshot_dir=self.screenshot_dir,
                video_dir=self.video_dir
            )
        except Exception as e:
            self.fail(f"Init with existing dirs raised: {e}")


class TestSaveScreenshot(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        self.em = EvidenceManager(
            screenshot_dir=os.path.join(
                self.tmp_dir,
                "screenshots"
            ),
            video_dir=os.path.join(
                self.tmp_dir,
                "videos"
            )
        )

        self.frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        )

    def tearDown(self):
        shutil.rmtree(
            self.tmp_dir,
            ignore_errors=True
        )

    @patch("cv2.imwrite")
    def test_returns_evidence_dict(self, mock_imwrite):

        mock_imwrite.return_value = True

        result = self.em.save_screenshot(
            self.frame,
            1,
            "CELL_PHONE"
        )

        self.assertIsInstance(
            result,
            dict
        )

        self.assertIn(
            "path",
            result
        )


    @patch("cv2.imwrite")
    def test_path_contains_student_id(self, mock_imwrite):

        mock_imwrite.return_value = True

        result = self.em.save_screenshot(
            self.frame,
            7,
            "STANDING"
        )

        self.assertEqual(
            result["student_id"],
            7
        )

        self.assertIn(
            "Student_7",
            result["path"]
        )


    @patch("cv2.imwrite")
    def test_path_contains_event(self, mock_imwrite):

        mock_imwrite.return_value = True

        result = self.em.save_screenshot(
            self.frame,
            1,
            "LOOK_LEFT"
        )

        self.assertEqual(
            result["event"],
            "LOOK_LEFT"
        )

        self.assertIn(
            "LOOK_LEFT",
            result["path"]
        )


    @patch("cv2.imwrite")
    def test_path_ends_with_jpg(self, mock_imwrite):

        mock_imwrite.return_value = True

        result = self.em.save_screenshot(
            self.frame,
            1,
            "STANDING"
        )

        self.assertTrue(
            result["path"].endswith(
                ".jpg"
            )
        )


    @patch("cv2.imwrite")
    def test_path_is_inside_screenshot_dir(self, mock_imwrite):

        mock_imwrite.return_value = True

        result = self.em.save_screenshot(
            self.frame,
            1,
            "TALKING"
        )

        self.assertTrue(
            result["path"].startswith(
                self.em.screenshot_dir
            )
        )


    @patch("cv2.imwrite")
    def test_cv2_imwrite_called_with_frame(self, mock_imwrite):

        mock_imwrite.return_value = True

        self.em.save_screenshot(
            self.frame,
            1,
            "BOOK"
        )

        mock_imwrite.assert_called_once()

        call_args = mock_imwrite.call_args[0]

        self.assertIs(
            call_args[1],
            self.frame
        )


    @patch("cv2.imwrite")
    def test_filename_contains_timestamp(self, mock_imwrite):

        mock_imwrite.return_value = True

        result = self.em.save_screenshot(
            self.frame,
            1,
            "TEST"
        )

        filename = os.path.basename(
            result["path"]
        )

        import re

        self.assertRegex(
            filename,
            r"\d{8}_\d{6}"
        )


    @patch("cv2.imwrite")
    def test_evidence_contains_required_fields(
        self,
        mock_imwrite
    ):

        mock_imwrite.return_value = True

        result = self.em.save_screenshot(
            self.frame,
            1,
            "CELL_PHONE"
        )

        self.assertIn(
            "student_id",
            result
        )

        self.assertIn(
            "event",
            result
        )

        self.assertIn(
            "timestamp",
            result
        )

        self.assertIn(
            "path",
            result
        )

        self.assertIn(
            "type",
            result
        )


    @patch("cv2.imwrite")
    def test_evidence_type_is_screenshot(
        self,
        mock_imwrite
    ):

        mock_imwrite.return_value = True

        result = self.em.save_screenshot(
            self.frame,
            1,
            "BOOK"
        )

        self.assertEqual(
            result["type"],
            "SCREENSHOT"
        )
class TestStartVideoRecording(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.em = EvidenceManager(
            screenshot_dir=os.path.join(self.tmp_dir, "screenshots"),
            video_dir=os.path.join(self.tmp_dir, "videos")
        )
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch("cv2.VideoWriter")
    def test_returns_tuple_of_writer_and_path(self, MockWriter):
        MockWriter.return_value = MagicMock()
        result = self.em.start_video_recording(self.frame, 1, "CELL_PHONE")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @patch("cv2.VideoWriter")
    def test_path_ends_with_mp4(self, MockWriter):
        MockWriter.return_value = MagicMock()
        _, path = self.em.start_video_recording(self.frame, 1, "CELL_PHONE")
        self.assertTrue(path.endswith(".mp4"))

    @patch("cv2.VideoWriter")
    def test_path_contains_student_id(self, MockWriter):
        MockWriter.return_value = MagicMock()
        _, path = self.em.start_video_recording(self.frame, 3, "STANDING")
        self.assertIn("3", path)

    @patch("cv2.VideoWriter")
    def test_path_contains_event(self, MockWriter):
        MockWriter.return_value = MagicMock()
        _, path = self.em.start_video_recording(self.frame, 1, "TALKING")
        self.assertIn("TALKING", path)

    @patch("cv2.VideoWriter")
    def test_path_inside_video_dir(self, MockWriter):
        MockWriter.return_value = MagicMock()
        _, path = self.em.start_video_recording(self.frame, 1, "TEST")
        self.assertTrue(path.startswith(self.em.video_dir))

    @patch("cv2.VideoWriter")
    def test_video_writer_created_with_frame_dims(self, MockWriter):
        MockWriter.return_value = MagicMock()
        self.em.start_video_recording(self.frame, 1, "TEST")
        call_args = MockWriter.call_args[0]
        w_h = call_args[3]
        self.assertEqual(w_h, (640, 480))


class TestStopVideoRecording(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.em = EvidenceManager(
            screenshot_dir=os.path.join(self.tmp_dir, "screenshots"),
            video_dir=os.path.join(self.tmp_dir, "videos")
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_calls_release_on_writer(self):
        mock_writer = MagicMock()
        self.em.stop_video_recording(mock_writer)
        mock_writer.release.assert_called_once()

    def test_none_writer_does_not_raise(self):
        try:
            self.em.stop_video_recording(None)
        except Exception as e:
            self.fail(f"stop_video_recording(None) raised: {e}")


if __name__ == "__main__":
    unittest.main()
