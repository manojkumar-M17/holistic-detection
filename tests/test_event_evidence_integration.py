"""
Event Manager + Evidence Manager Integration Test
-------------------------------------------------
Tests the integration between EventManager and EvidenceManager.

Run:
    python -m pytest tests/test_event_evidence_integration.py -v
"""

import sys
import os
import unittest
import tempfile
import shutil
from unittest.mock import patch
import numpy as np


# Add project root to Python path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


from modules.event_manager import EventManager
from modules.evidence_manager import EvidenceManager


class TestEventEvidenceIntegration(unittest.TestCase):

    def setUp(self):

        # Temporary directory for testing
        self.tmp_dir = tempfile.mkdtemp()

        # Create Evidence Manager
        self.evidence_manager = EvidenceManager(

            screenshot_dir=os.path.join(
                self.tmp_dir,
                "screenshots"
            ),

            video_dir=os.path.join(
                self.tmp_dir,
                "videos"
            ),

            cooldown_seconds=0
        )

        # Create Event Manager
        self.event_manager = EventManager()

        # Dummy camera frame
        self.frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        )


    def tearDown(self):

        shutil.rmtree(
            self.tmp_dir,
            ignore_errors=True
        )


    # ==========================================
    # TEST 1
    # EVENT + SCREENSHOT
    # ==========================================

    @patch("cv2.imwrite")
    def test_event_with_screenshot(
        self,
        mock_imwrite
    ):

        mock_imwrite.return_value = True

        # Save evidence
        evidence = self.evidence_manager.save_screenshot(

            frame=self.frame,

            student_id=1,

            event="CELL_PHONE"
        )

        # Check evidence
        self.assertIsNotNone(evidence)

        self.assertEqual(
            evidence["student_id"],
            1
        )

        self.assertEqual(
            evidence["event"],
            "CELL_PHONE"
        )

        self.assertEqual(
            evidence["type"],
            "SCREENSHOT"
        )

        # Send event
        incident = self.event_manager.handle_event(

            student_id=1,

            event="CELL_PHONE",

            risk_score=60,

            screenshot_path=evidence["path"]
        )

        # Check incident
        self.assertEqual(
            incident["student_id"],
            1
        )

        self.assertEqual(
            incident["event"],
            "CELL_PHONE"
        )

        self.assertEqual(
            incident["risk_score"],
            60
        )

        self.assertEqual(
            incident["screenshot"],
            evidence["path"]
        )


    # ==========================================
    # TEST 2
    # STANDING EVENT
    # ==========================================

    @patch("cv2.imwrite")
    def test_standing_event_with_evidence(
        self,
        mock_imwrite
    ):

        mock_imwrite.return_value = True

        evidence = self.evidence_manager.save_screenshot(

            self.frame,

            2,

            "STANDING"
        )

        incident = self.event_manager.handle_event(

            student_id=2,

            event="STANDING",

            risk_score=25,

            screenshot_path=evidence["path"]
        )

        self.assertEqual(
            incident["event"],
            "STANDING"
        )

        self.assertEqual(
            incident["risk_score"],
            25
        )

        self.assertIsNotNone(
            incident["screenshot"]
        )


    # ==========================================
    # TEST 3
    # LOOKING AWAY EVENT
    # ==========================================

    @patch("cv2.imwrite")
    def test_looking_away_event(
        self,
        mock_imwrite
    ):

        mock_imwrite.return_value = True

        evidence = self.evidence_manager.save_screenshot(

            self.frame,

            3,

            "LOOK_RIGHT"
        )

        incident = self.event_manager.handle_event(

            student_id=3,

            event="LOOK_RIGHT",

            risk_score=8,

            screenshot_path=evidence["path"]
        )

        self.assertEqual(
            incident["student_id"],
            3
        )

        self.assertEqual(
            incident["event"],
            "LOOK_RIGHT"
        )


    # ==========================================
    # TEST 4
    # EVENT WITHOUT EVIDENCE
    # ==========================================

    def test_event_without_screenshot(self):

        incident = self.event_manager.handle_event(

            student_id=4,

            event="NORMAL",

            risk_score=0
        )

        self.assertIsNone(
            incident["screenshot"]
        )

        self.assertEqual(
            incident["event"],
            "NORMAL"
        )


    # ==========================================
    # TEST 5
    # MULTIPLE STUDENTS
    # ==========================================

    @patch("cv2.imwrite")
    def test_multiple_students_evidence(
        self,
        mock_imwrite
    ):

        mock_imwrite.return_value = True

        students = [

            (1, "LOOK_LEFT", 8),

            (2, "HAND_FACE", 20),

            (3, "CELL_PHONE", 60)

        ]

        for student_id, event, risk_score in students:

            evidence = self.evidence_manager.save_screenshot(

                self.frame,

                student_id,

                event
            )

            incident = self.event_manager.handle_event(

                student_id=student_id,

                event=event,

                risk_score=risk_score,

                screenshot_path=evidence["path"]
            )

            self.assertEqual(
                incident["student_id"],
                student_id
            )

            self.assertEqual(
                incident["event"],
                event
            )


    # ==========================================
    # TEST 6
    # VIDEO RECORDING
    # ==========================================

    @patch("cv2.VideoWriter")
    def test_video_recording_integration(
        self,
        MockWriter
    ):

        mock_writer = MockWriter.return_value

        # Important for EvidenceManager
        mock_writer.isOpened.return_value = True

        writer, video_path = (
            self.evidence_manager.start_video_recording(

                self.frame,

                1,

                "TALKING"
            )
        )

        self.assertIsNotNone(
            writer
        )

        self.assertIsNotNone(
            video_path
        )

        self.assertTrue(
            video_path.endswith(".mp4")
        )

        # Stop recording
        self.evidence_manager.stop_video_recording(
            writer
        )

        mock_writer.release.assert_called_once()


if __name__ == "__main__":

    unittest.main()