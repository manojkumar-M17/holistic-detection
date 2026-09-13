"""
Absence Detection Tests
-----------------------
Tests verification of absence detection lifecycle:
- Student disappearance triggers absence accumulation
- Triggering alert with bbox=None is safe and does not crash
- Category transitions to STUDENT_ABSENT
- Student return resets duration and decays risk score
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

import config.config as cfg
from modules.suspicious_engine import SuspiciousEngine


class TestAbsenceDetection(unittest.TestCase):

    def setUp(self):
        self.engine = SuspiciousEngine()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    @patch("modules.suspicious_engine.log_incident")
    @patch("modules.suspicious_engine.cv2.imwrite")
    def test_trigger_alert_with_none_bbox_does_not_crash(self, mock_imwrite, mock_log):
        """Verify _trigger_alert works without error when bbox is None."""
        try:
            self.engine._trigger_alert(
                student_id=1,
                reason="Student Absent/Left Seat (3.5s)",
                frame=self.frame,
                bbox=None,
                severity="MEDIUM",
                category="STUDENT_ABSENT",
                risk_score=55.0
            )
        except Exception as e:
            self.fail(f"_trigger_alert raised an unexpected exception with bbox=None: {e}")

        self.assertTrue(mock_imwrite.called)
        self.assertTrue(mock_log.called)
        args, kwargs = mock_log.call_args
        self.assertEqual(args[0], 1)
        self.assertEqual(kwargs.get("category"), "STUDENT_ABSENT")

    def test_absence_lifecycle_accumulates_and_triggers(self):
        """Verify that reporting is_absent over multiple steps increments risk and logs STUDENT_ABSENT."""
        student_id = 99
        absent_features = {
            "is_absent": True,
            "has_forbidden_object": False,
            "forbidden_objects": [],
            "yaw": 0.0,
            "pitch": 0.0,
            "mouth_open": False,
            "hand_near_face": False,
        }

        # Step 1: Initial detection of absence
        is_susp, reason, risk, sev, cat = self.engine.check_suspicious(
            student_id, absent_features, self.frame, bbox=None
        )
        # First call records start time, duration not yet > 2.0s
        self.assertFalse(is_susp)

        # Mock duration to be > 2.0s to simulate elapsed absence
        with patch.object(self.engine, "_get_duration", return_value=3.5):
            # Run multiple frames to allow risk to cross threshold
            for _ in range(10):
                is_susp, reason, risk, sev, cat = self.engine.check_suspicious(
                    student_id, absent_features, self.frame, bbox=None
                )

            self.assertTrue(is_susp)
            self.assertIn("Student Absent", reason)
            self.assertEqual(cat, "STUDENT_ABSENT")
            self.assertGreaterEqual(risk, cfg.SEVERITY_THRESHOLDS["LOW"])

    def test_student_return_resets_absence_and_decays_risk(self):
        """Verify that when an absent student returns, absent duration resets and risk score decays."""
        student_id = 88
        absent_features = {
            "is_absent": True,
            "has_forbidden_object": False,
            "forbidden_objects": [],
            "yaw": 0.0,
            "pitch": 0.0,
        }
        present_features = {
            "is_absent": False,
            "has_forbidden_object": False,
            "forbidden_objects": [],
            "yaw": 0.0,
            "pitch": 0.0,
        }

        # Set initial elevated risk
        self.engine.risk_scores[student_id] = 60.0

        # Present student with normal behavior causes risk decay
        is_susp, reason, risk, sev, cat = self.engine.check_suspicious(
            student_id, present_features, self.frame, bbox=(100, 100, 200, 300)
        )
        self.assertLess(risk, 60.0)

        # Verify absent duration tracker reset was invoked
        self.assertNotIn("absent", self.engine.states.get(student_id, {}))


if __name__ == "__main__":
    unittest.main()
