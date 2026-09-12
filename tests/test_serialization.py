"""Tests for serialization of API responses and telemetry types."""

from datetime import datetime
from enum import Enum
import json
import unittest

import numpy as np

from dashboard.app import _json_safe, SharedState, app
import database.db_manager as db_manager


class StatusEnum(Enum):
    ACTIVE = "active"
    SUSPICIOUS = "suspicious"


class TestSerialization(unittest.TestCase):

    def test_json_safe_handles_numpy_integers(self):
        cases = [
            np.int8(42),
            np.int16(1000),
            np.int32(100000),
            np.int64(999999999),
            np.uint8(255),
        ]
        for val in cases:
            safe_val = _json_safe(val)
            self.assertIsInstance(safe_val, int)
            self.assertEqual(safe_val, int(val))
            # Must serialize with standard json
            self.assertEqual(json.dumps(safe_val), str(int(val)))

    def test_json_safe_handles_numpy_floats(self):
        cases = [
            np.float16(0.5),
            np.float32(3.14),
            np.float64(2.71828),
        ]
        for val in cases:
            safe_val = _json_safe(val)
            self.assertIsInstance(safe_val, float)
            # Must serialize with standard json
            self.assertIn(".", json.dumps(safe_val))

    def test_json_safe_handles_numpy_booleans(self):
        safe_true = _json_safe(np.bool_(True))
        safe_false = _json_safe(np.bool_(False))
        self.assertIs(safe_true, True)
        self.assertIs(safe_false, False)
        self.assertEqual(json.dumps(safe_true), "true")
        self.assertEqual(json.dumps(safe_false), "false")

    def test_json_safe_handles_numpy_arrays(self):
        arr = np.array([1, 2, 3])
        safe_arr = _json_safe(arr)
        self.assertEqual(safe_arr, [1, 2, 3])
        self.assertEqual(json.dumps(safe_arr), "[1, 2, 3]")

    def test_json_safe_handles_datetimes_and_enums(self):
        now = datetime(2026, 9, 12, 12, 0, 0)
        safe_dt = _json_safe(now)
        self.assertEqual(safe_dt, "2026-09-12T12:00:00")
        self.assertEqual(json.dumps(safe_dt), '"2026-09-12T12:00:00"')

        safe_enum = _json_safe(StatusEnum.ACTIVE)
        self.assertEqual(safe_enum, "active")
        self.assertEqual(json.dumps(safe_enum), '"active"')

    def test_api_stats_serializes_with_complex_telemetry(self):
        client = app.test_client()
        SharedState.student_risk_scores = {
            1: np.float64(75.5),
            2: np.int32(90),
        }
        SharedState.audio_metrics = {
            "speech_detected": np.bool_(True),
            "rms": np.float32(0.12),
            "volume_db": np.float64(-15.2),
            "waveform_snippet": np.array([0.01, 0.05, -0.02]),
        }
        response = client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["student_risk_scores"]["1"], 75.5)
        self.assertEqual(data["student_risk_scores"]["2"], 90)
        self.assertEqual(data["audio_metrics"]["speech_detected"], True)
        self.assertAlmostEqual(data["audio_metrics"]["rms"], 0.12, places=2)
        self.assertEqual(data["audio_metrics"]["waveform_snippet"], [0.01, 0.05, -0.02])
        # Verify it can be dumped without error
        serialized = json.dumps(data)
        self.assertIsInstance(serialized, str)


if __name__ == "__main__":
    unittest.main()

